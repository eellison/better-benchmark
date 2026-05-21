class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 7, 7]", primals_2: "f32[32, 3, 224, 224]", primals_3: "i64[]", primals_4: "f32[64]", primals_5: "f32[64]", primals_6: "f32[64]", primals_7: "f32[64]", primals_8: "f32[64, 64, 1, 1]", primals_9: "i64[]", primals_10: "f32[64]", primals_11: "f32[64]", primals_12: "f32[64]", primals_13: "f32[64]", primals_14: "f32[64, 64, 3, 3]", primals_15: "i64[]", primals_16: "f32[64]", primals_17: "f32[64]", primals_18: "f32[64]", primals_19: "f32[64]", primals_20: "f32[256, 64, 1, 1]", primals_21: "i64[]", primals_22: "f32[256]", primals_23: "f32[256]", primals_24: "f32[256]", primals_25: "f32[256]", primals_26: "f32[256, 64, 1, 1]", primals_27: "i64[]", primals_28: "f32[256]", primals_29: "f32[256]", primals_30: "f32[256]", primals_31: "f32[256]", primals_32: "f32[64, 256, 1, 1]", primals_33: "i64[]", primals_34: "f32[64]", primals_35: "f32[64]", primals_36: "f32[64]", primals_37: "f32[64]", primals_38: "f32[64, 64, 3, 3]", primals_39: "i64[]", primals_40: "f32[64]", primals_41: "f32[64]", primals_42: "f32[64]", primals_43: "f32[64]", primals_44: "f32[256, 64, 1, 1]", primals_45: "i64[]", primals_46: "f32[256]", primals_47: "f32[256]", primals_48: "f32[256]", primals_49: "f32[256]", primals_50: "f32[64, 256, 1, 1]", primals_51: "i64[]", primals_52: "f32[64]", primals_53: "f32[64]", primals_54: "f32[64]", primals_55: "f32[64]", primals_56: "f32[64, 64, 3, 3]", primals_57: "i64[]", primals_58: "f32[64]", primals_59: "f32[64]", primals_60: "f32[64]", primals_61: "f32[64]", primals_62: "f32[256, 64, 1, 1]", primals_63: "i64[]", primals_64: "f32[256]", primals_65: "f32[256]", primals_66: "f32[256]", primals_67: "f32[256]", primals_68: "f32[128, 256, 1, 1]", primals_69: "i64[]", primals_70: "f32[128]", primals_71: "f32[128]", primals_72: "f32[128]", primals_73: "f32[128]", primals_74: "f32[128, 128, 3, 3]", primals_75: "i64[]", primals_76: "f32[128]", primals_77: "f32[128]", primals_78: "f32[128]", primals_79: "f32[128]", primals_80: "f32[512, 128, 1, 1]", primals_81: "i64[]", primals_82: "f32[512]", primals_83: "f32[512]", primals_84: "f32[512]", primals_85: "f32[512]", primals_86: "f32[512, 256, 1, 1]", primals_87: "i64[]", primals_88: "f32[512]", primals_89: "f32[512]", primals_90: "f32[512]", primals_91: "f32[512]", primals_92: "f32[128, 512, 1, 1]", primals_93: "i64[]", primals_94: "f32[128]", primals_95: "f32[128]", primals_96: "f32[128]", primals_97: "f32[128]", primals_98: "f32[128, 128, 3, 3]", primals_99: "i64[]", primals_100: "f32[128]", primals_101: "f32[128]", primals_102: "f32[128]", primals_103: "f32[128]", primals_104: "f32[512, 128, 1, 1]", primals_105: "i64[]", primals_106: "f32[512]", primals_107: "f32[512]", primals_108: "f32[512]", primals_109: "f32[512]", primals_110: "f32[128, 512, 1, 1]", primals_111: "i64[]", primals_112: "f32[128]", primals_113: "f32[128]", primals_114: "f32[128]", primals_115: "f32[128]", primals_116: "f32[128, 128, 3, 3]", primals_117: "i64[]", primals_118: "f32[128]", primals_119: "f32[128]", primals_120: "f32[128]", primals_121: "f32[128]", primals_122: "f32[512, 128, 1, 1]", primals_123: "i64[]", primals_124: "f32[512]", primals_125: "f32[512]", primals_126: "f32[512]", primals_127: "f32[512]", primals_128: "f32[128, 512, 1, 1]", primals_129: "i64[]", primals_130: "f32[128]", primals_131: "f32[128]", primals_132: "f32[128]", primals_133: "f32[128]", primals_134: "f32[128, 128, 3, 3]", primals_135: "i64[]", primals_136: "f32[128]", primals_137: "f32[128]", primals_138: "f32[128]", primals_139: "f32[128]", primals_140: "f32[512, 128, 1, 1]", primals_141: "i64[]", primals_142: "f32[512]", primals_143: "f32[512]", primals_144: "f32[512]", primals_145: "f32[512]", primals_146: "f32[128, 512, 1, 1]", primals_147: "i64[]", primals_148: "f32[128]", primals_149: "f32[128]", primals_150: "f32[128]", primals_151: "f32[128]", primals_152: "f32[128, 128, 3, 3]", primals_153: "i64[]", primals_154: "f32[128]", primals_155: "f32[128]", primals_156: "f32[128]", primals_157: "f32[128]", primals_158: "f32[512, 128, 1, 1]", primals_159: "i64[]", primals_160: "f32[512]", primals_161: "f32[512]", primals_162: "f32[512]", primals_163: "f32[512]", primals_164: "f32[128, 512, 1, 1]", primals_165: "i64[]", primals_166: "f32[128]", primals_167: "f32[128]", primals_168: "f32[128]", primals_169: "f32[128]", primals_170: "f32[128, 128, 3, 3]", primals_171: "i64[]", primals_172: "f32[128]", primals_173: "f32[128]", primals_174: "f32[128]", primals_175: "f32[128]", primals_176: "f32[512, 128, 1, 1]", primals_177: "i64[]", primals_178: "f32[512]", primals_179: "f32[512]", primals_180: "f32[512]", primals_181: "f32[512]", primals_182: "f32[128, 512, 1, 1]", primals_183: "i64[]", primals_184: "f32[128]", primals_185: "f32[128]", primals_186: "f32[128]", primals_187: "f32[128]", primals_188: "f32[128, 128, 3, 3]", primals_189: "i64[]", primals_190: "f32[128]", primals_191: "f32[128]", primals_192: "f32[128]", primals_193: "f32[128]", primals_194: "f32[512, 128, 1, 1]", primals_195: "i64[]", primals_196: "f32[512]", primals_197: "f32[512]", primals_198: "f32[512]", primals_199: "f32[512]", primals_200: "f32[128, 512, 1, 1]", primals_201: "i64[]", primals_202: "f32[128]", primals_203: "f32[128]", primals_204: "f32[128]", primals_205: "f32[128]", primals_206: "f32[128, 128, 3, 3]", primals_207: "i64[]", primals_208: "f32[128]", primals_209: "f32[128]", primals_210: "f32[128]", primals_211: "f32[128]", primals_212: "f32[512, 128, 1, 1]", primals_213: "i64[]", primals_214: "f32[512]", primals_215: "f32[512]", primals_216: "f32[512]", primals_217: "f32[512]", primals_218: "f32[256, 512, 1, 1]", primals_219: "i64[]", primals_220: "f32[256]", primals_221: "f32[256]", primals_222: "f32[256]", primals_223: "f32[256]", primals_224: "f32[256, 256, 3, 3]", primals_225: "i64[]", primals_226: "f32[256]", primals_227: "f32[256]", primals_228: "f32[256]", primals_229: "f32[256]", primals_230: "f32[1024, 256, 1, 1]", primals_231: "i64[]", primals_232: "f32[1024]", primals_233: "f32[1024]", primals_234: "f32[1024]", primals_235: "f32[1024]", primals_236: "f32[1024, 512, 1, 1]", primals_237: "i64[]", primals_238: "f32[1024]", primals_239: "f32[1024]", primals_240: "f32[1024]", primals_241: "f32[1024]", primals_242: "f32[256, 1024, 1, 1]", primals_243: "i64[]", primals_244: "f32[256]", primals_245: "f32[256]", primals_246: "f32[256]", primals_247: "f32[256]", primals_248: "f32[256, 256, 3, 3]", primals_249: "i64[]", primals_250: "f32[256]", primals_251: "f32[256]", primals_252: "f32[256]", primals_253: "f32[256]", primals_254: "f32[1024, 256, 1, 1]", primals_255: "i64[]", primals_256: "f32[1024]", primals_257: "f32[1024]", primals_258: "f32[1024]", primals_259: "f32[1024]", primals_260: "f32[256, 1024, 1, 1]", primals_261: "i64[]", primals_262: "f32[256]", primals_263: "f32[256]", primals_264: "f32[256]", primals_265: "f32[256]", primals_266: "f32[256, 256, 3, 3]", primals_267: "i64[]", primals_268: "f32[256]", primals_269: "f32[256]", primals_270: "f32[256]", primals_271: "f32[256]", primals_272: "f32[1024, 256, 1, 1]", primals_273: "i64[]", primals_274: "f32[1024]", primals_275: "f32[1024]", primals_276: "f32[1024]", primals_277: "f32[1024]", primals_278: "f32[256, 1024, 1, 1]", primals_279: "i64[]", primals_280: "f32[256]", primals_281: "f32[256]", primals_282: "f32[256]", primals_283: "f32[256]", primals_284: "f32[256, 256, 3, 3]", primals_285: "i64[]", primals_286: "f32[256]", primals_287: "f32[256]", primals_288: "f32[256]", primals_289: "f32[256]", primals_290: "f32[1024, 256, 1, 1]", primals_291: "i64[]", primals_292: "f32[1024]", primals_293: "f32[1024]", primals_294: "f32[1024]", primals_295: "f32[1024]", primals_296: "f32[256, 1024, 1, 1]", primals_297: "i64[]", primals_298: "f32[256]", primals_299: "f32[256]", primals_300: "f32[256]", primals_301: "f32[256]", primals_302: "f32[256, 256, 3, 3]", primals_303: "i64[]", primals_304: "f32[256]", primals_305: "f32[256]", primals_306: "f32[256]", primals_307: "f32[256]", primals_308: "f32[1024, 256, 1, 1]", primals_309: "i64[]", primals_310: "f32[1024]", primals_311: "f32[1024]", primals_312: "f32[1024]", primals_313: "f32[1024]", primals_314: "f32[256, 1024, 1, 1]", primals_315: "i64[]", primals_316: "f32[256]", primals_317: "f32[256]", primals_318: "f32[256]", primals_319: "f32[256]", primals_320: "f32[256, 256, 3, 3]", primals_321: "i64[]", primals_322: "f32[256]", primals_323: "f32[256]", primals_324: "f32[256]", primals_325: "f32[256]", primals_326: "f32[1024, 256, 1, 1]", primals_327: "i64[]", primals_328: "f32[1024]", primals_329: "f32[1024]", primals_330: "f32[1024]", primals_331: "f32[1024]", primals_332: "f32[256, 1024, 1, 1]", primals_333: "i64[]", primals_334: "f32[256]", primals_335: "f32[256]", primals_336: "f32[256]", primals_337: "f32[256]", primals_338: "f32[256, 256, 3, 3]", primals_339: "i64[]", primals_340: "f32[256]", primals_341: "f32[256]", primals_342: "f32[256]", primals_343: "f32[256]", primals_344: "f32[1024, 256, 1, 1]", primals_345: "i64[]", primals_346: "f32[1024]", primals_347: "f32[1024]", primals_348: "f32[1024]", primals_349: "f32[1024]", primals_350: "f32[256, 1024, 1, 1]", primals_351: "i64[]", primals_352: "f32[256]", primals_353: "f32[256]", primals_354: "f32[256]", primals_355: "f32[256]", primals_356: "f32[256, 256, 3, 3]", primals_357: "i64[]", primals_358: "f32[256]", primals_359: "f32[256]", primals_360: "f32[256]", primals_361: "f32[256]", primals_362: "f32[1024, 256, 1, 1]", primals_363: "i64[]", primals_364: "f32[1024]", primals_365: "f32[1024]", primals_366: "f32[1024]", primals_367: "f32[1024]", primals_368: "f32[256, 1024, 1, 1]", primals_369: "i64[]", primals_370: "f32[256]", primals_371: "f32[256]", primals_372: "f32[256]", primals_373: "f32[256]", primals_374: "f32[256, 256, 3, 3]", primals_375: "i64[]", primals_376: "f32[256]", primals_377: "f32[256]", primals_378: "f32[256]", primals_379: "f32[256]", primals_380: "f32[1024, 256, 1, 1]", primals_381: "i64[]", primals_382: "f32[1024]", primals_383: "f32[1024]", primals_384: "f32[1024]", primals_385: "f32[1024]", primals_386: "f32[256, 1024, 1, 1]", primals_387: "i64[]", primals_388: "f32[256]", primals_389: "f32[256]", primals_390: "f32[256]", primals_391: "f32[256]", primals_392: "f32[256, 256, 3, 3]", primals_393: "i64[]", primals_394: "f32[256]", primals_395: "f32[256]", primals_396: "f32[256]", primals_397: "f32[256]", primals_398: "f32[1024, 256, 1, 1]", primals_399: "i64[]", primals_400: "f32[1024]", primals_401: "f32[1024]", primals_402: "f32[1024]", primals_403: "f32[1024]", primals_404: "f32[256, 1024, 1, 1]", primals_405: "i64[]", primals_406: "f32[256]", primals_407: "f32[256]", primals_408: "f32[256]", primals_409: "f32[256]", primals_410: "f32[256, 256, 3, 3]", primals_411: "i64[]", primals_412: "f32[256]", primals_413: "f32[256]", primals_414: "f32[256]", primals_415: "f32[256]", primals_416: "f32[1024, 256, 1, 1]", primals_417: "i64[]", primals_418: "f32[1024]", primals_419: "f32[1024]", primals_420: "f32[1024]", primals_421: "f32[1024]", primals_422: "f32[256, 1024, 1, 1]", primals_423: "i64[]", primals_424: "f32[256]", primals_425: "f32[256]", primals_426: "f32[256]", primals_427: "f32[256]", primals_428: "f32[256, 256, 3, 3]", primals_429: "i64[]", primals_430: "f32[256]", primals_431: "f32[256]", primals_432: "f32[256]", primals_433: "f32[256]", primals_434: "f32[1024, 256, 1, 1]", primals_435: "i64[]", primals_436: "f32[1024]", primals_437: "f32[1024]", primals_438: "f32[1024]", primals_439: "f32[1024]", primals_440: "f32[256, 1024, 1, 1]", primals_441: "i64[]", primals_442: "f32[256]", primals_443: "f32[256]", primals_444: "f32[256]", primals_445: "f32[256]", primals_446: "f32[256, 256, 3, 3]", primals_447: "i64[]", primals_448: "f32[256]", primals_449: "f32[256]", primals_450: "f32[256]", primals_451: "f32[256]", primals_452: "f32[1024, 256, 1, 1]", primals_453: "i64[]", primals_454: "f32[1024]", primals_455: "f32[1024]", primals_456: "f32[1024]", primals_457: "f32[1024]", primals_458: "f32[256, 1024, 1, 1]", primals_459: "i64[]", primals_460: "f32[256]", primals_461: "f32[256]", primals_462: "f32[256]", primals_463: "f32[256]", primals_464: "f32[256, 256, 3, 3]", primals_465: "i64[]", primals_466: "f32[256]", primals_467: "f32[256]", primals_468: "f32[256]", primals_469: "f32[256]", primals_470: "f32[1024, 256, 1, 1]", primals_471: "i64[]", primals_472: "f32[1024]", primals_473: "f32[1024]", primals_474: "f32[1024]", primals_475: "f32[1024]", primals_476: "f32[256, 1024, 1, 1]", primals_477: "i64[]", primals_478: "f32[256]", primals_479: "f32[256]", primals_480: "f32[256]", primals_481: "f32[256]", primals_482: "f32[256, 256, 3, 3]", primals_483: "i64[]", primals_484: "f32[256]", primals_485: "f32[256]", primals_486: "f32[256]", primals_487: "f32[256]", primals_488: "f32[1024, 256, 1, 1]", primals_489: "i64[]", primals_490: "f32[1024]", primals_491: "f32[1024]", primals_492: "f32[1024]", primals_493: "f32[1024]", primals_494: "f32[256, 1024, 1, 1]", primals_495: "i64[]", primals_496: "f32[256]", primals_497: "f32[256]", primals_498: "f32[256]", primals_499: "f32[256]", primals_500: "f32[256, 256, 3, 3]", primals_501: "i64[]", primals_502: "f32[256]", primals_503: "f32[256]", primals_504: "f32[256]", primals_505: "f32[256]", primals_506: "f32[1024, 256, 1, 1]", primals_507: "i64[]", primals_508: "f32[1024]", primals_509: "f32[1024]", primals_510: "f32[1024]", primals_511: "f32[1024]", primals_512: "f32[256, 1024, 1, 1]", primals_513: "i64[]", primals_514: "f32[256]", primals_515: "f32[256]", primals_516: "f32[256]", primals_517: "f32[256]", primals_518: "f32[256, 256, 3, 3]", primals_519: "i64[]", primals_520: "f32[256]", primals_521: "f32[256]", primals_522: "f32[256]", primals_523: "f32[256]", primals_524: "f32[1024, 256, 1, 1]", primals_525: "i64[]", primals_526: "f32[1024]", primals_527: "f32[1024]", primals_528: "f32[1024]", primals_529: "f32[1024]", primals_530: "f32[256, 1024, 1, 1]", primals_531: "i64[]", primals_532: "f32[256]", primals_533: "f32[256]", primals_534: "f32[256]", primals_535: "f32[256]", primals_536: "f32[256, 256, 3, 3]", primals_537: "i64[]", primals_538: "f32[256]", primals_539: "f32[256]", primals_540: "f32[256]", primals_541: "f32[256]", primals_542: "f32[1024, 256, 1, 1]", primals_543: "i64[]", primals_544: "f32[1024]", primals_545: "f32[1024]", primals_546: "f32[1024]", primals_547: "f32[1024]", primals_548: "f32[256, 1024, 1, 1]", primals_549: "i64[]", primals_550: "f32[256]", primals_551: "f32[256]", primals_552: "f32[256]", primals_553: "f32[256]", primals_554: "f32[256, 256, 3, 3]", primals_555: "i64[]", primals_556: "f32[256]", primals_557: "f32[256]", primals_558: "f32[256]", primals_559: "f32[256]", primals_560: "f32[1024, 256, 1, 1]", primals_561: "i64[]", primals_562: "f32[1024]", primals_563: "f32[1024]", primals_564: "f32[1024]", primals_565: "f32[1024]", primals_566: "f32[256, 1024, 1, 1]", primals_567: "i64[]", primals_568: "f32[256]", primals_569: "f32[256]", primals_570: "f32[256]", primals_571: "f32[256]", primals_572: "f32[256, 256, 3, 3]", primals_573: "i64[]", primals_574: "f32[256]", primals_575: "f32[256]", primals_576: "f32[256]", primals_577: "f32[256]", primals_578: "f32[1024, 256, 1, 1]", primals_579: "i64[]", primals_580: "f32[1024]", primals_581: "f32[1024]", primals_582: "f32[1024]", primals_583: "f32[1024]", primals_584: "f32[256, 1024, 1, 1]", primals_585: "i64[]", primals_586: "f32[256]", primals_587: "f32[256]", primals_588: "f32[256]", primals_589: "f32[256]", primals_590: "f32[256, 256, 3, 3]", primals_591: "i64[]", primals_592: "f32[256]", primals_593: "f32[256]", primals_594: "f32[256]", primals_595: "f32[256]", primals_596: "f32[1024, 256, 1, 1]", primals_597: "i64[]", primals_598: "f32[1024]", primals_599: "f32[1024]", primals_600: "f32[1024]", primals_601: "f32[1024]", primals_602: "f32[256, 1024, 1, 1]", primals_603: "i64[]", primals_604: "f32[256]", primals_605: "f32[256]", primals_606: "f32[256]", primals_607: "f32[256]", primals_608: "f32[256, 256, 3, 3]", primals_609: "i64[]", primals_610: "f32[256]", primals_611: "f32[256]", primals_612: "f32[256]", primals_613: "f32[256]", primals_614: "f32[1024, 256, 1, 1]", primals_615: "i64[]", primals_616: "f32[1024]", primals_617: "f32[1024]", primals_618: "f32[1024]", primals_619: "f32[1024]", primals_620: "f32[256, 1024, 1, 1]", primals_621: "i64[]", primals_622: "f32[256]", primals_623: "f32[256]", primals_624: "f32[256]", primals_625: "f32[256]", primals_626: "f32[256, 256, 3, 3]", primals_627: "i64[]", primals_628: "f32[256]", primals_629: "f32[256]", primals_630: "f32[256]", primals_631: "f32[256]", primals_632: "f32[1024, 256, 1, 1]", primals_633: "i64[]", primals_634: "f32[1024]", primals_635: "f32[1024]", primals_636: "f32[1024]", primals_637: "f32[1024]", primals_638: "f32[256, 1024, 1, 1]", primals_639: "i64[]", primals_640: "f32[256]", primals_641: "f32[256]", primals_642: "f32[256]", primals_643: "f32[256]", primals_644: "f32[256, 256, 3, 3]", primals_645: "i64[]", primals_646: "f32[256]", primals_647: "f32[256]", primals_648: "f32[256]", primals_649: "f32[256]", primals_650: "f32[1024, 256, 1, 1]", primals_651: "i64[]", primals_652: "f32[1024]", primals_653: "f32[1024]", primals_654: "f32[1024]", primals_655: "f32[1024]", primals_656: "f32[256, 1024, 1, 1]", primals_657: "i64[]", primals_658: "f32[256]", primals_659: "f32[256]", primals_660: "f32[256]", primals_661: "f32[256]", primals_662: "f32[256, 256, 3, 3]", primals_663: "i64[]", primals_664: "f32[256]", primals_665: "f32[256]", primals_666: "f32[256]", primals_667: "f32[256]", primals_668: "f32[1024, 256, 1, 1]", primals_669: "i64[]", primals_670: "f32[1024]", primals_671: "f32[1024]", primals_672: "f32[1024]", primals_673: "f32[1024]", primals_674: "f32[256, 1024, 1, 1]", primals_675: "i64[]", primals_676: "f32[256]", primals_677: "f32[256]", primals_678: "f32[256]", primals_679: "f32[256]", primals_680: "f32[256, 256, 3, 3]", primals_681: "i64[]", primals_682: "f32[256]", primals_683: "f32[256]", primals_684: "f32[256]", primals_685: "f32[256]", primals_686: "f32[1024, 256, 1, 1]", primals_687: "i64[]", primals_688: "f32[1024]", primals_689: "f32[1024]", primals_690: "f32[1024]", primals_691: "f32[1024]", primals_692: "f32[256, 1024, 1, 1]", primals_693: "i64[]", primals_694: "f32[256]", primals_695: "f32[256]", primals_696: "f32[256]", primals_697: "f32[256]", primals_698: "f32[256, 256, 3, 3]", primals_699: "i64[]", primals_700: "f32[256]", primals_701: "f32[256]", primals_702: "f32[256]", primals_703: "f32[256]", primals_704: "f32[1024, 256, 1, 1]", primals_705: "i64[]", primals_706: "f32[1024]", primals_707: "f32[1024]", primals_708: "f32[1024]", primals_709: "f32[1024]", primals_710: "f32[256, 1024, 1, 1]", primals_711: "i64[]", primals_712: "f32[256]", primals_713: "f32[256]", primals_714: "f32[256]", primals_715: "f32[256]", primals_716: "f32[256, 256, 3, 3]", primals_717: "i64[]", primals_718: "f32[256]", primals_719: "f32[256]", primals_720: "f32[256]", primals_721: "f32[256]", primals_722: "f32[1024, 256, 1, 1]", primals_723: "i64[]", primals_724: "f32[1024]", primals_725: "f32[1024]", primals_726: "f32[1024]", primals_727: "f32[1024]", primals_728: "f32[256, 1024, 1, 1]", primals_729: "i64[]", primals_730: "f32[256]", primals_731: "f32[256]", primals_732: "f32[256]", primals_733: "f32[256]", primals_734: "f32[256, 256, 3, 3]", primals_735: "i64[]", primals_736: "f32[256]", primals_737: "f32[256]", primals_738: "f32[256]", primals_739: "f32[256]", primals_740: "f32[1024, 256, 1, 1]", primals_741: "i64[]", primals_742: "f32[1024]", primals_743: "f32[1024]", primals_744: "f32[1024]", primals_745: "f32[1024]", primals_746: "f32[256, 1024, 1, 1]", primals_747: "i64[]", primals_748: "f32[256]", primals_749: "f32[256]", primals_750: "f32[256]", primals_751: "f32[256]", primals_752: "f32[256, 256, 3, 3]", primals_753: "i64[]", primals_754: "f32[256]", primals_755: "f32[256]", primals_756: "f32[256]", primals_757: "f32[256]", primals_758: "f32[1024, 256, 1, 1]", primals_759: "i64[]", primals_760: "f32[1024]", primals_761: "f32[1024]", primals_762: "f32[1024]", primals_763: "f32[1024]", primals_764: "f32[256, 1024, 1, 1]", primals_765: "i64[]", primals_766: "f32[256]", primals_767: "f32[256]", primals_768: "f32[256]", primals_769: "f32[256]", primals_770: "f32[256, 256, 3, 3]", primals_771: "i64[]", primals_772: "f32[256]", primals_773: "f32[256]", primals_774: "f32[256]", primals_775: "f32[256]", primals_776: "f32[1024, 256, 1, 1]", primals_777: "i64[]", primals_778: "f32[1024]", primals_779: "f32[1024]", primals_780: "f32[1024]", primals_781: "f32[1024]", primals_782: "f32[256, 1024, 1, 1]", primals_783: "i64[]", primals_784: "f32[256]", primals_785: "f32[256]", primals_786: "f32[256]", primals_787: "f32[256]", primals_788: "f32[256, 256, 3, 3]", primals_789: "i64[]", primals_790: "f32[256]", primals_791: "f32[256]", primals_792: "f32[256]", primals_793: "f32[256]", primals_794: "f32[1024, 256, 1, 1]", primals_795: "i64[]", primals_796: "f32[1024]", primals_797: "f32[1024]", primals_798: "f32[1024]", primals_799: "f32[1024]", primals_800: "f32[256, 1024, 1, 1]", primals_801: "i64[]", primals_802: "f32[256]", primals_803: "f32[256]", primals_804: "f32[256]", primals_805: "f32[256]", primals_806: "f32[256, 256, 3, 3]", primals_807: "i64[]", primals_808: "f32[256]", primals_809: "f32[256]", primals_810: "f32[256]", primals_811: "f32[256]", primals_812: "f32[1024, 256, 1, 1]", primals_813: "i64[]", primals_814: "f32[1024]", primals_815: "f32[1024]", primals_816: "f32[1024]", primals_817: "f32[1024]", primals_818: "f32[256, 1024, 1, 1]", primals_819: "i64[]", primals_820: "f32[256]", primals_821: "f32[256]", primals_822: "f32[256]", primals_823: "f32[256]", primals_824: "f32[256, 256, 3, 3]", primals_825: "i64[]", primals_826: "f32[256]", primals_827: "f32[256]", primals_828: "f32[256]", primals_829: "f32[256]", primals_830: "f32[1024, 256, 1, 1]", primals_831: "i64[]", primals_832: "f32[1024]", primals_833: "f32[1024]", primals_834: "f32[1024]", primals_835: "f32[1024]", primals_836: "f32[256, 1024, 1, 1]", primals_837: "i64[]", primals_838: "f32[256]", primals_839: "f32[256]", primals_840: "f32[256]", primals_841: "f32[256]", primals_842: "f32[256, 256, 3, 3]", primals_843: "i64[]", primals_844: "f32[256]", primals_845: "f32[256]", primals_846: "f32[256]", primals_847: "f32[256]", primals_848: "f32[1024, 256, 1, 1]", primals_849: "i64[]", primals_850: "f32[1024]", primals_851: "f32[1024]", primals_852: "f32[1024]", primals_853: "f32[1024]", primals_854: "f32[256, 1024, 1, 1]", primals_855: "i64[]", primals_856: "f32[256]", primals_857: "f32[256]", primals_858: "f32[256]", primals_859: "f32[256]", primals_860: "f32[256, 256, 3, 3]", primals_861: "i64[]", primals_862: "f32[256]", primals_863: "f32[256]", primals_864: "f32[256]", primals_865: "f32[256]", primals_866: "f32[1024, 256, 1, 1]", primals_867: "i64[]", primals_868: "f32[1024]", primals_869: "f32[1024]", primals_870: "f32[1024]", primals_871: "f32[1024]", primals_872: "f32[512, 1024, 1, 1]", primals_873: "i64[]", primals_874: "f32[512]", primals_875: "f32[512]", primals_876: "f32[512]", primals_877: "f32[512]", primals_878: "f32[512, 512, 3, 3]", primals_879: "i64[]", primals_880: "f32[512]", primals_881: "f32[512]", primals_882: "f32[512]", primals_883: "f32[512]", primals_884: "f32[2048, 512, 1, 1]", primals_885: "i64[]", primals_886: "f32[2048]", primals_887: "f32[2048]", primals_888: "f32[2048]", primals_889: "f32[2048]", primals_890: "f32[2048, 1024, 1, 1]", primals_891: "i64[]", primals_892: "f32[2048]", primals_893: "f32[2048]", primals_894: "f32[2048]", primals_895: "f32[2048]", primals_896: "f32[512, 2048, 1, 1]", primals_897: "i64[]", primals_898: "f32[512]", primals_899: "f32[512]", primals_900: "f32[512]", primals_901: "f32[512]", primals_902: "f32[512, 512, 3, 3]", primals_903: "i64[]", primals_904: "f32[512]", primals_905: "f32[512]", primals_906: "f32[512]", primals_907: "f32[512]", primals_908: "f32[2048, 512, 1, 1]", primals_909: "i64[]", primals_910: "f32[2048]", primals_911: "f32[2048]", primals_912: "f32[2048]", primals_913: "f32[2048]", primals_914: "f32[512, 2048, 1, 1]", primals_915: "i64[]", primals_916: "f32[512]", primals_917: "f32[512]", primals_918: "f32[512]", primals_919: "f32[512]", primals_920: "f32[512, 512, 3, 3]", primals_921: "i64[]", primals_922: "f32[512]", primals_923: "f32[512]", primals_924: "f32[512]", primals_925: "f32[512]", primals_926: "f32[2048, 512, 1, 1]", primals_927: "i64[]", primals_928: "f32[2048]", primals_929: "f32[2048]", primals_930: "f32[2048]", primals_931: "f32[2048]", primals_932: "f32[1000, 2048]", primals_933: "f32[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:268 in _forward_impl, code: x = self.conv1(x)
        convolution: "f32[32, 64, 112, 112]" = torch.ops.aten.convolution.default(primals_2, primals_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        add: "i64[]" = torch.ops.aten.add.Tensor(primals_3, 1)
        var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 64, 1, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[32, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000024912370735);  squeeze_2 = None
        mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3: "f32[64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
        unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[32, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:270 in _forward_impl, code: x = self.relu(x)
        relu: "f32[32, 64, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:271 in _forward_impl, code: x = self.maxpool(x)
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu = None
        getitem_2: "f32[32, 64, 56, 56]" = _low_memory_max_pool_with_offsets[0]
        getitem_3: "i8[32, 64, 56, 56]" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_1: "f32[32, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem_2, primals_8, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_9, 1)
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 64, 1, 1]" = var_mean_1[0]
        getitem_5: "f32[1, 64, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_1: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[32, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_5)
        mul_7: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
        mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[32, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_1: "f32[32, 64, 56, 56]" = torch.ops.aten.relu.default(add_9);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_2: "f32[32, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_14, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_15, 1)
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6: "f32[1, 64, 1, 1]" = var_mean_2[0]
        getitem_7: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2: "f32[32, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_7)
        mul_14: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_16: "f32[64]" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_12: "f32[64]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
        mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[32, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_2: "f32[32, 64, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_3: "f32[32, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_2, primals_20, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_21, 1)
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8: "f32[1, 256, 1, 1]" = var_mean_3[0]
        getitem_9: "f32[1, 256, 1, 1]" = var_mean_3[1];  var_mean_3 = None
        add_16: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_3: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_3: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
        mul_21: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_10: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_23: "f32[256]" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_17: "f32[256]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_24: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.00000996502277);  squeeze_11 = None
        mul_25: "f32[256]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "f32[256]" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_18: "f32[256]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1)
        unsqueeze_13: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
        unsqueeze_15: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_19: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convolution_4: "f32[32, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_2, primals_26, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_27, 1)
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10: "f32[1, 256, 1, 1]" = var_mean_4[0]
        getitem_11: "f32[1, 256, 1, 1]" = var_mean_4[1];  var_mean_4 = None
        add_21: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_4: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_4: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
        mul_28: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_13: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_29: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_30: "f32[256]" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_22: "f32[256]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_31: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.00000996502277);  squeeze_14 = None
        mul_32: "f32[256]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33: "f32[256]" = torch.ops.aten.mul.Tensor(primals_29, 0.9)
        add_23: "f32[256]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_19: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_24: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_25: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_19, add_24);  add_19 = add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_3: "f32[32, 256, 56, 56]" = torch.ops.aten.relu.default(add_25);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_5: "f32[32, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_32, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_26: "i64[]" = torch.ops.aten.add.Tensor(primals_33, 1)
        var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem_12: "f32[1, 64, 1, 1]" = var_mean_5[0]
        getitem_13: "f32[1, 64, 1, 1]" = var_mean_5[1];  var_mean_5 = None
        add_27: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_5: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_5: "f32[32, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_13)
        mul_35: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_16: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_36: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_37: "f32[64]" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_28: "f32[64]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_38: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.00000996502277);  squeeze_17 = None
        mul_39: "f32[64]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40: "f32[64]" = torch.ops.aten.mul.Tensor(primals_35, 0.9)
        add_29: "f32[64]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
        unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_30: "f32[32, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_4: "f32[32, 64, 56, 56]" = torch.ops.aten.relu.default(add_30);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_6: "f32[32, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_38, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_31: "i64[]" = torch.ops.aten.add.Tensor(primals_39, 1)
        var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
        getitem_14: "f32[1, 64, 1, 1]" = var_mean_6[0]
        getitem_15: "f32[1, 64, 1, 1]" = var_mean_6[1];  var_mean_6 = None
        add_32: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_6: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_6: "f32[32, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_15)
        mul_42: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_19: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_44: "f32[64]" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_33: "f32[64]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_45: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.00000996502277);  squeeze_20 = None
        mul_46: "f32[64]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47: "f32[64]" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_34: "f32[64]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_48: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        unsqueeze_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_35: "f32[32, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_5: "f32[32, 64, 56, 56]" = torch.ops.aten.relu.default(add_35);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_7: "f32[32, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_44, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_45, 1)
        var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
        getitem_16: "f32[1, 256, 1, 1]" = var_mean_7[0]
        getitem_17: "f32[1, 256, 1, 1]" = var_mean_7[1];  var_mean_7 = None
        add_37: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_7: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_7: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_17)
        mul_49: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_22: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_50: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_51: "f32[256]" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_38: "f32[256]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_52: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.00000996502277);  squeeze_23 = None
        mul_53: "f32[256]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54: "f32[256]" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_39: "f32[256]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1);  primals_49 = None
        unsqueeze_31: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_40: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_41: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_40, relu_3);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_6: "f32[32, 256, 56, 56]" = torch.ops.aten.relu.default(add_41);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_8: "f32[32, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_6, primals_50, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_42: "i64[]" = torch.ops.aten.add.Tensor(primals_51, 1)
        var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
        getitem_18: "f32[1, 64, 1, 1]" = var_mean_8[0]
        getitem_19: "f32[1, 64, 1, 1]" = var_mean_8[1];  var_mean_8 = None
        add_43: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_8: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_8: "f32[32, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_19)
        mul_56: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_25: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_57: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_58: "f32[64]" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_44: "f32[64]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_59: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.00000996502277);  squeeze_26 = None
        mul_60: "f32[64]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61: "f32[64]" = torch.ops.aten.mul.Tensor(primals_53, 0.9)
        add_45: "f32[64]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1);  primals_55 = None
        unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_46: "f32[32, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_7: "f32[32, 64, 56, 56]" = torch.ops.aten.relu.default(add_46);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_9: "f32[32, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_7, primals_56, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_47: "i64[]" = torch.ops.aten.add.Tensor(primals_57, 1)
        var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
        getitem_20: "f32[1, 64, 1, 1]" = var_mean_9[0]
        getitem_21: "f32[1, 64, 1, 1]" = var_mean_9[1];  var_mean_9 = None
        add_48: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_9: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_9: "f32[32, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_21)
        mul_63: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_28: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_64: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_65: "f32[64]" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_49: "f32[64]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_66: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.00000996502277);  squeeze_29 = None
        mul_67: "f32[64]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68: "f32[64]" = torch.ops.aten.mul.Tensor(primals_59, 0.9)
        add_50: "f32[64]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1)
        unsqueeze_37: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_69: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        unsqueeze_38: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
        unsqueeze_39: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_51: "f32[32, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_8: "f32[32, 64, 56, 56]" = torch.ops.aten.relu.default(add_51);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_10: "f32[32, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_8, primals_62, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_52: "i64[]" = torch.ops.aten.add.Tensor(primals_63, 1)
        var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
        getitem_22: "f32[1, 256, 1, 1]" = var_mean_10[0]
        getitem_23: "f32[1, 256, 1, 1]" = var_mean_10[1];  var_mean_10 = None
        add_53: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_10: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_10: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_23)
        mul_70: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_31: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_71: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_72: "f32[256]" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_54: "f32[256]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        squeeze_32: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_73: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.00000996502277);  squeeze_32 = None
        mul_74: "f32[256]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
        mul_75: "f32[256]" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_55: "f32[256]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        unsqueeze_40: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_41: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
        unsqueeze_43: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_56: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_57: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_56, relu_6);  add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_9: "f32[32, 256, 56, 56]" = torch.ops.aten.relu.default(add_57);  add_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_11: "f32[32, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_9, primals_68, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_58: "i64[]" = torch.ops.aten.add.Tensor(primals_69, 1)
        var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
        getitem_24: "f32[1, 128, 1, 1]" = var_mean_11[0]
        getitem_25: "f32[1, 128, 1, 1]" = var_mean_11[1];  var_mean_11 = None
        add_59: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_11: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_11: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_25)
        mul_77: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_34: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_78: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_79: "f32[128]" = torch.ops.aten.mul.Tensor(primals_70, 0.9)
        add_60: "f32[128]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        squeeze_35: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_80: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.00000996502277);  squeeze_35 = None
        mul_81: "f32[128]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
        mul_82: "f32[128]" = torch.ops.aten.mul.Tensor(primals_71, 0.9)
        add_61: "f32[128]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        unsqueeze_44: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_45: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1);  primals_73 = None
        unsqueeze_47: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_62: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_10: "f32[32, 128, 56, 56]" = torch.ops.aten.relu.default(add_62);  add_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_12: "f32[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_10, primals_74, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_63: "i64[]" = torch.ops.aten.add.Tensor(primals_75, 1)
        var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
        getitem_26: "f32[1, 128, 1, 1]" = var_mean_12[0]
        getitem_27: "f32[1, 128, 1, 1]" = var_mean_12[1];  var_mean_12 = None
        add_64: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_12: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_12: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_27)
        mul_84: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_37: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_85: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_86: "f32[128]" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_65: "f32[128]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        squeeze_38: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_87: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000398612827361);  squeeze_38 = None
        mul_88: "f32[128]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
        mul_89: "f32[128]" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_66: "f32[128]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_48: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_49: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_90: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        unsqueeze_50: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
        unsqueeze_51: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_67: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_11: "f32[32, 128, 28, 28]" = torch.ops.aten.relu.default(add_67);  add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_13: "f32[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_11, primals_80, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_68: "i64[]" = torch.ops.aten.add.Tensor(primals_81, 1)
        var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
        getitem_28: "f32[1, 512, 1, 1]" = var_mean_13[0]
        getitem_29: "f32[1, 512, 1, 1]" = var_mean_13[1];  var_mean_13 = None
        add_69: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_13: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_13: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_29)
        mul_91: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_40: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_92: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
        mul_93: "f32[512]" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_70: "f32[512]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        squeeze_41: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_94: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000398612827361);  squeeze_41 = None
        mul_95: "f32[512]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
        mul_96: "f32[512]" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_71: "f32[512]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        unsqueeze_52: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_53: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
        unsqueeze_55: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_72: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convolution_14: "f32[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_9, primals_86, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_87, 1)
        var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
        getitem_30: "f32[1, 512, 1, 1]" = var_mean_14[0]
        getitem_31: "f32[1, 512, 1, 1]" = var_mean_14[1];  var_mean_14 = None
        add_74: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_14: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_14: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_31)
        mul_98: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_43: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_99: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_100: "f32[512]" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_75: "f32[512]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        squeeze_44: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_101: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000398612827361);  squeeze_44 = None
        mul_102: "f32[512]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
        mul_103: "f32[512]" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_76: "f32[512]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        unsqueeze_56: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_57: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
        unsqueeze_59: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_77: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_78: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_72, add_77);  add_72 = add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_12: "f32[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_78);  add_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_15: "f32[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_12, primals_92, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_79: "i64[]" = torch.ops.aten.add.Tensor(primals_93, 1)
        var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
        getitem_32: "f32[1, 128, 1, 1]" = var_mean_15[0]
        getitem_33: "f32[1, 128, 1, 1]" = var_mean_15[1];  var_mean_15 = None
        add_80: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_15: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_15: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_33)
        mul_105: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_46: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_106: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_107: "f32[128]" = torch.ops.aten.mul.Tensor(primals_94, 0.9)
        add_81: "f32[128]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_47: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_108: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000398612827361);  squeeze_47 = None
        mul_109: "f32[128]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110: "f32[128]" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_82: "f32[128]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_60: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1)
        unsqueeze_61: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_111: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        unsqueeze_62: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1);  primals_97 = None
        unsqueeze_63: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_83: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_13: "f32[32, 128, 28, 28]" = torch.ops.aten.relu.default(add_83);  add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_16: "f32[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_13, primals_98, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_84: "i64[]" = torch.ops.aten.add.Tensor(primals_99, 1)
        var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
        getitem_34: "f32[1, 128, 1, 1]" = var_mean_16[0]
        getitem_35: "f32[1, 128, 1, 1]" = var_mean_16[1];  var_mean_16 = None
        add_85: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_16: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_16: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_35)
        mul_112: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_49: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_113: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_114: "f32[128]" = torch.ops.aten.mul.Tensor(primals_100, 0.9)
        add_86: "f32[128]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_50: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_115: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000398612827361);  squeeze_50 = None
        mul_116: "f32[128]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117: "f32[128]" = torch.ops.aten.mul.Tensor(primals_101, 0.9)
        add_87: "f32[128]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_64: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1)
        unsqueeze_65: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
        unsqueeze_67: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_88: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_14: "f32[32, 128, 28, 28]" = torch.ops.aten.relu.default(add_88);  add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_17: "f32[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_14, primals_104, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_89: "i64[]" = torch.ops.aten.add.Tensor(primals_105, 1)
        var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
        getitem_36: "f32[1, 512, 1, 1]" = var_mean_17[0]
        getitem_37: "f32[1, 512, 1, 1]" = var_mean_17[1];  var_mean_17 = None
        add_90: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_17: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_17: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_37)
        mul_119: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_52: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_120: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_121: "f32[512]" = torch.ops.aten.mul.Tensor(primals_106, 0.9)
        add_91: "f32[512]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
        squeeze_53: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_122: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0000398612827361);  squeeze_53 = None
        mul_123: "f32[512]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
        mul_124: "f32[512]" = torch.ops.aten.mul.Tensor(primals_107, 0.9)
        add_92: "f32[512]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        unsqueeze_68: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1)
        unsqueeze_69: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_125: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
        unsqueeze_70: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
        unsqueeze_71: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_93: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_94: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_93, relu_12);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_15: "f32[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_94);  add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_18: "f32[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_15, primals_110, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_95: "i64[]" = torch.ops.aten.add.Tensor(primals_111, 1)
        var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
        getitem_38: "f32[1, 128, 1, 1]" = var_mean_18[0]
        getitem_39: "f32[1, 128, 1, 1]" = var_mean_18[1];  var_mean_18 = None
        add_96: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
        rsqrt_18: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_18: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_39)
        mul_126: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_55: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_127: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
        mul_128: "f32[128]" = torch.ops.aten.mul.Tensor(primals_112, 0.9)
        add_97: "f32[128]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
        squeeze_56: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_129: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0000398612827361);  squeeze_56 = None
        mul_130: "f32[128]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
        mul_131: "f32[128]" = torch.ops.aten.mul.Tensor(primals_113, 0.9)
        add_98: "f32[128]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        unsqueeze_72: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1)
        unsqueeze_73: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_132: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
        unsqueeze_74: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
        unsqueeze_75: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_99: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_16: "f32[32, 128, 28, 28]" = torch.ops.aten.relu.default(add_99);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_19: "f32[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_16, primals_116, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_100: "i64[]" = torch.ops.aten.add.Tensor(primals_117, 1)
        var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
        getitem_40: "f32[1, 128, 1, 1]" = var_mean_19[0]
        getitem_41: "f32[1, 128, 1, 1]" = var_mean_19[1];  var_mean_19 = None
        add_101: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
        rsqrt_19: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_19: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_41)
        mul_133: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_58: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_134: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
        mul_135: "f32[128]" = torch.ops.aten.mul.Tensor(primals_118, 0.9)
        add_102: "f32[128]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        squeeze_59: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_136: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0000398612827361);  squeeze_59 = None
        mul_137: "f32[128]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
        mul_138: "f32[128]" = torch.ops.aten.mul.Tensor(primals_119, 0.9)
        add_103: "f32[128]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        unsqueeze_76: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1)
        unsqueeze_77: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_139: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        unsqueeze_78: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
        unsqueeze_79: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_104: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_17: "f32[32, 128, 28, 28]" = torch.ops.aten.relu.default(add_104);  add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_20: "f32[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_17, primals_122, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_105: "i64[]" = torch.ops.aten.add.Tensor(primals_123, 1)
        var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem_42: "f32[1, 512, 1, 1]" = var_mean_20[0]
        getitem_43: "f32[1, 512, 1, 1]" = var_mean_20[1];  var_mean_20 = None
        add_106: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
        rsqrt_20: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_20: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_43)
        mul_140: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_60: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_61: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_141: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_142: "f32[512]" = torch.ops.aten.mul.Tensor(primals_124, 0.9)
        add_107: "f32[512]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
        squeeze_62: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_143: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0000398612827361);  squeeze_62 = None
        mul_144: "f32[512]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
        mul_145: "f32[512]" = torch.ops.aten.mul.Tensor(primals_125, 0.9)
        add_108: "f32[512]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        unsqueeze_80: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1)
        unsqueeze_81: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_146: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
        unsqueeze_82: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1);  primals_127 = None
        unsqueeze_83: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_109: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_110: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_109, relu_15);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_18: "f32[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_110);  add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_21: "f32[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_18, primals_128, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_111: "i64[]" = torch.ops.aten.add.Tensor(primals_129, 1)
        var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
        getitem_44: "f32[1, 128, 1, 1]" = var_mean_21[0]
        getitem_45: "f32[1, 128, 1, 1]" = var_mean_21[1];  var_mean_21 = None
        add_112: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
        rsqrt_21: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        sub_21: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_45)
        mul_147: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        squeeze_63: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_64: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_148: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
        mul_149: "f32[128]" = torch.ops.aten.mul.Tensor(primals_130, 0.9)
        add_113: "f32[128]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
        squeeze_65: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_150: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0000398612827361);  squeeze_65 = None
        mul_151: "f32[128]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
        mul_152: "f32[128]" = torch.ops.aten.mul.Tensor(primals_131, 0.9)
        add_114: "f32[128]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        unsqueeze_84: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1)
        unsqueeze_85: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_153: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
        unsqueeze_86: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_133, -1);  primals_133 = None
        unsqueeze_87: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_115: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_19: "f32[32, 128, 28, 28]" = torch.ops.aten.relu.default(add_115);  add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_22: "f32[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_19, primals_134, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_116: "i64[]" = torch.ops.aten.add.Tensor(primals_135, 1)
        var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
        getitem_46: "f32[1, 128, 1, 1]" = var_mean_22[0]
        getitem_47: "f32[1, 128, 1, 1]" = var_mean_22[1];  var_mean_22 = None
        add_117: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
        rsqrt_22: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        sub_22: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_47)
        mul_154: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_66: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_67: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_155: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
        mul_156: "f32[128]" = torch.ops.aten.mul.Tensor(primals_136, 0.9)
        add_118: "f32[128]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        squeeze_68: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_157: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0000398612827361);  squeeze_68 = None
        mul_158: "f32[128]" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
        mul_159: "f32[128]" = torch.ops.aten.mul.Tensor(primals_137, 0.9)
        add_119: "f32[128]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        unsqueeze_88: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1)
        unsqueeze_89: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_160: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
        unsqueeze_90: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_139, -1);  primals_139 = None
        unsqueeze_91: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_120: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_20: "f32[32, 128, 28, 28]" = torch.ops.aten.relu.default(add_120);  add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_23: "f32[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_20, primals_140, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_121: "i64[]" = torch.ops.aten.add.Tensor(primals_141, 1)
        var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
        getitem_48: "f32[1, 512, 1, 1]" = var_mean_23[0]
        getitem_49: "f32[1, 512, 1, 1]" = var_mean_23[1];  var_mean_23 = None
        add_122: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
        rsqrt_23: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_23: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_49)
        mul_161: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_69: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_70: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_162: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
        mul_163: "f32[512]" = torch.ops.aten.mul.Tensor(primals_142, 0.9)
        add_123: "f32[512]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        squeeze_71: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_164: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0000398612827361);  squeeze_71 = None
        mul_165: "f32[512]" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
        mul_166: "f32[512]" = torch.ops.aten.mul.Tensor(primals_143, 0.9)
        add_124: "f32[512]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        unsqueeze_92: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1)
        unsqueeze_93: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_167: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
        unsqueeze_94: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_145, -1);  primals_145 = None
        unsqueeze_95: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_125: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_126: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_125, relu_18);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_21: "f32[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_126);  add_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_24: "f32[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_21, primals_146, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_127: "i64[]" = torch.ops.aten.add.Tensor(primals_147, 1)
        var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
        getitem_50: "f32[1, 128, 1, 1]" = var_mean_24[0]
        getitem_51: "f32[1, 128, 1, 1]" = var_mean_24[1];  var_mean_24 = None
        add_128: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
        rsqrt_24: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        sub_24: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_51)
        mul_168: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_72: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_73: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_169: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
        mul_170: "f32[128]" = torch.ops.aten.mul.Tensor(primals_148, 0.9)
        add_129: "f32[128]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        squeeze_74: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_171: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0000398612827361);  squeeze_74 = None
        mul_172: "f32[128]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
        mul_173: "f32[128]" = torch.ops.aten.mul.Tensor(primals_149, 0.9)
        add_130: "f32[128]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        unsqueeze_96: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1)
        unsqueeze_97: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_174: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
        unsqueeze_98: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_151, -1);  primals_151 = None
        unsqueeze_99: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_131: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_22: "f32[32, 128, 28, 28]" = torch.ops.aten.relu.default(add_131);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_25: "f32[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_22, primals_152, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_132: "i64[]" = torch.ops.aten.add.Tensor(primals_153, 1)
        var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
        getitem_52: "f32[1, 128, 1, 1]" = var_mean_25[0]
        getitem_53: "f32[1, 128, 1, 1]" = var_mean_25[1];  var_mean_25 = None
        add_133: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
        rsqrt_25: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        sub_25: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_53)
        mul_175: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_75: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_76: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_176: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
        mul_177: "f32[128]" = torch.ops.aten.mul.Tensor(primals_154, 0.9)
        add_134: "f32[128]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
        squeeze_77: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_178: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0000398612827361);  squeeze_77 = None
        mul_179: "f32[128]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
        mul_180: "f32[128]" = torch.ops.aten.mul.Tensor(primals_155, 0.9)
        add_135: "f32[128]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        unsqueeze_100: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1)
        unsqueeze_101: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_181: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
        unsqueeze_102: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_157, -1);  primals_157 = None
        unsqueeze_103: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_136: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_23: "f32[32, 128, 28, 28]" = torch.ops.aten.relu.default(add_136);  add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_26: "f32[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_23, primals_158, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_137: "i64[]" = torch.ops.aten.add.Tensor(primals_159, 1)
        var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
        getitem_54: "f32[1, 512, 1, 1]" = var_mean_26[0]
        getitem_55: "f32[1, 512, 1, 1]" = var_mean_26[1];  var_mean_26 = None
        add_138: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
        rsqrt_26: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_26: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_55)
        mul_182: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_78: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_79: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_183: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
        mul_184: "f32[512]" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
        add_139: "f32[512]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
        squeeze_80: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_185: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0000398612827361);  squeeze_80 = None
        mul_186: "f32[512]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
        mul_187: "f32[512]" = torch.ops.aten.mul.Tensor(primals_161, 0.9)
        add_140: "f32[512]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
        unsqueeze_104: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1)
        unsqueeze_105: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_188: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
        unsqueeze_106: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_163, -1);  primals_163 = None
        unsqueeze_107: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_141: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_142: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_141, relu_21);  add_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_24: "f32[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_142);  add_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_27: "f32[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_24, primals_164, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_143: "i64[]" = torch.ops.aten.add.Tensor(primals_165, 1)
        var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
        getitem_56: "f32[1, 128, 1, 1]" = var_mean_27[0]
        getitem_57: "f32[1, 128, 1, 1]" = var_mean_27[1];  var_mean_27 = None
        add_144: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
        rsqrt_27: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        sub_27: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_57)
        mul_189: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_81: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        squeeze_82: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_190: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
        mul_191: "f32[128]" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
        add_145: "f32[128]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        squeeze_83: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_192: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0000398612827361);  squeeze_83 = None
        mul_193: "f32[128]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
        mul_194: "f32[128]" = torch.ops.aten.mul.Tensor(primals_167, 0.9)
        add_146: "f32[128]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
        unsqueeze_108: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_168, -1)
        unsqueeze_109: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_195: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
        unsqueeze_110: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_169, -1);  primals_169 = None
        unsqueeze_111: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_147: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_25: "f32[32, 128, 28, 28]" = torch.ops.aten.relu.default(add_147);  add_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_28: "f32[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_25, primals_170, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_148: "i64[]" = torch.ops.aten.add.Tensor(primals_171, 1)
        var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
        getitem_58: "f32[1, 128, 1, 1]" = var_mean_28[0]
        getitem_59: "f32[1, 128, 1, 1]" = var_mean_28[1];  var_mean_28 = None
        add_149: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
        rsqrt_28: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        sub_28: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_59)
        mul_196: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        squeeze_84: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_85: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_197: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
        mul_198: "f32[128]" = torch.ops.aten.mul.Tensor(primals_172, 0.9)
        add_150: "f32[128]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        squeeze_86: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_199: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0000398612827361);  squeeze_86 = None
        mul_200: "f32[128]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
        mul_201: "f32[128]" = torch.ops.aten.mul.Tensor(primals_173, 0.9)
        add_151: "f32[128]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        unsqueeze_112: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_174, -1)
        unsqueeze_113: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_202: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
        unsqueeze_114: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_175, -1);  primals_175 = None
        unsqueeze_115: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_152: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_26: "f32[32, 128, 28, 28]" = torch.ops.aten.relu.default(add_152);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_29: "f32[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_26, primals_176, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_153: "i64[]" = torch.ops.aten.add.Tensor(primals_177, 1)
        var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
        getitem_60: "f32[1, 512, 1, 1]" = var_mean_29[0]
        getitem_61: "f32[1, 512, 1, 1]" = var_mean_29[1];  var_mean_29 = None
        add_154: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
        rsqrt_29: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_29: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_61)
        mul_203: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        squeeze_87: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_88: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_204: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
        mul_205: "f32[512]" = torch.ops.aten.mul.Tensor(primals_178, 0.9)
        add_155: "f32[512]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
        squeeze_89: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_206: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0000398612827361);  squeeze_89 = None
        mul_207: "f32[512]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
        mul_208: "f32[512]" = torch.ops.aten.mul.Tensor(primals_179, 0.9)
        add_156: "f32[512]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
        unsqueeze_116: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_180, -1)
        unsqueeze_117: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_209: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
        unsqueeze_118: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_181, -1);  primals_181 = None
        unsqueeze_119: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_157: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_158: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_157, relu_24);  add_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_27: "f32[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_158);  add_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_30: "f32[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_27, primals_182, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_159: "i64[]" = torch.ops.aten.add.Tensor(primals_183, 1)
        var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
        getitem_62: "f32[1, 128, 1, 1]" = var_mean_30[0]
        getitem_63: "f32[1, 128, 1, 1]" = var_mean_30[1];  var_mean_30 = None
        add_160: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
        rsqrt_30: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        sub_30: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_63)
        mul_210: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        squeeze_90: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_91: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_211: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
        mul_212: "f32[128]" = torch.ops.aten.mul.Tensor(primals_184, 0.9)
        add_161: "f32[128]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        squeeze_92: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_213: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0000398612827361);  squeeze_92 = None
        mul_214: "f32[128]" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
        mul_215: "f32[128]" = torch.ops.aten.mul.Tensor(primals_185, 0.9)
        add_162: "f32[128]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        unsqueeze_120: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_186, -1)
        unsqueeze_121: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_216: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
        unsqueeze_122: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_187, -1);  primals_187 = None
        unsqueeze_123: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_163: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_28: "f32[32, 128, 28, 28]" = torch.ops.aten.relu.default(add_163);  add_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_31: "f32[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_28, primals_188, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_164: "i64[]" = torch.ops.aten.add.Tensor(primals_189, 1)
        var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
        getitem_64: "f32[1, 128, 1, 1]" = var_mean_31[0]
        getitem_65: "f32[1, 128, 1, 1]" = var_mean_31[1];  var_mean_31 = None
        add_165: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
        rsqrt_31: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_31: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_65)
        mul_217: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        squeeze_93: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_94: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_218: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
        mul_219: "f32[128]" = torch.ops.aten.mul.Tensor(primals_190, 0.9)
        add_166: "f32[128]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
        squeeze_95: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_220: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0000398612827361);  squeeze_95 = None
        mul_221: "f32[128]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
        mul_222: "f32[128]" = torch.ops.aten.mul.Tensor(primals_191, 0.9)
        add_167: "f32[128]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        unsqueeze_124: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_192, -1)
        unsqueeze_125: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_223: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
        unsqueeze_126: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_193, -1);  primals_193 = None
        unsqueeze_127: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_168: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_29: "f32[32, 128, 28, 28]" = torch.ops.aten.relu.default(add_168);  add_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_32: "f32[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_29, primals_194, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_169: "i64[]" = torch.ops.aten.add.Tensor(primals_195, 1)
        var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
        getitem_66: "f32[1, 512, 1, 1]" = var_mean_32[0]
        getitem_67: "f32[1, 512, 1, 1]" = var_mean_32[1];  var_mean_32 = None
        add_170: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
        rsqrt_32: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_32: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_67)
        mul_224: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        squeeze_96: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_97: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_225: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
        mul_226: "f32[512]" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
        add_171: "f32[512]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        squeeze_98: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_227: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0000398612827361);  squeeze_98 = None
        mul_228: "f32[512]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
        mul_229: "f32[512]" = torch.ops.aten.mul.Tensor(primals_197, 0.9)
        add_172: "f32[512]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        unsqueeze_128: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_198, -1)
        unsqueeze_129: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        mul_230: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
        unsqueeze_130: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_199, -1);  primals_199 = None
        unsqueeze_131: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        add_173: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_174: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_173, relu_27);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_30: "f32[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_174);  add_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_33: "f32[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_30, primals_200, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_175: "i64[]" = torch.ops.aten.add.Tensor(primals_201, 1)
        var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
        getitem_68: "f32[1, 128, 1, 1]" = var_mean_33[0]
        getitem_69: "f32[1, 128, 1, 1]" = var_mean_33[1];  var_mean_33 = None
        add_176: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
        rsqrt_33: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
        sub_33: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_69)
        mul_231: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        squeeze_99: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_100: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_232: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
        mul_233: "f32[128]" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
        add_177: "f32[128]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        squeeze_101: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_234: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0000398612827361);  squeeze_101 = None
        mul_235: "f32[128]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
        mul_236: "f32[128]" = torch.ops.aten.mul.Tensor(primals_203, 0.9)
        add_178: "f32[128]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
        unsqueeze_132: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_204, -1)
        unsqueeze_133: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_237: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
        unsqueeze_134: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_205, -1);  primals_205 = None
        unsqueeze_135: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_179: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_31: "f32[32, 128, 28, 28]" = torch.ops.aten.relu.default(add_179);  add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_34: "f32[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_31, primals_206, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_180: "i64[]" = torch.ops.aten.add.Tensor(primals_207, 1)
        var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
        getitem_70: "f32[1, 128, 1, 1]" = var_mean_34[0]
        getitem_71: "f32[1, 128, 1, 1]" = var_mean_34[1];  var_mean_34 = None
        add_181: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
        rsqrt_34: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_34: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_71)
        mul_238: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        squeeze_102: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_103: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_239: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
        mul_240: "f32[128]" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
        add_182: "f32[128]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
        squeeze_104: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_241: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0000398612827361);  squeeze_104 = None
        mul_242: "f32[128]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
        mul_243: "f32[128]" = torch.ops.aten.mul.Tensor(primals_209, 0.9)
        add_183: "f32[128]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        unsqueeze_136: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_210, -1)
        unsqueeze_137: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_244: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
        unsqueeze_138: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_211, -1);  primals_211 = None
        unsqueeze_139: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_184: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_32: "f32[32, 128, 28, 28]" = torch.ops.aten.relu.default(add_184);  add_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_35: "f32[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_32, primals_212, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_185: "i64[]" = torch.ops.aten.add.Tensor(primals_213, 1)
        var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
        getitem_72: "f32[1, 512, 1, 1]" = var_mean_35[0]
        getitem_73: "f32[1, 512, 1, 1]" = var_mean_35[1];  var_mean_35 = None
        add_186: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
        rsqrt_35: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_35: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_73)
        mul_245: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        squeeze_105: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_106: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_246: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
        mul_247: "f32[512]" = torch.ops.aten.mul.Tensor(primals_214, 0.9)
        add_187: "f32[512]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        squeeze_107: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_248: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0000398612827361);  squeeze_107 = None
        mul_249: "f32[512]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
        mul_250: "f32[512]" = torch.ops.aten.mul.Tensor(primals_215, 0.9)
        add_188: "f32[512]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
        unsqueeze_140: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_216, -1)
        unsqueeze_141: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_251: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
        unsqueeze_142: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_217, -1);  primals_217 = None
        unsqueeze_143: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_189: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_190: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_189, relu_30);  add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_33: "f32[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_190);  add_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_36: "f32[32, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_33, primals_218, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_191: "i64[]" = torch.ops.aten.add.Tensor(primals_219, 1)
        var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
        getitem_74: "f32[1, 256, 1, 1]" = var_mean_36[0]
        getitem_75: "f32[1, 256, 1, 1]" = var_mean_36[1];  var_mean_36 = None
        add_192: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
        rsqrt_36: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        sub_36: "f32[32, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_75)
        mul_252: "f32[32, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        squeeze_108: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_109: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_253: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
        mul_254: "f32[256]" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
        add_193: "f32[256]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
        squeeze_110: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_255: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0000398612827361);  squeeze_110 = None
        mul_256: "f32[256]" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
        mul_257: "f32[256]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
        add_194: "f32[256]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        unsqueeze_144: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_222, -1)
        unsqueeze_145: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        mul_258: "f32[32, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
        unsqueeze_146: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_223, -1);  primals_223 = None
        unsqueeze_147: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        add_195: "f32[32, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_34: "f32[32, 256, 28, 28]" = torch.ops.aten.relu.default(add_195);  add_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_37: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_34, primals_224, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_196: "i64[]" = torch.ops.aten.add.Tensor(primals_225, 1)
        var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
        getitem_76: "f32[1, 256, 1, 1]" = var_mean_37[0]
        getitem_77: "f32[1, 256, 1, 1]" = var_mean_37[1];  var_mean_37 = None
        add_197: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
        rsqrt_37: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        sub_37: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_77)
        mul_259: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_111: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_112: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_260: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
        mul_261: "f32[256]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
        add_198: "f32[256]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
        squeeze_113: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_262: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0001594642002871);  squeeze_113 = None
        mul_263: "f32[256]" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
        mul_264: "f32[256]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
        add_199: "f32[256]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
        unsqueeze_148: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_228, -1)
        unsqueeze_149: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_265: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
        unsqueeze_150: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_229, -1);  primals_229 = None
        unsqueeze_151: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_200: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_35: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_200);  add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_38: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_35, primals_230, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_201: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)
        var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
        getitem_78: "f32[1, 1024, 1, 1]" = var_mean_38[0]
        getitem_79: "f32[1, 1024, 1, 1]" = var_mean_38[1];  var_mean_38 = None
        add_202: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
        rsqrt_38: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
        sub_38: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_79)
        mul_266: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_114: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_115: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_267: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
        mul_268: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
        add_203: "f32[1024]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
        squeeze_116: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_269: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0001594642002871);  squeeze_116 = None
        mul_270: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
        mul_271: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
        add_204: "f32[1024]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
        unsqueeze_152: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_234, -1)
        unsqueeze_153: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        mul_272: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
        unsqueeze_154: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_235, -1);  primals_235 = None
        unsqueeze_155: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        add_205: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convolution_39: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_33, primals_236, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_206: "i64[]" = torch.ops.aten.add.Tensor(primals_237, 1)
        var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
        getitem_80: "f32[1, 1024, 1, 1]" = var_mean_39[0]
        getitem_81: "f32[1, 1024, 1, 1]" = var_mean_39[1];  var_mean_39 = None
        add_207: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
        rsqrt_39: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        sub_39: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_81)
        mul_273: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        squeeze_117: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_118: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_274: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
        mul_275: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
        add_208: "f32[1024]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
        squeeze_119: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_276: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0001594642002871);  squeeze_119 = None
        mul_277: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
        mul_278: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
        add_209: "f32[1024]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        unsqueeze_156: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_240, -1)
        unsqueeze_157: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_279: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
        unsqueeze_158: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_241, -1);  primals_241 = None
        unsqueeze_159: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_210: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_211: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_205, add_210);  add_205 = add_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_36: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_211);  add_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_40: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_36, primals_242, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_243, 1)
        var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
        getitem_82: "f32[1, 256, 1, 1]" = var_mean_40[0]
        getitem_83: "f32[1, 256, 1, 1]" = var_mean_40[1];  var_mean_40 = None
        add_213: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
        rsqrt_40: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        sub_40: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_83)
        mul_280: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        squeeze_120: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_121: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_281: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
        mul_282: "f32[256]" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
        add_214: "f32[256]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
        squeeze_122: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_283: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0001594642002871);  squeeze_122 = None
        mul_284: "f32[256]" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
        mul_285: "f32[256]" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
        add_215: "f32[256]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
        unsqueeze_160: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_246, -1)
        unsqueeze_161: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_286: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
        unsqueeze_162: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_247, -1);  primals_247 = None
        unsqueeze_163: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_216: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_37: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_216);  add_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_41: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_37, primals_248, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_217: "i64[]" = torch.ops.aten.add.Tensor(primals_249, 1)
        var_mean_41 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
        getitem_84: "f32[1, 256, 1, 1]" = var_mean_41[0]
        getitem_85: "f32[1, 256, 1, 1]" = var_mean_41[1];  var_mean_41 = None
        add_218: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
        rsqrt_41: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        sub_41: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_85)
        mul_287: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        squeeze_123: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_124: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_288: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
        mul_289: "f32[256]" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
        add_219: "f32[256]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
        squeeze_125: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_290: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0001594642002871);  squeeze_125 = None
        mul_291: "f32[256]" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
        mul_292: "f32[256]" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
        add_220: "f32[256]" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
        unsqueeze_164: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_252, -1)
        unsqueeze_165: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_293: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
        unsqueeze_166: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_253, -1);  primals_253 = None
        unsqueeze_167: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_221: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_38: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_221);  add_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_42: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_38, primals_254, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_222: "i64[]" = torch.ops.aten.add.Tensor(primals_255, 1)
        var_mean_42 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
        getitem_86: "f32[1, 1024, 1, 1]" = var_mean_42[0]
        getitem_87: "f32[1, 1024, 1, 1]" = var_mean_42[1];  var_mean_42 = None
        add_223: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
        rsqrt_42: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
        sub_42: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_87)
        mul_294: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        squeeze_126: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        squeeze_127: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_295: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
        mul_296: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
        add_224: "f32[1024]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
        squeeze_128: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_297: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0001594642002871);  squeeze_128 = None
        mul_298: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
        mul_299: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
        add_225: "f32[1024]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
        unsqueeze_168: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_258, -1)
        unsqueeze_169: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        mul_300: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
        unsqueeze_170: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_259, -1);  primals_259 = None
        unsqueeze_171: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        add_226: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_227: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_226, relu_36);  add_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_39: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_227);  add_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_43: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_39, primals_260, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_228: "i64[]" = torch.ops.aten.add.Tensor(primals_261, 1)
        var_mean_43 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
        getitem_88: "f32[1, 256, 1, 1]" = var_mean_43[0]
        getitem_89: "f32[1, 256, 1, 1]" = var_mean_43[1];  var_mean_43 = None
        add_229: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
        rsqrt_43: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_43: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_89)
        mul_301: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        squeeze_129: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_130: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_302: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
        mul_303: "f32[256]" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
        add_230: "f32[256]" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
        squeeze_131: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_304: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0001594642002871);  squeeze_131 = None
        mul_305: "f32[256]" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
        mul_306: "f32[256]" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
        add_231: "f32[256]" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
        unsqueeze_172: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_264, -1)
        unsqueeze_173: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_307: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
        unsqueeze_174: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_265, -1);  primals_265 = None
        unsqueeze_175: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_232: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_40: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_232);  add_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_44: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_40, primals_266, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_233: "i64[]" = torch.ops.aten.add.Tensor(primals_267, 1)
        var_mean_44 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
        getitem_90: "f32[1, 256, 1, 1]" = var_mean_44[0]
        getitem_91: "f32[1, 256, 1, 1]" = var_mean_44[1];  var_mean_44 = None
        add_234: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
        rsqrt_44: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
        sub_44: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_91)
        mul_308: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        squeeze_132: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_133: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_309: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
        mul_310: "f32[256]" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
        add_235: "f32[256]" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
        squeeze_134: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_311: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0001594642002871);  squeeze_134 = None
        mul_312: "f32[256]" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
        mul_313: "f32[256]" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
        add_236: "f32[256]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
        unsqueeze_176: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_270, -1)
        unsqueeze_177: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        mul_314: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
        unsqueeze_178: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_271, -1);  primals_271 = None
        unsqueeze_179: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        add_237: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_41: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_237);  add_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_45: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_41, primals_272, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_238: "i64[]" = torch.ops.aten.add.Tensor(primals_273, 1)
        var_mean_45 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
        getitem_92: "f32[1, 1024, 1, 1]" = var_mean_45[0]
        getitem_93: "f32[1, 1024, 1, 1]" = var_mean_45[1];  var_mean_45 = None
        add_239: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
        rsqrt_45: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        sub_45: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_93)
        mul_315: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        squeeze_135: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        squeeze_136: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_316: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
        mul_317: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
        add_240: "f32[1024]" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
        squeeze_137: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_318: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0001594642002871);  squeeze_137 = None
        mul_319: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
        mul_320: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
        add_241: "f32[1024]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
        unsqueeze_180: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_276, -1)
        unsqueeze_181: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_321: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
        unsqueeze_182: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_277, -1);  primals_277 = None
        unsqueeze_183: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_242: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_243: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_242, relu_39);  add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_42: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_243);  add_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_46: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_42, primals_278, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_244: "i64[]" = torch.ops.aten.add.Tensor(primals_279, 1)
        var_mean_46 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
        getitem_94: "f32[1, 256, 1, 1]" = var_mean_46[0]
        getitem_95: "f32[1, 256, 1, 1]" = var_mean_46[1];  var_mean_46 = None
        add_245: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
        rsqrt_46: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_245);  add_245 = None
        sub_46: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_95)
        mul_322: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        squeeze_138: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        squeeze_139: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_323: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
        mul_324: "f32[256]" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
        add_246: "f32[256]" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
        squeeze_140: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_325: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0001594642002871);  squeeze_140 = None
        mul_326: "f32[256]" = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
        mul_327: "f32[256]" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
        add_247: "f32[256]" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
        unsqueeze_184: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_282, -1)
        unsqueeze_185: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        mul_328: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
        unsqueeze_186: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_283, -1);  primals_283 = None
        unsqueeze_187: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        add_248: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_43: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_248);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_47: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_43, primals_284, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_249: "i64[]" = torch.ops.aten.add.Tensor(primals_285, 1)
        var_mean_47 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
        getitem_96: "f32[1, 256, 1, 1]" = var_mean_47[0]
        getitem_97: "f32[1, 256, 1, 1]" = var_mean_47[1];  var_mean_47 = None
        add_250: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
        rsqrt_47: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
        sub_47: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_97)
        mul_329: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        squeeze_141: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        squeeze_142: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_330: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
        mul_331: "f32[256]" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
        add_251: "f32[256]" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
        squeeze_143: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_332: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0001594642002871);  squeeze_143 = None
        mul_333: "f32[256]" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
        mul_334: "f32[256]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
        add_252: "f32[256]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
        unsqueeze_188: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_288, -1)
        unsqueeze_189: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_335: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
        unsqueeze_190: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_289, -1);  primals_289 = None
        unsqueeze_191: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_253: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_44: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_253);  add_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_48: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_44, primals_290, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_254: "i64[]" = torch.ops.aten.add.Tensor(primals_291, 1)
        var_mean_48 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
        getitem_98: "f32[1, 1024, 1, 1]" = var_mean_48[0]
        getitem_99: "f32[1, 1024, 1, 1]" = var_mean_48[1];  var_mean_48 = None
        add_255: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
        rsqrt_48: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        sub_48: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_99)
        mul_336: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        squeeze_144: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_145: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_337: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
        mul_338: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
        add_256: "f32[1024]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
        squeeze_146: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_339: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0001594642002871);  squeeze_146 = None
        mul_340: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
        mul_341: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
        add_257: "f32[1024]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        unsqueeze_192: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_294, -1)
        unsqueeze_193: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        mul_342: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
        unsqueeze_194: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_295, -1);  primals_295 = None
        unsqueeze_195: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        add_258: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_259: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_258, relu_42);  add_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_45: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_259);  add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_49: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_45, primals_296, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_260: "i64[]" = torch.ops.aten.add.Tensor(primals_297, 1)
        var_mean_49 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
        getitem_100: "f32[1, 256, 1, 1]" = var_mean_49[0]
        getitem_101: "f32[1, 256, 1, 1]" = var_mean_49[1];  var_mean_49 = None
        add_261: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
        rsqrt_49: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        sub_49: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_101)
        mul_343: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        squeeze_147: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        squeeze_148: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_344: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
        mul_345: "f32[256]" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
        add_262: "f32[256]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
        squeeze_149: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_346: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0001594642002871);  squeeze_149 = None
        mul_347: "f32[256]" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
        mul_348: "f32[256]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
        add_263: "f32[256]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
        unsqueeze_196: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_300, -1)
        unsqueeze_197: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_349: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
        unsqueeze_198: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_301, -1);  primals_301 = None
        unsqueeze_199: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_264: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_46: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_264);  add_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_50: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_46, primals_302, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_265: "i64[]" = torch.ops.aten.add.Tensor(primals_303, 1)
        var_mean_50 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
        getitem_102: "f32[1, 256, 1, 1]" = var_mean_50[0]
        getitem_103: "f32[1, 256, 1, 1]" = var_mean_50[1];  var_mean_50 = None
        add_266: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
        rsqrt_50: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
        sub_50: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_103)
        mul_350: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        squeeze_150: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        squeeze_151: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_351: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
        mul_352: "f32[256]" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
        add_267: "f32[256]" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
        squeeze_152: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_353: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0001594642002871);  squeeze_152 = None
        mul_354: "f32[256]" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
        mul_355: "f32[256]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
        add_268: "f32[256]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
        unsqueeze_200: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_306, -1)
        unsqueeze_201: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        mul_356: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
        unsqueeze_202: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_307, -1);  primals_307 = None
        unsqueeze_203: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        add_269: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_47: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_269);  add_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_51: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_47, primals_308, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_270: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
        var_mean_51 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
        getitem_104: "f32[1, 1024, 1, 1]" = var_mean_51[0]
        getitem_105: "f32[1, 1024, 1, 1]" = var_mean_51[1];  var_mean_51 = None
        add_271: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
        rsqrt_51: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        sub_51: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_105)
        mul_357: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        squeeze_153: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        squeeze_154: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_358: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
        mul_359: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
        add_272: "f32[1024]" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
        squeeze_155: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        mul_360: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0001594642002871);  squeeze_155 = None
        mul_361: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
        mul_362: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
        add_273: "f32[1024]" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
        unsqueeze_204: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_312, -1)
        unsqueeze_205: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_363: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
        unsqueeze_206: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_313, -1);  primals_313 = None
        unsqueeze_207: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_274: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_275: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_274, relu_45);  add_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_48: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_275);  add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_52: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_48, primals_314, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_276: "i64[]" = torch.ops.aten.add.Tensor(primals_315, 1)
        var_mean_52 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
        getitem_106: "f32[1, 256, 1, 1]" = var_mean_52[0]
        getitem_107: "f32[1, 256, 1, 1]" = var_mean_52[1];  var_mean_52 = None
        add_277: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
        rsqrt_52: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_277);  add_277 = None
        sub_52: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_107)
        mul_364: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
        squeeze_156: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        squeeze_157: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_365: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
        mul_366: "f32[256]" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
        add_278: "f32[256]" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
        squeeze_158: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        mul_367: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0001594642002871);  squeeze_158 = None
        mul_368: "f32[256]" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
        mul_369: "f32[256]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
        add_279: "f32[256]" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
        unsqueeze_208: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_318, -1)
        unsqueeze_209: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        mul_370: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
        unsqueeze_210: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_319, -1);  primals_319 = None
        unsqueeze_211: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        add_280: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_49: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_280);  add_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_53: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_49, primals_320, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_281: "i64[]" = torch.ops.aten.add.Tensor(primals_321, 1)
        var_mean_53 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
        getitem_108: "f32[1, 256, 1, 1]" = var_mean_53[0]
        getitem_109: "f32[1, 256, 1, 1]" = var_mean_53[1];  var_mean_53 = None
        add_282: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
        rsqrt_53: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
        sub_53: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_109)
        mul_371: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
        squeeze_159: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
        squeeze_160: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
        mul_372: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
        mul_373: "f32[256]" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
        add_283: "f32[256]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
        squeeze_161: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
        mul_374: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0001594642002871);  squeeze_161 = None
        mul_375: "f32[256]" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
        mul_376: "f32[256]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
        add_284: "f32[256]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
        unsqueeze_212: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_324, -1)
        unsqueeze_213: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_377: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_213);  mul_371 = unsqueeze_213 = None
        unsqueeze_214: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_325, -1);  primals_325 = None
        unsqueeze_215: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_285: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_215);  mul_377 = unsqueeze_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_50: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_285);  add_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_54: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_50, primals_326, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_286: "i64[]" = torch.ops.aten.add.Tensor(primals_327, 1)
        var_mean_54 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
        getitem_110: "f32[1, 1024, 1, 1]" = var_mean_54[0]
        getitem_111: "f32[1, 1024, 1, 1]" = var_mean_54[1];  var_mean_54 = None
        add_287: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
        rsqrt_54: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_287);  add_287 = None
        sub_54: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_111)
        mul_378: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
        squeeze_162: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
        squeeze_163: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
        mul_379: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
        mul_380: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
        add_288: "f32[1024]" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
        squeeze_164: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
        mul_381: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0001594642002871);  squeeze_164 = None
        mul_382: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_381, 0.1);  mul_381 = None
        mul_383: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
        add_289: "f32[1024]" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
        unsqueeze_216: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_330, -1)
        unsqueeze_217: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        mul_384: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_217);  mul_378 = unsqueeze_217 = None
        unsqueeze_218: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_331, -1);  primals_331 = None
        unsqueeze_219: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        add_290: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_219);  mul_384 = unsqueeze_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_291: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_290, relu_48);  add_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_51: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_291);  add_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_55: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_51, primals_332, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_292: "i64[]" = torch.ops.aten.add.Tensor(primals_333, 1)
        var_mean_55 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
        getitem_112: "f32[1, 256, 1, 1]" = var_mean_55[0]
        getitem_113: "f32[1, 256, 1, 1]" = var_mean_55[1];  var_mean_55 = None
        add_293: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
        rsqrt_55: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_293);  add_293 = None
        sub_55: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_113)
        mul_385: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
        squeeze_165: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
        squeeze_166: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
        mul_386: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
        mul_387: "f32[256]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
        add_294: "f32[256]" = torch.ops.aten.add.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
        squeeze_167: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
        mul_388: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0001594642002871);  squeeze_167 = None
        mul_389: "f32[256]" = torch.ops.aten.mul.Tensor(mul_388, 0.1);  mul_388 = None
        mul_390: "f32[256]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
        add_295: "f32[256]" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
        unsqueeze_220: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_336, -1)
        unsqueeze_221: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_391: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_221);  mul_385 = unsqueeze_221 = None
        unsqueeze_222: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_337, -1);  primals_337 = None
        unsqueeze_223: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_296: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_391, unsqueeze_223);  mul_391 = unsqueeze_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_52: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_296);  add_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_56: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_52, primals_338, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_297: "i64[]" = torch.ops.aten.add.Tensor(primals_339, 1)
        var_mean_56 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
        getitem_114: "f32[1, 256, 1, 1]" = var_mean_56[0]
        getitem_115: "f32[1, 256, 1, 1]" = var_mean_56[1];  var_mean_56 = None
        add_298: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05)
        rsqrt_56: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_298);  add_298 = None
        sub_56: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_115)
        mul_392: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
        squeeze_168: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
        squeeze_169: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
        mul_393: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
        mul_394: "f32[256]" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
        add_299: "f32[256]" = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
        squeeze_170: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
        mul_395: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0001594642002871);  squeeze_170 = None
        mul_396: "f32[256]" = torch.ops.aten.mul.Tensor(mul_395, 0.1);  mul_395 = None
        mul_397: "f32[256]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
        add_300: "f32[256]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
        unsqueeze_224: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_342, -1)
        unsqueeze_225: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        mul_398: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_225);  mul_392 = unsqueeze_225 = None
        unsqueeze_226: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_343, -1);  primals_343 = None
        unsqueeze_227: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        add_301: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_227);  mul_398 = unsqueeze_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_53: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_301);  add_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_57: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_53, primals_344, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_302: "i64[]" = torch.ops.aten.add.Tensor(primals_345, 1)
        var_mean_57 = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
        getitem_116: "f32[1, 1024, 1, 1]" = var_mean_57[0]
        getitem_117: "f32[1, 1024, 1, 1]" = var_mean_57[1];  var_mean_57 = None
        add_303: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
        rsqrt_57: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        sub_57: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_117)
        mul_399: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
        squeeze_171: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
        squeeze_172: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
        mul_400: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
        mul_401: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
        add_304: "f32[1024]" = torch.ops.aten.add.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
        squeeze_173: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
        mul_402: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0001594642002871);  squeeze_173 = None
        mul_403: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_402, 0.1);  mul_402 = None
        mul_404: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
        add_305: "f32[1024]" = torch.ops.aten.add.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
        unsqueeze_228: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_348, -1)
        unsqueeze_229: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_405: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_399, unsqueeze_229);  mul_399 = unsqueeze_229 = None
        unsqueeze_230: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_349, -1);  primals_349 = None
        unsqueeze_231: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_306: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_405, unsqueeze_231);  mul_405 = unsqueeze_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_307: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_306, relu_51);  add_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_54: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_307);  add_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_58: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_54, primals_350, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_308: "i64[]" = torch.ops.aten.add.Tensor(primals_351, 1)
        var_mean_58 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
        getitem_118: "f32[1, 256, 1, 1]" = var_mean_58[0]
        getitem_119: "f32[1, 256, 1, 1]" = var_mean_58[1];  var_mean_58 = None
        add_309: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
        rsqrt_58: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_309);  add_309 = None
        sub_58: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_119)
        mul_406: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
        squeeze_174: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
        squeeze_175: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
        mul_407: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
        mul_408: "f32[256]" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
        add_310: "f32[256]" = torch.ops.aten.add.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
        squeeze_176: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
        mul_409: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0001594642002871);  squeeze_176 = None
        mul_410: "f32[256]" = torch.ops.aten.mul.Tensor(mul_409, 0.1);  mul_409 = None
        mul_411: "f32[256]" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
        add_311: "f32[256]" = torch.ops.aten.add.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
        unsqueeze_232: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_354, -1)
        unsqueeze_233: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        mul_412: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_233);  mul_406 = unsqueeze_233 = None
        unsqueeze_234: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_355, -1);  primals_355 = None
        unsqueeze_235: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        add_312: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_412, unsqueeze_235);  mul_412 = unsqueeze_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_55: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_312);  add_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_59: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_55, primals_356, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_313: "i64[]" = torch.ops.aten.add.Tensor(primals_357, 1)
        var_mean_59 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
        getitem_120: "f32[1, 256, 1, 1]" = var_mean_59[0]
        getitem_121: "f32[1, 256, 1, 1]" = var_mean_59[1];  var_mean_59 = None
        add_314: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
        rsqrt_59: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_314);  add_314 = None
        sub_59: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_121)
        mul_413: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
        squeeze_177: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        squeeze_178: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
        mul_414: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
        mul_415: "f32[256]" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
        add_315: "f32[256]" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
        squeeze_179: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
        mul_416: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0001594642002871);  squeeze_179 = None
        mul_417: "f32[256]" = torch.ops.aten.mul.Tensor(mul_416, 0.1);  mul_416 = None
        mul_418: "f32[256]" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
        add_316: "f32[256]" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
        unsqueeze_236: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_360, -1)
        unsqueeze_237: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_419: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_237);  mul_413 = unsqueeze_237 = None
        unsqueeze_238: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_361, -1);  primals_361 = None
        unsqueeze_239: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_317: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_239);  mul_419 = unsqueeze_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_56: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_317);  add_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_60: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_56, primals_362, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_318: "i64[]" = torch.ops.aten.add.Tensor(primals_363, 1)
        var_mean_60 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
        getitem_122: "f32[1, 1024, 1, 1]" = var_mean_60[0]
        getitem_123: "f32[1, 1024, 1, 1]" = var_mean_60[1];  var_mean_60 = None
        add_319: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05)
        rsqrt_60: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_319);  add_319 = None
        sub_60: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_123)
        mul_420: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
        squeeze_180: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
        squeeze_181: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
        mul_421: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
        mul_422: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
        add_320: "f32[1024]" = torch.ops.aten.add.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
        squeeze_182: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
        mul_423: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0001594642002871);  squeeze_182 = None
        mul_424: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_423, 0.1);  mul_423 = None
        mul_425: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
        add_321: "f32[1024]" = torch.ops.aten.add.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
        unsqueeze_240: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_366, -1)
        unsqueeze_241: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        mul_426: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_241);  mul_420 = unsqueeze_241 = None
        unsqueeze_242: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_367, -1);  primals_367 = None
        unsqueeze_243: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        add_322: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_426, unsqueeze_243);  mul_426 = unsqueeze_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_323: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_322, relu_54);  add_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_57: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_323);  add_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_61: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_57, primals_368, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_324: "i64[]" = torch.ops.aten.add.Tensor(primals_369, 1)
        var_mean_61 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
        getitem_124: "f32[1, 256, 1, 1]" = var_mean_61[0]
        getitem_125: "f32[1, 256, 1, 1]" = var_mean_61[1];  var_mean_61 = None
        add_325: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05)
        rsqrt_61: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_325);  add_325 = None
        sub_61: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_125)
        mul_427: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
        squeeze_183: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
        squeeze_184: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
        mul_428: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
        mul_429: "f32[256]" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
        add_326: "f32[256]" = torch.ops.aten.add.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
        squeeze_185: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
        mul_430: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0001594642002871);  squeeze_185 = None
        mul_431: "f32[256]" = torch.ops.aten.mul.Tensor(mul_430, 0.1);  mul_430 = None
        mul_432: "f32[256]" = torch.ops.aten.mul.Tensor(primals_371, 0.9)
        add_327: "f32[256]" = torch.ops.aten.add.Tensor(mul_431, mul_432);  mul_431 = mul_432 = None
        unsqueeze_244: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_372, -1)
        unsqueeze_245: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_433: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_245);  mul_427 = unsqueeze_245 = None
        unsqueeze_246: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_373, -1);  primals_373 = None
        unsqueeze_247: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_328: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_433, unsqueeze_247);  mul_433 = unsqueeze_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_58: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_328);  add_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_62: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_58, primals_374, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_329: "i64[]" = torch.ops.aten.add.Tensor(primals_375, 1)
        var_mean_62 = torch.ops.aten.var_mean.correction(convolution_62, [0, 2, 3], correction = 0, keepdim = True)
        getitem_126: "f32[1, 256, 1, 1]" = var_mean_62[0]
        getitem_127: "f32[1, 256, 1, 1]" = var_mean_62[1];  var_mean_62 = None
        add_330: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05)
        rsqrt_62: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_330);  add_330 = None
        sub_62: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, getitem_127)
        mul_434: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
        squeeze_186: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
        squeeze_187: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
        mul_435: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
        mul_436: "f32[256]" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
        add_331: "f32[256]" = torch.ops.aten.add.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
        squeeze_188: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
        mul_437: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0001594642002871);  squeeze_188 = None
        mul_438: "f32[256]" = torch.ops.aten.mul.Tensor(mul_437, 0.1);  mul_437 = None
        mul_439: "f32[256]" = torch.ops.aten.mul.Tensor(primals_377, 0.9)
        add_332: "f32[256]" = torch.ops.aten.add.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
        unsqueeze_248: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_378, -1)
        unsqueeze_249: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        mul_440: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_434, unsqueeze_249);  mul_434 = unsqueeze_249 = None
        unsqueeze_250: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_379, -1);  primals_379 = None
        unsqueeze_251: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        add_333: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_251);  mul_440 = unsqueeze_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_59: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_333);  add_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_63: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_59, primals_380, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_334: "i64[]" = torch.ops.aten.add.Tensor(primals_381, 1)
        var_mean_63 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
        getitem_128: "f32[1, 1024, 1, 1]" = var_mean_63[0]
        getitem_129: "f32[1, 1024, 1, 1]" = var_mean_63[1];  var_mean_63 = None
        add_335: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05)
        rsqrt_63: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_335);  add_335 = None
        sub_63: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_129)
        mul_441: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
        squeeze_189: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
        squeeze_190: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
        mul_442: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
        mul_443: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
        add_336: "f32[1024]" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
        squeeze_191: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
        mul_444: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0001594642002871);  squeeze_191 = None
        mul_445: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_444, 0.1);  mul_444 = None
        mul_446: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_383, 0.9)
        add_337: "f32[1024]" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
        unsqueeze_252: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_384, -1)
        unsqueeze_253: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_447: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_441, unsqueeze_253);  mul_441 = unsqueeze_253 = None
        unsqueeze_254: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_385, -1);  primals_385 = None
        unsqueeze_255: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_338: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_447, unsqueeze_255);  mul_447 = unsqueeze_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_339: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_338, relu_57);  add_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_60: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_339);  add_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_64: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_60, primals_386, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_340: "i64[]" = torch.ops.aten.add.Tensor(primals_387, 1)
        var_mean_64 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
        getitem_130: "f32[1, 256, 1, 1]" = var_mean_64[0]
        getitem_131: "f32[1, 256, 1, 1]" = var_mean_64[1];  var_mean_64 = None
        add_341: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05)
        rsqrt_64: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
        sub_64: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_131)
        mul_448: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
        squeeze_192: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_131, [0, 2, 3]);  getitem_131 = None
        squeeze_193: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
        mul_449: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
        mul_450: "f32[256]" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
        add_342: "f32[256]" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
        squeeze_194: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_130, [0, 2, 3]);  getitem_130 = None
        mul_451: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_194, 1.0001594642002871);  squeeze_194 = None
        mul_452: "f32[256]" = torch.ops.aten.mul.Tensor(mul_451, 0.1);  mul_451 = None
        mul_453: "f32[256]" = torch.ops.aten.mul.Tensor(primals_389, 0.9)
        add_343: "f32[256]" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
        unsqueeze_256: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_390, -1)
        unsqueeze_257: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        mul_454: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_257);  mul_448 = unsqueeze_257 = None
        unsqueeze_258: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_391, -1);  primals_391 = None
        unsqueeze_259: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        add_344: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_454, unsqueeze_259);  mul_454 = unsqueeze_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_61: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_344);  add_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_65: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_61, primals_392, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_345: "i64[]" = torch.ops.aten.add.Tensor(primals_393, 1)
        var_mean_65 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
        getitem_132: "f32[1, 256, 1, 1]" = var_mean_65[0]
        getitem_133: "f32[1, 256, 1, 1]" = var_mean_65[1];  var_mean_65 = None
        add_346: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05)
        rsqrt_65: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_346);  add_346 = None
        sub_65: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_133)
        mul_455: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = None
        squeeze_195: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
        squeeze_196: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
        mul_456: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_195, 0.1)
        mul_457: "f32[256]" = torch.ops.aten.mul.Tensor(primals_394, 0.9)
        add_347: "f32[256]" = torch.ops.aten.add.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
        squeeze_197: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
        mul_458: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_197, 1.0001594642002871);  squeeze_197 = None
        mul_459: "f32[256]" = torch.ops.aten.mul.Tensor(mul_458, 0.1);  mul_458 = None
        mul_460: "f32[256]" = torch.ops.aten.mul.Tensor(primals_395, 0.9)
        add_348: "f32[256]" = torch.ops.aten.add.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
        unsqueeze_260: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_396, -1)
        unsqueeze_261: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_461: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_455, unsqueeze_261);  mul_455 = unsqueeze_261 = None
        unsqueeze_262: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_397, -1);  primals_397 = None
        unsqueeze_263: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_349: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_461, unsqueeze_263);  mul_461 = unsqueeze_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_62: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_349);  add_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_66: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_62, primals_398, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_350: "i64[]" = torch.ops.aten.add.Tensor(primals_399, 1)
        var_mean_66 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
        getitem_134: "f32[1, 1024, 1, 1]" = var_mean_66[0]
        getitem_135: "f32[1, 1024, 1, 1]" = var_mean_66[1];  var_mean_66 = None
        add_351: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05)
        rsqrt_66: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
        sub_66: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_135)
        mul_462: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = None
        squeeze_198: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
        squeeze_199: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
        mul_463: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_198, 0.1)
        mul_464: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_400, 0.9)
        add_352: "f32[1024]" = torch.ops.aten.add.Tensor(mul_463, mul_464);  mul_463 = mul_464 = None
        squeeze_200: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
        mul_465: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_200, 1.0001594642002871);  squeeze_200 = None
        mul_466: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_465, 0.1);  mul_465 = None
        mul_467: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_401, 0.9)
        add_353: "f32[1024]" = torch.ops.aten.add.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
        unsqueeze_264: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_402, -1)
        unsqueeze_265: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        mul_468: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_462, unsqueeze_265);  mul_462 = unsqueeze_265 = None
        unsqueeze_266: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_403, -1);  primals_403 = None
        unsqueeze_267: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        add_354: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_468, unsqueeze_267);  mul_468 = unsqueeze_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_355: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_354, relu_60);  add_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_63: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_355);  add_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_67: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_63, primals_404, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_356: "i64[]" = torch.ops.aten.add.Tensor(primals_405, 1)
        var_mean_67 = torch.ops.aten.var_mean.correction(convolution_67, [0, 2, 3], correction = 0, keepdim = True)
        getitem_136: "f32[1, 256, 1, 1]" = var_mean_67[0]
        getitem_137: "f32[1, 256, 1, 1]" = var_mean_67[1];  var_mean_67 = None
        add_357: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05)
        rsqrt_67: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_357);  add_357 = None
        sub_67: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, getitem_137)
        mul_469: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = None
        squeeze_201: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
        squeeze_202: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
        mul_470: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_201, 0.1)
        mul_471: "f32[256]" = torch.ops.aten.mul.Tensor(primals_406, 0.9)
        add_358: "f32[256]" = torch.ops.aten.add.Tensor(mul_470, mul_471);  mul_470 = mul_471 = None
        squeeze_203: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
        mul_472: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_203, 1.0001594642002871);  squeeze_203 = None
        mul_473: "f32[256]" = torch.ops.aten.mul.Tensor(mul_472, 0.1);  mul_472 = None
        mul_474: "f32[256]" = torch.ops.aten.mul.Tensor(primals_407, 0.9)
        add_359: "f32[256]" = torch.ops.aten.add.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
        unsqueeze_268: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_408, -1)
        unsqueeze_269: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_475: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_269);  mul_469 = unsqueeze_269 = None
        unsqueeze_270: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_409, -1);  primals_409 = None
        unsqueeze_271: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_360: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_475, unsqueeze_271);  mul_475 = unsqueeze_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_64: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_360);  add_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_68: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_64, primals_410, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_361: "i64[]" = torch.ops.aten.add.Tensor(primals_411, 1)
        var_mean_68 = torch.ops.aten.var_mean.correction(convolution_68, [0, 2, 3], correction = 0, keepdim = True)
        getitem_138: "f32[1, 256, 1, 1]" = var_mean_68[0]
        getitem_139: "f32[1, 256, 1, 1]" = var_mean_68[1];  var_mean_68 = None
        add_362: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05)
        rsqrt_68: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_362);  add_362 = None
        sub_68: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, getitem_139)
        mul_476: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = None
        squeeze_204: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
        squeeze_205: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
        mul_477: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_204, 0.1)
        mul_478: "f32[256]" = torch.ops.aten.mul.Tensor(primals_412, 0.9)
        add_363: "f32[256]" = torch.ops.aten.add.Tensor(mul_477, mul_478);  mul_477 = mul_478 = None
        squeeze_206: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
        mul_479: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_206, 1.0001594642002871);  squeeze_206 = None
        mul_480: "f32[256]" = torch.ops.aten.mul.Tensor(mul_479, 0.1);  mul_479 = None
        mul_481: "f32[256]" = torch.ops.aten.mul.Tensor(primals_413, 0.9)
        add_364: "f32[256]" = torch.ops.aten.add.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
        unsqueeze_272: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_414, -1)
        unsqueeze_273: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        mul_482: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_476, unsqueeze_273);  mul_476 = unsqueeze_273 = None
        unsqueeze_274: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_415, -1);  primals_415 = None
        unsqueeze_275: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        add_365: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_482, unsqueeze_275);  mul_482 = unsqueeze_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_65: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_365);  add_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_69: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_65, primals_416, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_366: "i64[]" = torch.ops.aten.add.Tensor(primals_417, 1)
        var_mean_69 = torch.ops.aten.var_mean.correction(convolution_69, [0, 2, 3], correction = 0, keepdim = True)
        getitem_140: "f32[1, 1024, 1, 1]" = var_mean_69[0]
        getitem_141: "f32[1, 1024, 1, 1]" = var_mean_69[1];  var_mean_69 = None
        add_367: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05)
        rsqrt_69: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_367);  add_367 = None
        sub_69: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_141)
        mul_483: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = None
        squeeze_207: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
        squeeze_208: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
        mul_484: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_207, 0.1)
        mul_485: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_418, 0.9)
        add_368: "f32[1024]" = torch.ops.aten.add.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
        squeeze_209: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
        mul_486: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_209, 1.0001594642002871);  squeeze_209 = None
        mul_487: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_486, 0.1);  mul_486 = None
        mul_488: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_419, 0.9)
        add_369: "f32[1024]" = torch.ops.aten.add.Tensor(mul_487, mul_488);  mul_487 = mul_488 = None
        unsqueeze_276: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_420, -1)
        unsqueeze_277: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_489: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_483, unsqueeze_277);  mul_483 = unsqueeze_277 = None
        unsqueeze_278: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_421, -1);  primals_421 = None
        unsqueeze_279: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_370: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_489, unsqueeze_279);  mul_489 = unsqueeze_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_371: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_370, relu_63);  add_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_66: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_371);  add_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_70: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_66, primals_422, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_372: "i64[]" = torch.ops.aten.add.Tensor(primals_423, 1)
        var_mean_70 = torch.ops.aten.var_mean.correction(convolution_70, [0, 2, 3], correction = 0, keepdim = True)
        getitem_142: "f32[1, 256, 1, 1]" = var_mean_70[0]
        getitem_143: "f32[1, 256, 1, 1]" = var_mean_70[1];  var_mean_70 = None
        add_373: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05)
        rsqrt_70: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_373);  add_373 = None
        sub_70: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_143)
        mul_490: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = None
        squeeze_210: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
        squeeze_211: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
        mul_491: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_210, 0.1)
        mul_492: "f32[256]" = torch.ops.aten.mul.Tensor(primals_424, 0.9)
        add_374: "f32[256]" = torch.ops.aten.add.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
        squeeze_212: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
        mul_493: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_212, 1.0001594642002871);  squeeze_212 = None
        mul_494: "f32[256]" = torch.ops.aten.mul.Tensor(mul_493, 0.1);  mul_493 = None
        mul_495: "f32[256]" = torch.ops.aten.mul.Tensor(primals_425, 0.9)
        add_375: "f32[256]" = torch.ops.aten.add.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
        unsqueeze_280: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_426, -1)
        unsqueeze_281: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        mul_496: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_490, unsqueeze_281);  mul_490 = unsqueeze_281 = None
        unsqueeze_282: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_427, -1);  primals_427 = None
        unsqueeze_283: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        add_376: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_496, unsqueeze_283);  mul_496 = unsqueeze_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_67: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_376);  add_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_71: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_67, primals_428, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_377: "i64[]" = torch.ops.aten.add.Tensor(primals_429, 1)
        var_mean_71 = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True)
        getitem_144: "f32[1, 256, 1, 1]" = var_mean_71[0]
        getitem_145: "f32[1, 256, 1, 1]" = var_mean_71[1];  var_mean_71 = None
        add_378: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05)
        rsqrt_71: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_378);  add_378 = None
        sub_71: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_145)
        mul_497: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = None
        squeeze_213: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
        squeeze_214: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
        mul_498: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_213, 0.1)
        mul_499: "f32[256]" = torch.ops.aten.mul.Tensor(primals_430, 0.9)
        add_379: "f32[256]" = torch.ops.aten.add.Tensor(mul_498, mul_499);  mul_498 = mul_499 = None
        squeeze_215: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
        mul_500: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_215, 1.0001594642002871);  squeeze_215 = None
        mul_501: "f32[256]" = torch.ops.aten.mul.Tensor(mul_500, 0.1);  mul_500 = None
        mul_502: "f32[256]" = torch.ops.aten.mul.Tensor(primals_431, 0.9)
        add_380: "f32[256]" = torch.ops.aten.add.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
        unsqueeze_284: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_432, -1)
        unsqueeze_285: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_503: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_497, unsqueeze_285);  mul_497 = unsqueeze_285 = None
        unsqueeze_286: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_433, -1);  primals_433 = None
        unsqueeze_287: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_381: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_503, unsqueeze_287);  mul_503 = unsqueeze_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_68: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_381);  add_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_72: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_68, primals_434, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_382: "i64[]" = torch.ops.aten.add.Tensor(primals_435, 1)
        var_mean_72 = torch.ops.aten.var_mean.correction(convolution_72, [0, 2, 3], correction = 0, keepdim = True)
        getitem_146: "f32[1, 1024, 1, 1]" = var_mean_72[0]
        getitem_147: "f32[1, 1024, 1, 1]" = var_mean_72[1];  var_mean_72 = None
        add_383: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05)
        rsqrt_72: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_383);  add_383 = None
        sub_72: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, getitem_147)
        mul_504: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = None
        squeeze_216: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
        squeeze_217: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_72, [0, 2, 3]);  rsqrt_72 = None
        mul_505: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_216, 0.1)
        mul_506: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_436, 0.9)
        add_384: "f32[1024]" = torch.ops.aten.add.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
        squeeze_218: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
        mul_507: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_218, 1.0001594642002871);  squeeze_218 = None
        mul_508: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_507, 0.1);  mul_507 = None
        mul_509: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_437, 0.9)
        add_385: "f32[1024]" = torch.ops.aten.add.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
        unsqueeze_288: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_438, -1)
        unsqueeze_289: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        mul_510: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_504, unsqueeze_289);  mul_504 = unsqueeze_289 = None
        unsqueeze_290: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_439, -1);  primals_439 = None
        unsqueeze_291: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        add_386: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_510, unsqueeze_291);  mul_510 = unsqueeze_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_387: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_386, relu_66);  add_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_69: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_387);  add_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_73: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_69, primals_440, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_388: "i64[]" = torch.ops.aten.add.Tensor(primals_441, 1)
        var_mean_73 = torch.ops.aten.var_mean.correction(convolution_73, [0, 2, 3], correction = 0, keepdim = True)
        getitem_148: "f32[1, 256, 1, 1]" = var_mean_73[0]
        getitem_149: "f32[1, 256, 1, 1]" = var_mean_73[1];  var_mean_73 = None
        add_389: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05)
        rsqrt_73: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_389);  add_389 = None
        sub_73: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, getitem_149)
        mul_511: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = None
        squeeze_219: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
        squeeze_220: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
        mul_512: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_219, 0.1)
        mul_513: "f32[256]" = torch.ops.aten.mul.Tensor(primals_442, 0.9)
        add_390: "f32[256]" = torch.ops.aten.add.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
        squeeze_221: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
        mul_514: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_221, 1.0001594642002871);  squeeze_221 = None
        mul_515: "f32[256]" = torch.ops.aten.mul.Tensor(mul_514, 0.1);  mul_514 = None
        mul_516: "f32[256]" = torch.ops.aten.mul.Tensor(primals_443, 0.9)
        add_391: "f32[256]" = torch.ops.aten.add.Tensor(mul_515, mul_516);  mul_515 = mul_516 = None
        unsqueeze_292: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_444, -1)
        unsqueeze_293: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_517: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_511, unsqueeze_293);  mul_511 = unsqueeze_293 = None
        unsqueeze_294: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_445, -1);  primals_445 = None
        unsqueeze_295: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_392: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_517, unsqueeze_295);  mul_517 = unsqueeze_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_70: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_392);  add_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_74: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_70, primals_446, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_393: "i64[]" = torch.ops.aten.add.Tensor(primals_447, 1)
        var_mean_74 = torch.ops.aten.var_mean.correction(convolution_74, [0, 2, 3], correction = 0, keepdim = True)
        getitem_150: "f32[1, 256, 1, 1]" = var_mean_74[0]
        getitem_151: "f32[1, 256, 1, 1]" = var_mean_74[1];  var_mean_74 = None
        add_394: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05)
        rsqrt_74: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_394);  add_394 = None
        sub_74: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, getitem_151)
        mul_518: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = None
        squeeze_222: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
        squeeze_223: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
        mul_519: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_222, 0.1)
        mul_520: "f32[256]" = torch.ops.aten.mul.Tensor(primals_448, 0.9)
        add_395: "f32[256]" = torch.ops.aten.add.Tensor(mul_519, mul_520);  mul_519 = mul_520 = None
        squeeze_224: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
        mul_521: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_224, 1.0001594642002871);  squeeze_224 = None
        mul_522: "f32[256]" = torch.ops.aten.mul.Tensor(mul_521, 0.1);  mul_521 = None
        mul_523: "f32[256]" = torch.ops.aten.mul.Tensor(primals_449, 0.9)
        add_396: "f32[256]" = torch.ops.aten.add.Tensor(mul_522, mul_523);  mul_522 = mul_523 = None
        unsqueeze_296: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_450, -1)
        unsqueeze_297: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        mul_524: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_518, unsqueeze_297);  mul_518 = unsqueeze_297 = None
        unsqueeze_298: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_451, -1);  primals_451 = None
        unsqueeze_299: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        add_397: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_524, unsqueeze_299);  mul_524 = unsqueeze_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_71: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_397);  add_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_75: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_71, primals_452, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_398: "i64[]" = torch.ops.aten.add.Tensor(primals_453, 1)
        var_mean_75 = torch.ops.aten.var_mean.correction(convolution_75, [0, 2, 3], correction = 0, keepdim = True)
        getitem_152: "f32[1, 1024, 1, 1]" = var_mean_75[0]
        getitem_153: "f32[1, 1024, 1, 1]" = var_mean_75[1];  var_mean_75 = None
        add_399: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-05)
        rsqrt_75: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_399);  add_399 = None
        sub_75: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_153)
        mul_525: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = None
        squeeze_225: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
        squeeze_226: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
        mul_526: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_225, 0.1)
        mul_527: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_454, 0.9)
        add_400: "f32[1024]" = torch.ops.aten.add.Tensor(mul_526, mul_527);  mul_526 = mul_527 = None
        squeeze_227: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
        mul_528: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_227, 1.0001594642002871);  squeeze_227 = None
        mul_529: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_528, 0.1);  mul_528 = None
        mul_530: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_455, 0.9)
        add_401: "f32[1024]" = torch.ops.aten.add.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
        unsqueeze_300: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_456, -1)
        unsqueeze_301: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_531: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_525, unsqueeze_301);  mul_525 = unsqueeze_301 = None
        unsqueeze_302: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_457, -1);  primals_457 = None
        unsqueeze_303: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_402: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_531, unsqueeze_303);  mul_531 = unsqueeze_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_403: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_402, relu_69);  add_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_72: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_403);  add_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_76: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_72, primals_458, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_404: "i64[]" = torch.ops.aten.add.Tensor(primals_459, 1)
        var_mean_76 = torch.ops.aten.var_mean.correction(convolution_76, [0, 2, 3], correction = 0, keepdim = True)
        getitem_154: "f32[1, 256, 1, 1]" = var_mean_76[0]
        getitem_155: "f32[1, 256, 1, 1]" = var_mean_76[1];  var_mean_76 = None
        add_405: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05)
        rsqrt_76: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_405);  add_405 = None
        sub_76: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, getitem_155)
        mul_532: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = None
        squeeze_228: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
        squeeze_229: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
        mul_533: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_228, 0.1)
        mul_534: "f32[256]" = torch.ops.aten.mul.Tensor(primals_460, 0.9)
        add_406: "f32[256]" = torch.ops.aten.add.Tensor(mul_533, mul_534);  mul_533 = mul_534 = None
        squeeze_230: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
        mul_535: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_230, 1.0001594642002871);  squeeze_230 = None
        mul_536: "f32[256]" = torch.ops.aten.mul.Tensor(mul_535, 0.1);  mul_535 = None
        mul_537: "f32[256]" = torch.ops.aten.mul.Tensor(primals_461, 0.9)
        add_407: "f32[256]" = torch.ops.aten.add.Tensor(mul_536, mul_537);  mul_536 = mul_537 = None
        unsqueeze_304: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_462, -1)
        unsqueeze_305: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        mul_538: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_532, unsqueeze_305);  mul_532 = unsqueeze_305 = None
        unsqueeze_306: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_463, -1);  primals_463 = None
        unsqueeze_307: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        add_408: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_538, unsqueeze_307);  mul_538 = unsqueeze_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_73: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_408);  add_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_77: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_73, primals_464, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_409: "i64[]" = torch.ops.aten.add.Tensor(primals_465, 1)
        var_mean_77 = torch.ops.aten.var_mean.correction(convolution_77, [0, 2, 3], correction = 0, keepdim = True)
        getitem_156: "f32[1, 256, 1, 1]" = var_mean_77[0]
        getitem_157: "f32[1, 256, 1, 1]" = var_mean_77[1];  var_mean_77 = None
        add_410: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05)
        rsqrt_77: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_410);  add_410 = None
        sub_77: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, getitem_157)
        mul_539: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = None
        squeeze_231: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
        squeeze_232: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
        mul_540: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_231, 0.1)
        mul_541: "f32[256]" = torch.ops.aten.mul.Tensor(primals_466, 0.9)
        add_411: "f32[256]" = torch.ops.aten.add.Tensor(mul_540, mul_541);  mul_540 = mul_541 = None
        squeeze_233: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
        mul_542: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_233, 1.0001594642002871);  squeeze_233 = None
        mul_543: "f32[256]" = torch.ops.aten.mul.Tensor(mul_542, 0.1);  mul_542 = None
        mul_544: "f32[256]" = torch.ops.aten.mul.Tensor(primals_467, 0.9)
        add_412: "f32[256]" = torch.ops.aten.add.Tensor(mul_543, mul_544);  mul_543 = mul_544 = None
        unsqueeze_308: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_468, -1)
        unsqueeze_309: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_545: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_539, unsqueeze_309);  mul_539 = unsqueeze_309 = None
        unsqueeze_310: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_469, -1);  primals_469 = None
        unsqueeze_311: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_413: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_545, unsqueeze_311);  mul_545 = unsqueeze_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_74: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_413);  add_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_78: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_74, primals_470, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_414: "i64[]" = torch.ops.aten.add.Tensor(primals_471, 1)
        var_mean_78 = torch.ops.aten.var_mean.correction(convolution_78, [0, 2, 3], correction = 0, keepdim = True)
        getitem_158: "f32[1, 1024, 1, 1]" = var_mean_78[0]
        getitem_159: "f32[1, 1024, 1, 1]" = var_mean_78[1];  var_mean_78 = None
        add_415: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05)
        rsqrt_78: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_415);  add_415 = None
        sub_78: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, getitem_159)
        mul_546: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = None
        squeeze_234: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
        squeeze_235: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
        mul_547: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_234, 0.1)
        mul_548: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_472, 0.9)
        add_416: "f32[1024]" = torch.ops.aten.add.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
        squeeze_236: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
        mul_549: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_236, 1.0001594642002871);  squeeze_236 = None
        mul_550: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_549, 0.1);  mul_549 = None
        mul_551: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_473, 0.9)
        add_417: "f32[1024]" = torch.ops.aten.add.Tensor(mul_550, mul_551);  mul_550 = mul_551 = None
        unsqueeze_312: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_474, -1)
        unsqueeze_313: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        mul_552: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_546, unsqueeze_313);  mul_546 = unsqueeze_313 = None
        unsqueeze_314: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_475, -1);  primals_475 = None
        unsqueeze_315: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        add_418: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_552, unsqueeze_315);  mul_552 = unsqueeze_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_419: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_418, relu_72);  add_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_75: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_419);  add_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_79: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_75, primals_476, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_420: "i64[]" = torch.ops.aten.add.Tensor(primals_477, 1)
        var_mean_79 = torch.ops.aten.var_mean.correction(convolution_79, [0, 2, 3], correction = 0, keepdim = True)
        getitem_160: "f32[1, 256, 1, 1]" = var_mean_79[0]
        getitem_161: "f32[1, 256, 1, 1]" = var_mean_79[1];  var_mean_79 = None
        add_421: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05)
        rsqrt_79: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_421);  add_421 = None
        sub_79: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, getitem_161)
        mul_553: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = None
        squeeze_237: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_161, [0, 2, 3]);  getitem_161 = None
        squeeze_238: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
        mul_554: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_237, 0.1)
        mul_555: "f32[256]" = torch.ops.aten.mul.Tensor(primals_478, 0.9)
        add_422: "f32[256]" = torch.ops.aten.add.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
        squeeze_239: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_160, [0, 2, 3]);  getitem_160 = None
        mul_556: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_239, 1.0001594642002871);  squeeze_239 = None
        mul_557: "f32[256]" = torch.ops.aten.mul.Tensor(mul_556, 0.1);  mul_556 = None
        mul_558: "f32[256]" = torch.ops.aten.mul.Tensor(primals_479, 0.9)
        add_423: "f32[256]" = torch.ops.aten.add.Tensor(mul_557, mul_558);  mul_557 = mul_558 = None
        unsqueeze_316: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_480, -1)
        unsqueeze_317: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_559: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_553, unsqueeze_317);  mul_553 = unsqueeze_317 = None
        unsqueeze_318: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_481, -1);  primals_481 = None
        unsqueeze_319: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_424: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_559, unsqueeze_319);  mul_559 = unsqueeze_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_76: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_424);  add_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_80: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_76, primals_482, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_425: "i64[]" = torch.ops.aten.add.Tensor(primals_483, 1)
        var_mean_80 = torch.ops.aten.var_mean.correction(convolution_80, [0, 2, 3], correction = 0, keepdim = True)
        getitem_162: "f32[1, 256, 1, 1]" = var_mean_80[0]
        getitem_163: "f32[1, 256, 1, 1]" = var_mean_80[1];  var_mean_80 = None
        add_426: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05)
        rsqrt_80: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_426);  add_426 = None
        sub_80: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, getitem_163)
        mul_560: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = None
        squeeze_240: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
        squeeze_241: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_80, [0, 2, 3]);  rsqrt_80 = None
        mul_561: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_240, 0.1)
        mul_562: "f32[256]" = torch.ops.aten.mul.Tensor(primals_484, 0.9)
        add_427: "f32[256]" = torch.ops.aten.add.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
        squeeze_242: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_162, [0, 2, 3]);  getitem_162 = None
        mul_563: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_242, 1.0001594642002871);  squeeze_242 = None
        mul_564: "f32[256]" = torch.ops.aten.mul.Tensor(mul_563, 0.1);  mul_563 = None
        mul_565: "f32[256]" = torch.ops.aten.mul.Tensor(primals_485, 0.9)
        add_428: "f32[256]" = torch.ops.aten.add.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
        unsqueeze_320: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_486, -1)
        unsqueeze_321: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        mul_566: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_560, unsqueeze_321);  mul_560 = unsqueeze_321 = None
        unsqueeze_322: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_487, -1);  primals_487 = None
        unsqueeze_323: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        add_429: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_566, unsqueeze_323);  mul_566 = unsqueeze_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_77: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_429);  add_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_81: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_77, primals_488, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_430: "i64[]" = torch.ops.aten.add.Tensor(primals_489, 1)
        var_mean_81 = torch.ops.aten.var_mean.correction(convolution_81, [0, 2, 3], correction = 0, keepdim = True)
        getitem_164: "f32[1, 1024, 1, 1]" = var_mean_81[0]
        getitem_165: "f32[1, 1024, 1, 1]" = var_mean_81[1];  var_mean_81 = None
        add_431: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-05)
        rsqrt_81: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_431);  add_431 = None
        sub_81: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, getitem_165)
        mul_567: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = None
        squeeze_243: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3]);  getitem_165 = None
        squeeze_244: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_81, [0, 2, 3]);  rsqrt_81 = None
        mul_568: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_243, 0.1)
        mul_569: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_490, 0.9)
        add_432: "f32[1024]" = torch.ops.aten.add.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
        squeeze_245: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_164, [0, 2, 3]);  getitem_164 = None
        mul_570: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_245, 1.0001594642002871);  squeeze_245 = None
        mul_571: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_570, 0.1);  mul_570 = None
        mul_572: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_491, 0.9)
        add_433: "f32[1024]" = torch.ops.aten.add.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
        unsqueeze_324: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_492, -1)
        unsqueeze_325: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_573: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_567, unsqueeze_325);  mul_567 = unsqueeze_325 = None
        unsqueeze_326: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_493, -1);  primals_493 = None
        unsqueeze_327: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_434: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_573, unsqueeze_327);  mul_573 = unsqueeze_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_435: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_434, relu_75);  add_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_78: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_435);  add_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_82: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_78, primals_494, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_436: "i64[]" = torch.ops.aten.add.Tensor(primals_495, 1)
        var_mean_82 = torch.ops.aten.var_mean.correction(convolution_82, [0, 2, 3], correction = 0, keepdim = True)
        getitem_166: "f32[1, 256, 1, 1]" = var_mean_82[0]
        getitem_167: "f32[1, 256, 1, 1]" = var_mean_82[1];  var_mean_82 = None
        add_437: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-05)
        rsqrt_82: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_437);  add_437 = None
        sub_82: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, getitem_167)
        mul_574: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = None
        squeeze_246: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
        squeeze_247: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_82, [0, 2, 3]);  rsqrt_82 = None
        mul_575: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_246, 0.1)
        mul_576: "f32[256]" = torch.ops.aten.mul.Tensor(primals_496, 0.9)
        add_438: "f32[256]" = torch.ops.aten.add.Tensor(mul_575, mul_576);  mul_575 = mul_576 = None
        squeeze_248: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
        mul_577: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_248, 1.0001594642002871);  squeeze_248 = None
        mul_578: "f32[256]" = torch.ops.aten.mul.Tensor(mul_577, 0.1);  mul_577 = None
        mul_579: "f32[256]" = torch.ops.aten.mul.Tensor(primals_497, 0.9)
        add_439: "f32[256]" = torch.ops.aten.add.Tensor(mul_578, mul_579);  mul_578 = mul_579 = None
        unsqueeze_328: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_498, -1)
        unsqueeze_329: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        mul_580: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_574, unsqueeze_329);  mul_574 = unsqueeze_329 = None
        unsqueeze_330: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_499, -1);  primals_499 = None
        unsqueeze_331: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        add_440: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_580, unsqueeze_331);  mul_580 = unsqueeze_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_79: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_440);  add_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_83: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_79, primals_500, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_441: "i64[]" = torch.ops.aten.add.Tensor(primals_501, 1)
        var_mean_83 = torch.ops.aten.var_mean.correction(convolution_83, [0, 2, 3], correction = 0, keepdim = True)
        getitem_168: "f32[1, 256, 1, 1]" = var_mean_83[0]
        getitem_169: "f32[1, 256, 1, 1]" = var_mean_83[1];  var_mean_83 = None
        add_442: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-05)
        rsqrt_83: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_442);  add_442 = None
        sub_83: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, getitem_169)
        mul_581: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = None
        squeeze_249: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_169, [0, 2, 3]);  getitem_169 = None
        squeeze_250: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_83, [0, 2, 3]);  rsqrt_83 = None
        mul_582: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_249, 0.1)
        mul_583: "f32[256]" = torch.ops.aten.mul.Tensor(primals_502, 0.9)
        add_443: "f32[256]" = torch.ops.aten.add.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
        squeeze_251: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
        mul_584: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_251, 1.0001594642002871);  squeeze_251 = None
        mul_585: "f32[256]" = torch.ops.aten.mul.Tensor(mul_584, 0.1);  mul_584 = None
        mul_586: "f32[256]" = torch.ops.aten.mul.Tensor(primals_503, 0.9)
        add_444: "f32[256]" = torch.ops.aten.add.Tensor(mul_585, mul_586);  mul_585 = mul_586 = None
        unsqueeze_332: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_504, -1)
        unsqueeze_333: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_587: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_581, unsqueeze_333);  mul_581 = unsqueeze_333 = None
        unsqueeze_334: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_505, -1);  primals_505 = None
        unsqueeze_335: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_445: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_587, unsqueeze_335);  mul_587 = unsqueeze_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_80: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_445);  add_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_84: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_80, primals_506, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_446: "i64[]" = torch.ops.aten.add.Tensor(primals_507, 1)
        var_mean_84 = torch.ops.aten.var_mean.correction(convolution_84, [0, 2, 3], correction = 0, keepdim = True)
        getitem_170: "f32[1, 1024, 1, 1]" = var_mean_84[0]
        getitem_171: "f32[1, 1024, 1, 1]" = var_mean_84[1];  var_mean_84 = None
        add_447: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-05)
        rsqrt_84: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_447);  add_447 = None
        sub_84: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, getitem_171)
        mul_588: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = None
        squeeze_252: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_171, [0, 2, 3]);  getitem_171 = None
        squeeze_253: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_84, [0, 2, 3]);  rsqrt_84 = None
        mul_589: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_252, 0.1)
        mul_590: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_508, 0.9)
        add_448: "f32[1024]" = torch.ops.aten.add.Tensor(mul_589, mul_590);  mul_589 = mul_590 = None
        squeeze_254: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_170, [0, 2, 3]);  getitem_170 = None
        mul_591: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_254, 1.0001594642002871);  squeeze_254 = None
        mul_592: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_591, 0.1);  mul_591 = None
        mul_593: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_509, 0.9)
        add_449: "f32[1024]" = torch.ops.aten.add.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
        unsqueeze_336: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_510, -1)
        unsqueeze_337: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        mul_594: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_588, unsqueeze_337);  mul_588 = unsqueeze_337 = None
        unsqueeze_338: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_511, -1);  primals_511 = None
        unsqueeze_339: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        add_450: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_594, unsqueeze_339);  mul_594 = unsqueeze_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_451: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_450, relu_78);  add_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_81: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_451);  add_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_85: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_81, primals_512, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_452: "i64[]" = torch.ops.aten.add.Tensor(primals_513, 1)
        var_mean_85 = torch.ops.aten.var_mean.correction(convolution_85, [0, 2, 3], correction = 0, keepdim = True)
        getitem_172: "f32[1, 256, 1, 1]" = var_mean_85[0]
        getitem_173: "f32[1, 256, 1, 1]" = var_mean_85[1];  var_mean_85 = None
        add_453: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-05)
        rsqrt_85: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_453);  add_453 = None
        sub_85: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, getitem_173)
        mul_595: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = None
        squeeze_255: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_173, [0, 2, 3]);  getitem_173 = None
        squeeze_256: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_85, [0, 2, 3]);  rsqrt_85 = None
        mul_596: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_255, 0.1)
        mul_597: "f32[256]" = torch.ops.aten.mul.Tensor(primals_514, 0.9)
        add_454: "f32[256]" = torch.ops.aten.add.Tensor(mul_596, mul_597);  mul_596 = mul_597 = None
        squeeze_257: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_172, [0, 2, 3]);  getitem_172 = None
        mul_598: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_257, 1.0001594642002871);  squeeze_257 = None
        mul_599: "f32[256]" = torch.ops.aten.mul.Tensor(mul_598, 0.1);  mul_598 = None
        mul_600: "f32[256]" = torch.ops.aten.mul.Tensor(primals_515, 0.9)
        add_455: "f32[256]" = torch.ops.aten.add.Tensor(mul_599, mul_600);  mul_599 = mul_600 = None
        unsqueeze_340: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_516, -1)
        unsqueeze_341: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_601: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_595, unsqueeze_341);  mul_595 = unsqueeze_341 = None
        unsqueeze_342: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_517, -1);  primals_517 = None
        unsqueeze_343: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_456: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_601, unsqueeze_343);  mul_601 = unsqueeze_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_82: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_456);  add_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_86: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_82, primals_518, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_457: "i64[]" = torch.ops.aten.add.Tensor(primals_519, 1)
        var_mean_86 = torch.ops.aten.var_mean.correction(convolution_86, [0, 2, 3], correction = 0, keepdim = True)
        getitem_174: "f32[1, 256, 1, 1]" = var_mean_86[0]
        getitem_175: "f32[1, 256, 1, 1]" = var_mean_86[1];  var_mean_86 = None
        add_458: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05)
        rsqrt_86: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_458);  add_458 = None
        sub_86: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, getitem_175)
        mul_602: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = None
        squeeze_258: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_175, [0, 2, 3]);  getitem_175 = None
        squeeze_259: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_86, [0, 2, 3]);  rsqrt_86 = None
        mul_603: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_258, 0.1)
        mul_604: "f32[256]" = torch.ops.aten.mul.Tensor(primals_520, 0.9)
        add_459: "f32[256]" = torch.ops.aten.add.Tensor(mul_603, mul_604);  mul_603 = mul_604 = None
        squeeze_260: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_174, [0, 2, 3]);  getitem_174 = None
        mul_605: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_260, 1.0001594642002871);  squeeze_260 = None
        mul_606: "f32[256]" = torch.ops.aten.mul.Tensor(mul_605, 0.1);  mul_605 = None
        mul_607: "f32[256]" = torch.ops.aten.mul.Tensor(primals_521, 0.9)
        add_460: "f32[256]" = torch.ops.aten.add.Tensor(mul_606, mul_607);  mul_606 = mul_607 = None
        unsqueeze_344: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_522, -1)
        unsqueeze_345: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        mul_608: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_602, unsqueeze_345);  mul_602 = unsqueeze_345 = None
        unsqueeze_346: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_523, -1);  primals_523 = None
        unsqueeze_347: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        add_461: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_608, unsqueeze_347);  mul_608 = unsqueeze_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_83: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_461);  add_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_87: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_83, primals_524, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_462: "i64[]" = torch.ops.aten.add.Tensor(primals_525, 1)
        var_mean_87 = torch.ops.aten.var_mean.correction(convolution_87, [0, 2, 3], correction = 0, keepdim = True)
        getitem_176: "f32[1, 1024, 1, 1]" = var_mean_87[0]
        getitem_177: "f32[1, 1024, 1, 1]" = var_mean_87[1];  var_mean_87 = None
        add_463: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05)
        rsqrt_87: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_463);  add_463 = None
        sub_87: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, getitem_177)
        mul_609: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = None
        squeeze_261: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_177, [0, 2, 3]);  getitem_177 = None
        squeeze_262: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_87, [0, 2, 3]);  rsqrt_87 = None
        mul_610: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_261, 0.1)
        mul_611: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_526, 0.9)
        add_464: "f32[1024]" = torch.ops.aten.add.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
        squeeze_263: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_176, [0, 2, 3]);  getitem_176 = None
        mul_612: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_263, 1.0001594642002871);  squeeze_263 = None
        mul_613: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_612, 0.1);  mul_612 = None
        mul_614: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_527, 0.9)
        add_465: "f32[1024]" = torch.ops.aten.add.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
        unsqueeze_348: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_528, -1)
        unsqueeze_349: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_615: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_609, unsqueeze_349);  mul_609 = unsqueeze_349 = None
        unsqueeze_350: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_529, -1);  primals_529 = None
        unsqueeze_351: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_466: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_615, unsqueeze_351);  mul_615 = unsqueeze_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_467: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_466, relu_81);  add_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_84: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_467);  add_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_88: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_84, primals_530, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_468: "i64[]" = torch.ops.aten.add.Tensor(primals_531, 1)
        var_mean_88 = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True)
        getitem_178: "f32[1, 256, 1, 1]" = var_mean_88[0]
        getitem_179: "f32[1, 256, 1, 1]" = var_mean_88[1];  var_mean_88 = None
        add_469: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-05)
        rsqrt_88: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_469);  add_469 = None
        sub_88: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_179)
        mul_616: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = None
        squeeze_264: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_179, [0, 2, 3]);  getitem_179 = None
        squeeze_265: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_88, [0, 2, 3]);  rsqrt_88 = None
        mul_617: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_264, 0.1)
        mul_618: "f32[256]" = torch.ops.aten.mul.Tensor(primals_532, 0.9)
        add_470: "f32[256]" = torch.ops.aten.add.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
        squeeze_266: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_178, [0, 2, 3]);  getitem_178 = None
        mul_619: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_266, 1.0001594642002871);  squeeze_266 = None
        mul_620: "f32[256]" = torch.ops.aten.mul.Tensor(mul_619, 0.1);  mul_619 = None
        mul_621: "f32[256]" = torch.ops.aten.mul.Tensor(primals_533, 0.9)
        add_471: "f32[256]" = torch.ops.aten.add.Tensor(mul_620, mul_621);  mul_620 = mul_621 = None
        unsqueeze_352: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_534, -1)
        unsqueeze_353: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        mul_622: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_616, unsqueeze_353);  mul_616 = unsqueeze_353 = None
        unsqueeze_354: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_535, -1);  primals_535 = None
        unsqueeze_355: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        add_472: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_622, unsqueeze_355);  mul_622 = unsqueeze_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_85: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_472);  add_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_89: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_85, primals_536, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_473: "i64[]" = torch.ops.aten.add.Tensor(primals_537, 1)
        var_mean_89 = torch.ops.aten.var_mean.correction(convolution_89, [0, 2, 3], correction = 0, keepdim = True)
        getitem_180: "f32[1, 256, 1, 1]" = var_mean_89[0]
        getitem_181: "f32[1, 256, 1, 1]" = var_mean_89[1];  var_mean_89 = None
        add_474: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-05)
        rsqrt_89: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_474);  add_474 = None
        sub_89: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, getitem_181)
        mul_623: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = None
        squeeze_267: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_181, [0, 2, 3]);  getitem_181 = None
        squeeze_268: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_89, [0, 2, 3]);  rsqrt_89 = None
        mul_624: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_267, 0.1)
        mul_625: "f32[256]" = torch.ops.aten.mul.Tensor(primals_538, 0.9)
        add_475: "f32[256]" = torch.ops.aten.add.Tensor(mul_624, mul_625);  mul_624 = mul_625 = None
        squeeze_269: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_180, [0, 2, 3]);  getitem_180 = None
        mul_626: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_269, 1.0001594642002871);  squeeze_269 = None
        mul_627: "f32[256]" = torch.ops.aten.mul.Tensor(mul_626, 0.1);  mul_626 = None
        mul_628: "f32[256]" = torch.ops.aten.mul.Tensor(primals_539, 0.9)
        add_476: "f32[256]" = torch.ops.aten.add.Tensor(mul_627, mul_628);  mul_627 = mul_628 = None
        unsqueeze_356: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_540, -1)
        unsqueeze_357: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_629: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_623, unsqueeze_357);  mul_623 = unsqueeze_357 = None
        unsqueeze_358: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_541, -1);  primals_541 = None
        unsqueeze_359: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_477: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_629, unsqueeze_359);  mul_629 = unsqueeze_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_86: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_477);  add_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_90: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_86, primals_542, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_478: "i64[]" = torch.ops.aten.add.Tensor(primals_543, 1)
        var_mean_90 = torch.ops.aten.var_mean.correction(convolution_90, [0, 2, 3], correction = 0, keepdim = True)
        getitem_182: "f32[1, 1024, 1, 1]" = var_mean_90[0]
        getitem_183: "f32[1, 1024, 1, 1]" = var_mean_90[1];  var_mean_90 = None
        add_479: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_182, 1e-05)
        rsqrt_90: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_479);  add_479 = None
        sub_90: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, getitem_183)
        mul_630: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = None
        squeeze_270: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_183, [0, 2, 3]);  getitem_183 = None
        squeeze_271: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_90, [0, 2, 3]);  rsqrt_90 = None
        mul_631: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_270, 0.1)
        mul_632: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_544, 0.9)
        add_480: "f32[1024]" = torch.ops.aten.add.Tensor(mul_631, mul_632);  mul_631 = mul_632 = None
        squeeze_272: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_182, [0, 2, 3]);  getitem_182 = None
        mul_633: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_272, 1.0001594642002871);  squeeze_272 = None
        mul_634: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_633, 0.1);  mul_633 = None
        mul_635: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_545, 0.9)
        add_481: "f32[1024]" = torch.ops.aten.add.Tensor(mul_634, mul_635);  mul_634 = mul_635 = None
        unsqueeze_360: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_546, -1)
        unsqueeze_361: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        mul_636: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_630, unsqueeze_361);  mul_630 = unsqueeze_361 = None
        unsqueeze_362: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_547, -1);  primals_547 = None
        unsqueeze_363: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        add_482: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_636, unsqueeze_363);  mul_636 = unsqueeze_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_483: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_482, relu_84);  add_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_87: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_483);  add_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_91: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_87, primals_548, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_484: "i64[]" = torch.ops.aten.add.Tensor(primals_549, 1)
        var_mean_91 = torch.ops.aten.var_mean.correction(convolution_91, [0, 2, 3], correction = 0, keepdim = True)
        getitem_184: "f32[1, 256, 1, 1]" = var_mean_91[0]
        getitem_185: "f32[1, 256, 1, 1]" = var_mean_91[1];  var_mean_91 = None
        add_485: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_184, 1e-05)
        rsqrt_91: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_485);  add_485 = None
        sub_91: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, getitem_185)
        mul_637: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = None
        squeeze_273: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_185, [0, 2, 3]);  getitem_185 = None
        squeeze_274: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_91, [0, 2, 3]);  rsqrt_91 = None
        mul_638: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_273, 0.1)
        mul_639: "f32[256]" = torch.ops.aten.mul.Tensor(primals_550, 0.9)
        add_486: "f32[256]" = torch.ops.aten.add.Tensor(mul_638, mul_639);  mul_638 = mul_639 = None
        squeeze_275: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_184, [0, 2, 3]);  getitem_184 = None
        mul_640: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_275, 1.0001594642002871);  squeeze_275 = None
        mul_641: "f32[256]" = torch.ops.aten.mul.Tensor(mul_640, 0.1);  mul_640 = None
        mul_642: "f32[256]" = torch.ops.aten.mul.Tensor(primals_551, 0.9)
        add_487: "f32[256]" = torch.ops.aten.add.Tensor(mul_641, mul_642);  mul_641 = mul_642 = None
        unsqueeze_364: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_552, -1)
        unsqueeze_365: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_643: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_637, unsqueeze_365);  mul_637 = unsqueeze_365 = None
        unsqueeze_366: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_553, -1);  primals_553 = None
        unsqueeze_367: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_488: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_643, unsqueeze_367);  mul_643 = unsqueeze_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_88: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_488);  add_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_92: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_88, primals_554, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_489: "i64[]" = torch.ops.aten.add.Tensor(primals_555, 1)
        var_mean_92 = torch.ops.aten.var_mean.correction(convolution_92, [0, 2, 3], correction = 0, keepdim = True)
        getitem_186: "f32[1, 256, 1, 1]" = var_mean_92[0]
        getitem_187: "f32[1, 256, 1, 1]" = var_mean_92[1];  var_mean_92 = None
        add_490: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-05)
        rsqrt_92: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_490);  add_490 = None
        sub_92: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, getitem_187)
        mul_644: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = None
        squeeze_276: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_187, [0, 2, 3]);  getitem_187 = None
        squeeze_277: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_92, [0, 2, 3]);  rsqrt_92 = None
        mul_645: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_276, 0.1)
        mul_646: "f32[256]" = torch.ops.aten.mul.Tensor(primals_556, 0.9)
        add_491: "f32[256]" = torch.ops.aten.add.Tensor(mul_645, mul_646);  mul_645 = mul_646 = None
        squeeze_278: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_186, [0, 2, 3]);  getitem_186 = None
        mul_647: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_278, 1.0001594642002871);  squeeze_278 = None
        mul_648: "f32[256]" = torch.ops.aten.mul.Tensor(mul_647, 0.1);  mul_647 = None
        mul_649: "f32[256]" = torch.ops.aten.mul.Tensor(primals_557, 0.9)
        add_492: "f32[256]" = torch.ops.aten.add.Tensor(mul_648, mul_649);  mul_648 = mul_649 = None
        unsqueeze_368: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_558, -1)
        unsqueeze_369: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        mul_650: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_644, unsqueeze_369);  mul_644 = unsqueeze_369 = None
        unsqueeze_370: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_559, -1);  primals_559 = None
        unsqueeze_371: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        add_493: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_650, unsqueeze_371);  mul_650 = unsqueeze_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_89: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_493);  add_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_93: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_89, primals_560, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_494: "i64[]" = torch.ops.aten.add.Tensor(primals_561, 1)
        var_mean_93 = torch.ops.aten.var_mean.correction(convolution_93, [0, 2, 3], correction = 0, keepdim = True)
        getitem_188: "f32[1, 1024, 1, 1]" = var_mean_93[0]
        getitem_189: "f32[1, 1024, 1, 1]" = var_mean_93[1];  var_mean_93 = None
        add_495: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-05)
        rsqrt_93: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_495);  add_495 = None
        sub_93: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, getitem_189)
        mul_651: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = None
        squeeze_279: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_189, [0, 2, 3]);  getitem_189 = None
        squeeze_280: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_93, [0, 2, 3]);  rsqrt_93 = None
        mul_652: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_279, 0.1)
        mul_653: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_562, 0.9)
        add_496: "f32[1024]" = torch.ops.aten.add.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
        squeeze_281: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_188, [0, 2, 3]);  getitem_188 = None
        mul_654: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_281, 1.0001594642002871);  squeeze_281 = None
        mul_655: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_654, 0.1);  mul_654 = None
        mul_656: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_563, 0.9)
        add_497: "f32[1024]" = torch.ops.aten.add.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
        unsqueeze_372: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_564, -1)
        unsqueeze_373: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_657: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_651, unsqueeze_373);  mul_651 = unsqueeze_373 = None
        unsqueeze_374: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_565, -1);  primals_565 = None
        unsqueeze_375: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_498: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_657, unsqueeze_375);  mul_657 = unsqueeze_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_499: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_498, relu_87);  add_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_90: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_499);  add_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_94: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_90, primals_566, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_500: "i64[]" = torch.ops.aten.add.Tensor(primals_567, 1)
        var_mean_94 = torch.ops.aten.var_mean.correction(convolution_94, [0, 2, 3], correction = 0, keepdim = True)
        getitem_190: "f32[1, 256, 1, 1]" = var_mean_94[0]
        getitem_191: "f32[1, 256, 1, 1]" = var_mean_94[1];  var_mean_94 = None
        add_501: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-05)
        rsqrt_94: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_501);  add_501 = None
        sub_94: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, getitem_191)
        mul_658: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = None
        squeeze_282: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_191, [0, 2, 3]);  getitem_191 = None
        squeeze_283: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_94, [0, 2, 3]);  rsqrt_94 = None
        mul_659: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_282, 0.1)
        mul_660: "f32[256]" = torch.ops.aten.mul.Tensor(primals_568, 0.9)
        add_502: "f32[256]" = torch.ops.aten.add.Tensor(mul_659, mul_660);  mul_659 = mul_660 = None
        squeeze_284: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_190, [0, 2, 3]);  getitem_190 = None
        mul_661: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_284, 1.0001594642002871);  squeeze_284 = None
        mul_662: "f32[256]" = torch.ops.aten.mul.Tensor(mul_661, 0.1);  mul_661 = None
        mul_663: "f32[256]" = torch.ops.aten.mul.Tensor(primals_569, 0.9)
        add_503: "f32[256]" = torch.ops.aten.add.Tensor(mul_662, mul_663);  mul_662 = mul_663 = None
        unsqueeze_376: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_570, -1)
        unsqueeze_377: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        mul_664: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_658, unsqueeze_377);  mul_658 = unsqueeze_377 = None
        unsqueeze_378: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_571, -1);  primals_571 = None
        unsqueeze_379: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        add_504: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_664, unsqueeze_379);  mul_664 = unsqueeze_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_91: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_504);  add_504 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_95: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_91, primals_572, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_505: "i64[]" = torch.ops.aten.add.Tensor(primals_573, 1)
        var_mean_95 = torch.ops.aten.var_mean.correction(convolution_95, [0, 2, 3], correction = 0, keepdim = True)
        getitem_192: "f32[1, 256, 1, 1]" = var_mean_95[0]
        getitem_193: "f32[1, 256, 1, 1]" = var_mean_95[1];  var_mean_95 = None
        add_506: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-05)
        rsqrt_95: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_506);  add_506 = None
        sub_95: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_95, getitem_193)
        mul_665: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = None
        squeeze_285: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_193, [0, 2, 3]);  getitem_193 = None
        squeeze_286: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_95, [0, 2, 3]);  rsqrt_95 = None
        mul_666: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_285, 0.1)
        mul_667: "f32[256]" = torch.ops.aten.mul.Tensor(primals_574, 0.9)
        add_507: "f32[256]" = torch.ops.aten.add.Tensor(mul_666, mul_667);  mul_666 = mul_667 = None
        squeeze_287: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_192, [0, 2, 3]);  getitem_192 = None
        mul_668: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_287, 1.0001594642002871);  squeeze_287 = None
        mul_669: "f32[256]" = torch.ops.aten.mul.Tensor(mul_668, 0.1);  mul_668 = None
        mul_670: "f32[256]" = torch.ops.aten.mul.Tensor(primals_575, 0.9)
        add_508: "f32[256]" = torch.ops.aten.add.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
        unsqueeze_380: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_576, -1)
        unsqueeze_381: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_671: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_665, unsqueeze_381);  mul_665 = unsqueeze_381 = None
        unsqueeze_382: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_577, -1);  primals_577 = None
        unsqueeze_383: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_509: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_671, unsqueeze_383);  mul_671 = unsqueeze_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_92: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_509);  add_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_96: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_92, primals_578, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_510: "i64[]" = torch.ops.aten.add.Tensor(primals_579, 1)
        var_mean_96 = torch.ops.aten.var_mean.correction(convolution_96, [0, 2, 3], correction = 0, keepdim = True)
        getitem_194: "f32[1, 1024, 1, 1]" = var_mean_96[0]
        getitem_195: "f32[1, 1024, 1, 1]" = var_mean_96[1];  var_mean_96 = None
        add_511: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_194, 1e-05)
        rsqrt_96: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_511);  add_511 = None
        sub_96: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_96, getitem_195)
        mul_672: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_96);  sub_96 = None
        squeeze_288: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_195, [0, 2, 3]);  getitem_195 = None
        squeeze_289: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_96, [0, 2, 3]);  rsqrt_96 = None
        mul_673: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_288, 0.1)
        mul_674: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_580, 0.9)
        add_512: "f32[1024]" = torch.ops.aten.add.Tensor(mul_673, mul_674);  mul_673 = mul_674 = None
        squeeze_290: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_194, [0, 2, 3]);  getitem_194 = None
        mul_675: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_290, 1.0001594642002871);  squeeze_290 = None
        mul_676: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_675, 0.1);  mul_675 = None
        mul_677: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_581, 0.9)
        add_513: "f32[1024]" = torch.ops.aten.add.Tensor(mul_676, mul_677);  mul_676 = mul_677 = None
        unsqueeze_384: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_582, -1)
        unsqueeze_385: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        mul_678: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_672, unsqueeze_385);  mul_672 = unsqueeze_385 = None
        unsqueeze_386: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_583, -1);  primals_583 = None
        unsqueeze_387: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        add_514: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_678, unsqueeze_387);  mul_678 = unsqueeze_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_515: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_514, relu_90);  add_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_93: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_515);  add_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_97: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_93, primals_584, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_516: "i64[]" = torch.ops.aten.add.Tensor(primals_585, 1)
        var_mean_97 = torch.ops.aten.var_mean.correction(convolution_97, [0, 2, 3], correction = 0, keepdim = True)
        getitem_196: "f32[1, 256, 1, 1]" = var_mean_97[0]
        getitem_197: "f32[1, 256, 1, 1]" = var_mean_97[1];  var_mean_97 = None
        add_517: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_196, 1e-05)
        rsqrt_97: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_517);  add_517 = None
        sub_97: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_97, getitem_197)
        mul_679: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_97);  sub_97 = None
        squeeze_291: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_197, [0, 2, 3]);  getitem_197 = None
        squeeze_292: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_97, [0, 2, 3]);  rsqrt_97 = None
        mul_680: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_291, 0.1)
        mul_681: "f32[256]" = torch.ops.aten.mul.Tensor(primals_586, 0.9)
        add_518: "f32[256]" = torch.ops.aten.add.Tensor(mul_680, mul_681);  mul_680 = mul_681 = None
        squeeze_293: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_196, [0, 2, 3]);  getitem_196 = None
        mul_682: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_293, 1.0001594642002871);  squeeze_293 = None
        mul_683: "f32[256]" = torch.ops.aten.mul.Tensor(mul_682, 0.1);  mul_682 = None
        mul_684: "f32[256]" = torch.ops.aten.mul.Tensor(primals_587, 0.9)
        add_519: "f32[256]" = torch.ops.aten.add.Tensor(mul_683, mul_684);  mul_683 = mul_684 = None
        unsqueeze_388: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_588, -1)
        unsqueeze_389: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_685: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_679, unsqueeze_389);  mul_679 = unsqueeze_389 = None
        unsqueeze_390: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_589, -1);  primals_589 = None
        unsqueeze_391: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_520: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_685, unsqueeze_391);  mul_685 = unsqueeze_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_94: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_520);  add_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_98: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_94, primals_590, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_521: "i64[]" = torch.ops.aten.add.Tensor(primals_591, 1)
        var_mean_98 = torch.ops.aten.var_mean.correction(convolution_98, [0, 2, 3], correction = 0, keepdim = True)
        getitem_198: "f32[1, 256, 1, 1]" = var_mean_98[0]
        getitem_199: "f32[1, 256, 1, 1]" = var_mean_98[1];  var_mean_98 = None
        add_522: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-05)
        rsqrt_98: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_522);  add_522 = None
        sub_98: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_98, getitem_199)
        mul_686: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_98);  sub_98 = None
        squeeze_294: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_199, [0, 2, 3]);  getitem_199 = None
        squeeze_295: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_98, [0, 2, 3]);  rsqrt_98 = None
        mul_687: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_294, 0.1)
        mul_688: "f32[256]" = torch.ops.aten.mul.Tensor(primals_592, 0.9)
        add_523: "f32[256]" = torch.ops.aten.add.Tensor(mul_687, mul_688);  mul_687 = mul_688 = None
        squeeze_296: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_198, [0, 2, 3]);  getitem_198 = None
        mul_689: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_296, 1.0001594642002871);  squeeze_296 = None
        mul_690: "f32[256]" = torch.ops.aten.mul.Tensor(mul_689, 0.1);  mul_689 = None
        mul_691: "f32[256]" = torch.ops.aten.mul.Tensor(primals_593, 0.9)
        add_524: "f32[256]" = torch.ops.aten.add.Tensor(mul_690, mul_691);  mul_690 = mul_691 = None
        unsqueeze_392: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_594, -1)
        unsqueeze_393: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        mul_692: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_686, unsqueeze_393);  mul_686 = unsqueeze_393 = None
        unsqueeze_394: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_595, -1);  primals_595 = None
        unsqueeze_395: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        add_525: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_692, unsqueeze_395);  mul_692 = unsqueeze_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_95: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_525);  add_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_99: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_95, primals_596, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_526: "i64[]" = torch.ops.aten.add.Tensor(primals_597, 1)
        var_mean_99 = torch.ops.aten.var_mean.correction(convolution_99, [0, 2, 3], correction = 0, keepdim = True)
        getitem_200: "f32[1, 1024, 1, 1]" = var_mean_99[0]
        getitem_201: "f32[1, 1024, 1, 1]" = var_mean_99[1];  var_mean_99 = None
        add_527: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-05)
        rsqrt_99: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_527);  add_527 = None
        sub_99: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_99, getitem_201)
        mul_693: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_99);  sub_99 = None
        squeeze_297: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_201, [0, 2, 3]);  getitem_201 = None
        squeeze_298: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_99, [0, 2, 3]);  rsqrt_99 = None
        mul_694: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_297, 0.1)
        mul_695: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_598, 0.9)
        add_528: "f32[1024]" = torch.ops.aten.add.Tensor(mul_694, mul_695);  mul_694 = mul_695 = None
        squeeze_299: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_200, [0, 2, 3]);  getitem_200 = None
        mul_696: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_299, 1.0001594642002871);  squeeze_299 = None
        mul_697: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_696, 0.1);  mul_696 = None
        mul_698: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_599, 0.9)
        add_529: "f32[1024]" = torch.ops.aten.add.Tensor(mul_697, mul_698);  mul_697 = mul_698 = None
        unsqueeze_396: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_600, -1)
        unsqueeze_397: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_699: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_693, unsqueeze_397);  mul_693 = unsqueeze_397 = None
        unsqueeze_398: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_601, -1);  primals_601 = None
        unsqueeze_399: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_530: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_699, unsqueeze_399);  mul_699 = unsqueeze_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_531: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_530, relu_93);  add_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_96: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_531);  add_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_100: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_96, primals_602, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_532: "i64[]" = torch.ops.aten.add.Tensor(primals_603, 1)
        var_mean_100 = torch.ops.aten.var_mean.correction(convolution_100, [0, 2, 3], correction = 0, keepdim = True)
        getitem_202: "f32[1, 256, 1, 1]" = var_mean_100[0]
        getitem_203: "f32[1, 256, 1, 1]" = var_mean_100[1];  var_mean_100 = None
        add_533: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_202, 1e-05)
        rsqrt_100: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_533);  add_533 = None
        sub_100: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_100, getitem_203)
        mul_700: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_100);  sub_100 = None
        squeeze_300: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_203, [0, 2, 3]);  getitem_203 = None
        squeeze_301: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_100, [0, 2, 3]);  rsqrt_100 = None
        mul_701: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_300, 0.1)
        mul_702: "f32[256]" = torch.ops.aten.mul.Tensor(primals_604, 0.9)
        add_534: "f32[256]" = torch.ops.aten.add.Tensor(mul_701, mul_702);  mul_701 = mul_702 = None
        squeeze_302: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_202, [0, 2, 3]);  getitem_202 = None
        mul_703: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_302, 1.0001594642002871);  squeeze_302 = None
        mul_704: "f32[256]" = torch.ops.aten.mul.Tensor(mul_703, 0.1);  mul_703 = None
        mul_705: "f32[256]" = torch.ops.aten.mul.Tensor(primals_605, 0.9)
        add_535: "f32[256]" = torch.ops.aten.add.Tensor(mul_704, mul_705);  mul_704 = mul_705 = None
        unsqueeze_400: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_606, -1)
        unsqueeze_401: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        mul_706: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_700, unsqueeze_401);  mul_700 = unsqueeze_401 = None
        unsqueeze_402: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_607, -1);  primals_607 = None
        unsqueeze_403: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        add_536: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_706, unsqueeze_403);  mul_706 = unsqueeze_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_97: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_536);  add_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_101: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_97, primals_608, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_537: "i64[]" = torch.ops.aten.add.Tensor(primals_609, 1)
        var_mean_101 = torch.ops.aten.var_mean.correction(convolution_101, [0, 2, 3], correction = 0, keepdim = True)
        getitem_204: "f32[1, 256, 1, 1]" = var_mean_101[0]
        getitem_205: "f32[1, 256, 1, 1]" = var_mean_101[1];  var_mean_101 = None
        add_538: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_204, 1e-05)
        rsqrt_101: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_538);  add_538 = None
        sub_101: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_101, getitem_205)
        mul_707: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_101);  sub_101 = None
        squeeze_303: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_205, [0, 2, 3]);  getitem_205 = None
        squeeze_304: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_101, [0, 2, 3]);  rsqrt_101 = None
        mul_708: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_303, 0.1)
        mul_709: "f32[256]" = torch.ops.aten.mul.Tensor(primals_610, 0.9)
        add_539: "f32[256]" = torch.ops.aten.add.Tensor(mul_708, mul_709);  mul_708 = mul_709 = None
        squeeze_305: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_204, [0, 2, 3]);  getitem_204 = None
        mul_710: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_305, 1.0001594642002871);  squeeze_305 = None
        mul_711: "f32[256]" = torch.ops.aten.mul.Tensor(mul_710, 0.1);  mul_710 = None
        mul_712: "f32[256]" = torch.ops.aten.mul.Tensor(primals_611, 0.9)
        add_540: "f32[256]" = torch.ops.aten.add.Tensor(mul_711, mul_712);  mul_711 = mul_712 = None
        unsqueeze_404: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_612, -1)
        unsqueeze_405: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_713: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_707, unsqueeze_405);  mul_707 = unsqueeze_405 = None
        unsqueeze_406: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_613, -1);  primals_613 = None
        unsqueeze_407: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_541: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_713, unsqueeze_407);  mul_713 = unsqueeze_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_98: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_541);  add_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_102: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_98, primals_614, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_542: "i64[]" = torch.ops.aten.add.Tensor(primals_615, 1)
        var_mean_102 = torch.ops.aten.var_mean.correction(convolution_102, [0, 2, 3], correction = 0, keepdim = True)
        getitem_206: "f32[1, 1024, 1, 1]" = var_mean_102[0]
        getitem_207: "f32[1, 1024, 1, 1]" = var_mean_102[1];  var_mean_102 = None
        add_543: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-05)
        rsqrt_102: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_543);  add_543 = None
        sub_102: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_102, getitem_207)
        mul_714: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_102);  sub_102 = None
        squeeze_306: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_207, [0, 2, 3]);  getitem_207 = None
        squeeze_307: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_102, [0, 2, 3]);  rsqrt_102 = None
        mul_715: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_306, 0.1)
        mul_716: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_616, 0.9)
        add_544: "f32[1024]" = torch.ops.aten.add.Tensor(mul_715, mul_716);  mul_715 = mul_716 = None
        squeeze_308: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_206, [0, 2, 3]);  getitem_206 = None
        mul_717: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_308, 1.0001594642002871);  squeeze_308 = None
        mul_718: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_717, 0.1);  mul_717 = None
        mul_719: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_617, 0.9)
        add_545: "f32[1024]" = torch.ops.aten.add.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
        unsqueeze_408: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_618, -1)
        unsqueeze_409: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        mul_720: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_714, unsqueeze_409);  mul_714 = unsqueeze_409 = None
        unsqueeze_410: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_619, -1);  primals_619 = None
        unsqueeze_411: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        add_546: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_720, unsqueeze_411);  mul_720 = unsqueeze_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_547: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_546, relu_96);  add_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_99: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_547);  add_547 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_103: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_99, primals_620, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_548: "i64[]" = torch.ops.aten.add.Tensor(primals_621, 1)
        var_mean_103 = torch.ops.aten.var_mean.correction(convolution_103, [0, 2, 3], correction = 0, keepdim = True)
        getitem_208: "f32[1, 256, 1, 1]" = var_mean_103[0]
        getitem_209: "f32[1, 256, 1, 1]" = var_mean_103[1];  var_mean_103 = None
        add_549: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-05)
        rsqrt_103: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_549);  add_549 = None
        sub_103: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_103, getitem_209)
        mul_721: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_103);  sub_103 = None
        squeeze_309: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_209, [0, 2, 3]);  getitem_209 = None
        squeeze_310: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_103, [0, 2, 3]);  rsqrt_103 = None
        mul_722: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_309, 0.1)
        mul_723: "f32[256]" = torch.ops.aten.mul.Tensor(primals_622, 0.9)
        add_550: "f32[256]" = torch.ops.aten.add.Tensor(mul_722, mul_723);  mul_722 = mul_723 = None
        squeeze_311: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_208, [0, 2, 3]);  getitem_208 = None
        mul_724: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_311, 1.0001594642002871);  squeeze_311 = None
        mul_725: "f32[256]" = torch.ops.aten.mul.Tensor(mul_724, 0.1);  mul_724 = None
        mul_726: "f32[256]" = torch.ops.aten.mul.Tensor(primals_623, 0.9)
        add_551: "f32[256]" = torch.ops.aten.add.Tensor(mul_725, mul_726);  mul_725 = mul_726 = None
        unsqueeze_412: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_624, -1)
        unsqueeze_413: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_727: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_721, unsqueeze_413);  mul_721 = unsqueeze_413 = None
        unsqueeze_414: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_625, -1);  primals_625 = None
        unsqueeze_415: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_552: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_727, unsqueeze_415);  mul_727 = unsqueeze_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_100: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_552);  add_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_104: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_100, primals_626, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_553: "i64[]" = torch.ops.aten.add.Tensor(primals_627, 1)
        var_mean_104 = torch.ops.aten.var_mean.correction(convolution_104, [0, 2, 3], correction = 0, keepdim = True)
        getitem_210: "f32[1, 256, 1, 1]" = var_mean_104[0]
        getitem_211: "f32[1, 256, 1, 1]" = var_mean_104[1];  var_mean_104 = None
        add_554: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-05)
        rsqrt_104: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_554);  add_554 = None
        sub_104: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_104, getitem_211)
        mul_728: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_104);  sub_104 = None
        squeeze_312: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_211, [0, 2, 3]);  getitem_211 = None
        squeeze_313: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_104, [0, 2, 3]);  rsqrt_104 = None
        mul_729: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_312, 0.1)
        mul_730: "f32[256]" = torch.ops.aten.mul.Tensor(primals_628, 0.9)
        add_555: "f32[256]" = torch.ops.aten.add.Tensor(mul_729, mul_730);  mul_729 = mul_730 = None
        squeeze_314: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_210, [0, 2, 3]);  getitem_210 = None
        mul_731: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_314, 1.0001594642002871);  squeeze_314 = None
        mul_732: "f32[256]" = torch.ops.aten.mul.Tensor(mul_731, 0.1);  mul_731 = None
        mul_733: "f32[256]" = torch.ops.aten.mul.Tensor(primals_629, 0.9)
        add_556: "f32[256]" = torch.ops.aten.add.Tensor(mul_732, mul_733);  mul_732 = mul_733 = None
        unsqueeze_416: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_630, -1)
        unsqueeze_417: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        mul_734: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_728, unsqueeze_417);  mul_728 = unsqueeze_417 = None
        unsqueeze_418: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_631, -1);  primals_631 = None
        unsqueeze_419: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        add_557: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_734, unsqueeze_419);  mul_734 = unsqueeze_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_101: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_557);  add_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_105: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_101, primals_632, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_558: "i64[]" = torch.ops.aten.add.Tensor(primals_633, 1)
        var_mean_105 = torch.ops.aten.var_mean.correction(convolution_105, [0, 2, 3], correction = 0, keepdim = True)
        getitem_212: "f32[1, 1024, 1, 1]" = var_mean_105[0]
        getitem_213: "f32[1, 1024, 1, 1]" = var_mean_105[1];  var_mean_105 = None
        add_559: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-05)
        rsqrt_105: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_559);  add_559 = None
        sub_105: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_105, getitem_213)
        mul_735: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_105);  sub_105 = None
        squeeze_315: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_213, [0, 2, 3]);  getitem_213 = None
        squeeze_316: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_105, [0, 2, 3]);  rsqrt_105 = None
        mul_736: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_315, 0.1)
        mul_737: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_634, 0.9)
        add_560: "f32[1024]" = torch.ops.aten.add.Tensor(mul_736, mul_737);  mul_736 = mul_737 = None
        squeeze_317: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_212, [0, 2, 3]);  getitem_212 = None
        mul_738: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_317, 1.0001594642002871);  squeeze_317 = None
        mul_739: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_738, 0.1);  mul_738 = None
        mul_740: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_635, 0.9)
        add_561: "f32[1024]" = torch.ops.aten.add.Tensor(mul_739, mul_740);  mul_739 = mul_740 = None
        unsqueeze_420: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_636, -1)
        unsqueeze_421: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_741: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_735, unsqueeze_421);  mul_735 = unsqueeze_421 = None
        unsqueeze_422: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_637, -1);  primals_637 = None
        unsqueeze_423: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_562: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_741, unsqueeze_423);  mul_741 = unsqueeze_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_563: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_562, relu_99);  add_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_102: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_563);  add_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_106: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_102, primals_638, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_564: "i64[]" = torch.ops.aten.add.Tensor(primals_639, 1)
        var_mean_106 = torch.ops.aten.var_mean.correction(convolution_106, [0, 2, 3], correction = 0, keepdim = True)
        getitem_214: "f32[1, 256, 1, 1]" = var_mean_106[0]
        getitem_215: "f32[1, 256, 1, 1]" = var_mean_106[1];  var_mean_106 = None
        add_565: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_214, 1e-05)
        rsqrt_106: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_565);  add_565 = None
        sub_106: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_106, getitem_215)
        mul_742: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_106);  sub_106 = None
        squeeze_318: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_215, [0, 2, 3]);  getitem_215 = None
        squeeze_319: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_106, [0, 2, 3]);  rsqrt_106 = None
        mul_743: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_318, 0.1)
        mul_744: "f32[256]" = torch.ops.aten.mul.Tensor(primals_640, 0.9)
        add_566: "f32[256]" = torch.ops.aten.add.Tensor(mul_743, mul_744);  mul_743 = mul_744 = None
        squeeze_320: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_214, [0, 2, 3]);  getitem_214 = None
        mul_745: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_320, 1.0001594642002871);  squeeze_320 = None
        mul_746: "f32[256]" = torch.ops.aten.mul.Tensor(mul_745, 0.1);  mul_745 = None
        mul_747: "f32[256]" = torch.ops.aten.mul.Tensor(primals_641, 0.9)
        add_567: "f32[256]" = torch.ops.aten.add.Tensor(mul_746, mul_747);  mul_746 = mul_747 = None
        unsqueeze_424: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_642, -1)
        unsqueeze_425: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        mul_748: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_742, unsqueeze_425);  mul_742 = unsqueeze_425 = None
        unsqueeze_426: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_643, -1);  primals_643 = None
        unsqueeze_427: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        add_568: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_748, unsqueeze_427);  mul_748 = unsqueeze_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_103: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_568);  add_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_107: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_103, primals_644, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_569: "i64[]" = torch.ops.aten.add.Tensor(primals_645, 1)
        var_mean_107 = torch.ops.aten.var_mean.correction(convolution_107, [0, 2, 3], correction = 0, keepdim = True)
        getitem_216: "f32[1, 256, 1, 1]" = var_mean_107[0]
        getitem_217: "f32[1, 256, 1, 1]" = var_mean_107[1];  var_mean_107 = None
        add_570: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_216, 1e-05)
        rsqrt_107: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_570);  add_570 = None
        sub_107: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_107, getitem_217)
        mul_749: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_107);  sub_107 = None
        squeeze_321: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_217, [0, 2, 3]);  getitem_217 = None
        squeeze_322: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_107, [0, 2, 3]);  rsqrt_107 = None
        mul_750: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_321, 0.1)
        mul_751: "f32[256]" = torch.ops.aten.mul.Tensor(primals_646, 0.9)
        add_571: "f32[256]" = torch.ops.aten.add.Tensor(mul_750, mul_751);  mul_750 = mul_751 = None
        squeeze_323: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_216, [0, 2, 3]);  getitem_216 = None
        mul_752: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_323, 1.0001594642002871);  squeeze_323 = None
        mul_753: "f32[256]" = torch.ops.aten.mul.Tensor(mul_752, 0.1);  mul_752 = None
        mul_754: "f32[256]" = torch.ops.aten.mul.Tensor(primals_647, 0.9)
        add_572: "f32[256]" = torch.ops.aten.add.Tensor(mul_753, mul_754);  mul_753 = mul_754 = None
        unsqueeze_428: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_648, -1)
        unsqueeze_429: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_755: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_749, unsqueeze_429);  mul_749 = unsqueeze_429 = None
        unsqueeze_430: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_649, -1);  primals_649 = None
        unsqueeze_431: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_573: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_755, unsqueeze_431);  mul_755 = unsqueeze_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_104: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_573);  add_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_108: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_104, primals_650, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_574: "i64[]" = torch.ops.aten.add.Tensor(primals_651, 1)
        var_mean_108 = torch.ops.aten.var_mean.correction(convolution_108, [0, 2, 3], correction = 0, keepdim = True)
        getitem_218: "f32[1, 1024, 1, 1]" = var_mean_108[0]
        getitem_219: "f32[1, 1024, 1, 1]" = var_mean_108[1];  var_mean_108 = None
        add_575: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-05)
        rsqrt_108: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_575);  add_575 = None
        sub_108: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_108, getitem_219)
        mul_756: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_108);  sub_108 = None
        squeeze_324: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_219, [0, 2, 3]);  getitem_219 = None
        squeeze_325: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_108, [0, 2, 3]);  rsqrt_108 = None
        mul_757: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_324, 0.1)
        mul_758: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_652, 0.9)
        add_576: "f32[1024]" = torch.ops.aten.add.Tensor(mul_757, mul_758);  mul_757 = mul_758 = None
        squeeze_326: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_218, [0, 2, 3]);  getitem_218 = None
        mul_759: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_326, 1.0001594642002871);  squeeze_326 = None
        mul_760: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_759, 0.1);  mul_759 = None
        mul_761: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_653, 0.9)
        add_577: "f32[1024]" = torch.ops.aten.add.Tensor(mul_760, mul_761);  mul_760 = mul_761 = None
        unsqueeze_432: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_654, -1)
        unsqueeze_433: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        mul_762: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_756, unsqueeze_433);  mul_756 = unsqueeze_433 = None
        unsqueeze_434: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_655, -1);  primals_655 = None
        unsqueeze_435: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        add_578: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_762, unsqueeze_435);  mul_762 = unsqueeze_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_579: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_578, relu_102);  add_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_105: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_579);  add_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_109: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_105, primals_656, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_580: "i64[]" = torch.ops.aten.add.Tensor(primals_657, 1)
        var_mean_109 = torch.ops.aten.var_mean.correction(convolution_109, [0, 2, 3], correction = 0, keepdim = True)
        getitem_220: "f32[1, 256, 1, 1]" = var_mean_109[0]
        getitem_221: "f32[1, 256, 1, 1]" = var_mean_109[1];  var_mean_109 = None
        add_581: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-05)
        rsqrt_109: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_581);  add_581 = None
        sub_109: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_109, getitem_221)
        mul_763: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_109);  sub_109 = None
        squeeze_327: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_221, [0, 2, 3]);  getitem_221 = None
        squeeze_328: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_109, [0, 2, 3]);  rsqrt_109 = None
        mul_764: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_327, 0.1)
        mul_765: "f32[256]" = torch.ops.aten.mul.Tensor(primals_658, 0.9)
        add_582: "f32[256]" = torch.ops.aten.add.Tensor(mul_764, mul_765);  mul_764 = mul_765 = None
        squeeze_329: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_220, [0, 2, 3]);  getitem_220 = None
        mul_766: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_329, 1.0001594642002871);  squeeze_329 = None
        mul_767: "f32[256]" = torch.ops.aten.mul.Tensor(mul_766, 0.1);  mul_766 = None
        mul_768: "f32[256]" = torch.ops.aten.mul.Tensor(primals_659, 0.9)
        add_583: "f32[256]" = torch.ops.aten.add.Tensor(mul_767, mul_768);  mul_767 = mul_768 = None
        unsqueeze_436: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_660, -1)
        unsqueeze_437: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_769: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_763, unsqueeze_437);  mul_763 = unsqueeze_437 = None
        unsqueeze_438: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_661, -1);  primals_661 = None
        unsqueeze_439: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_584: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_769, unsqueeze_439);  mul_769 = unsqueeze_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_106: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_584);  add_584 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_110: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_106, primals_662, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_585: "i64[]" = torch.ops.aten.add.Tensor(primals_663, 1)
        var_mean_110 = torch.ops.aten.var_mean.correction(convolution_110, [0, 2, 3], correction = 0, keepdim = True)
        getitem_222: "f32[1, 256, 1, 1]" = var_mean_110[0]
        getitem_223: "f32[1, 256, 1, 1]" = var_mean_110[1];  var_mean_110 = None
        add_586: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-05)
        rsqrt_110: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_586);  add_586 = None
        sub_110: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_110, getitem_223)
        mul_770: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_110);  sub_110 = None
        squeeze_330: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_223, [0, 2, 3]);  getitem_223 = None
        squeeze_331: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_110, [0, 2, 3]);  rsqrt_110 = None
        mul_771: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_330, 0.1)
        mul_772: "f32[256]" = torch.ops.aten.mul.Tensor(primals_664, 0.9)
        add_587: "f32[256]" = torch.ops.aten.add.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
        squeeze_332: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_222, [0, 2, 3]);  getitem_222 = None
        mul_773: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_332, 1.0001594642002871);  squeeze_332 = None
        mul_774: "f32[256]" = torch.ops.aten.mul.Tensor(mul_773, 0.1);  mul_773 = None
        mul_775: "f32[256]" = torch.ops.aten.mul.Tensor(primals_665, 0.9)
        add_588: "f32[256]" = torch.ops.aten.add.Tensor(mul_774, mul_775);  mul_774 = mul_775 = None
        unsqueeze_440: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_666, -1)
        unsqueeze_441: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        mul_776: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_770, unsqueeze_441);  mul_770 = unsqueeze_441 = None
        unsqueeze_442: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_667, -1);  primals_667 = None
        unsqueeze_443: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        add_589: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_776, unsqueeze_443);  mul_776 = unsqueeze_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_107: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_589);  add_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_111: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_107, primals_668, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_590: "i64[]" = torch.ops.aten.add.Tensor(primals_669, 1)
        var_mean_111 = torch.ops.aten.var_mean.correction(convolution_111, [0, 2, 3], correction = 0, keepdim = True)
        getitem_224: "f32[1, 1024, 1, 1]" = var_mean_111[0]
        getitem_225: "f32[1, 1024, 1, 1]" = var_mean_111[1];  var_mean_111 = None
        add_591: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_224, 1e-05)
        rsqrt_111: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_591);  add_591 = None
        sub_111: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_111, getitem_225)
        mul_777: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_111);  sub_111 = None
        squeeze_333: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_225, [0, 2, 3]);  getitem_225 = None
        squeeze_334: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_111, [0, 2, 3]);  rsqrt_111 = None
        mul_778: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_333, 0.1)
        mul_779: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_670, 0.9)
        add_592: "f32[1024]" = torch.ops.aten.add.Tensor(mul_778, mul_779);  mul_778 = mul_779 = None
        squeeze_335: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_224, [0, 2, 3]);  getitem_224 = None
        mul_780: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_335, 1.0001594642002871);  squeeze_335 = None
        mul_781: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_780, 0.1);  mul_780 = None
        mul_782: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_671, 0.9)
        add_593: "f32[1024]" = torch.ops.aten.add.Tensor(mul_781, mul_782);  mul_781 = mul_782 = None
        unsqueeze_444: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_672, -1)
        unsqueeze_445: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_783: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_777, unsqueeze_445);  mul_777 = unsqueeze_445 = None
        unsqueeze_446: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_673, -1);  primals_673 = None
        unsqueeze_447: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_594: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_783, unsqueeze_447);  mul_783 = unsqueeze_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_595: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_594, relu_105);  add_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_108: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_595);  add_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_112: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_108, primals_674, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_596: "i64[]" = torch.ops.aten.add.Tensor(primals_675, 1)
        var_mean_112 = torch.ops.aten.var_mean.correction(convolution_112, [0, 2, 3], correction = 0, keepdim = True)
        getitem_226: "f32[1, 256, 1, 1]" = var_mean_112[0]
        getitem_227: "f32[1, 256, 1, 1]" = var_mean_112[1];  var_mean_112 = None
        add_597: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_226, 1e-05)
        rsqrt_112: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_597);  add_597 = None
        sub_112: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_112, getitem_227)
        mul_784: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_112);  sub_112 = None
        squeeze_336: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_227, [0, 2, 3]);  getitem_227 = None
        squeeze_337: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_112, [0, 2, 3]);  rsqrt_112 = None
        mul_785: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_336, 0.1)
        mul_786: "f32[256]" = torch.ops.aten.mul.Tensor(primals_676, 0.9)
        add_598: "f32[256]" = torch.ops.aten.add.Tensor(mul_785, mul_786);  mul_785 = mul_786 = None
        squeeze_338: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_226, [0, 2, 3]);  getitem_226 = None
        mul_787: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_338, 1.0001594642002871);  squeeze_338 = None
        mul_788: "f32[256]" = torch.ops.aten.mul.Tensor(mul_787, 0.1);  mul_787 = None
        mul_789: "f32[256]" = torch.ops.aten.mul.Tensor(primals_677, 0.9)
        add_599: "f32[256]" = torch.ops.aten.add.Tensor(mul_788, mul_789);  mul_788 = mul_789 = None
        unsqueeze_448: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_678, -1)
        unsqueeze_449: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        mul_790: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_784, unsqueeze_449);  mul_784 = unsqueeze_449 = None
        unsqueeze_450: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_679, -1);  primals_679 = None
        unsqueeze_451: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        add_600: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_790, unsqueeze_451);  mul_790 = unsqueeze_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_109: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_600);  add_600 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_113: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_109, primals_680, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_601: "i64[]" = torch.ops.aten.add.Tensor(primals_681, 1)
        var_mean_113 = torch.ops.aten.var_mean.correction(convolution_113, [0, 2, 3], correction = 0, keepdim = True)
        getitem_228: "f32[1, 256, 1, 1]" = var_mean_113[0]
        getitem_229: "f32[1, 256, 1, 1]" = var_mean_113[1];  var_mean_113 = None
        add_602: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_228, 1e-05)
        rsqrt_113: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_602);  add_602 = None
        sub_113: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_113, getitem_229)
        mul_791: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_113);  sub_113 = None
        squeeze_339: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_229, [0, 2, 3]);  getitem_229 = None
        squeeze_340: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_113, [0, 2, 3]);  rsqrt_113 = None
        mul_792: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_339, 0.1)
        mul_793: "f32[256]" = torch.ops.aten.mul.Tensor(primals_682, 0.9)
        add_603: "f32[256]" = torch.ops.aten.add.Tensor(mul_792, mul_793);  mul_792 = mul_793 = None
        squeeze_341: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_228, [0, 2, 3]);  getitem_228 = None
        mul_794: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_341, 1.0001594642002871);  squeeze_341 = None
        mul_795: "f32[256]" = torch.ops.aten.mul.Tensor(mul_794, 0.1);  mul_794 = None
        mul_796: "f32[256]" = torch.ops.aten.mul.Tensor(primals_683, 0.9)
        add_604: "f32[256]" = torch.ops.aten.add.Tensor(mul_795, mul_796);  mul_795 = mul_796 = None
        unsqueeze_452: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_684, -1)
        unsqueeze_453: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_797: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_791, unsqueeze_453);  mul_791 = unsqueeze_453 = None
        unsqueeze_454: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_685, -1);  primals_685 = None
        unsqueeze_455: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_605: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_797, unsqueeze_455);  mul_797 = unsqueeze_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_110: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_605);  add_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_114: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_110, primals_686, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_606: "i64[]" = torch.ops.aten.add.Tensor(primals_687, 1)
        var_mean_114 = torch.ops.aten.var_mean.correction(convolution_114, [0, 2, 3], correction = 0, keepdim = True)
        getitem_230: "f32[1, 1024, 1, 1]" = var_mean_114[0]
        getitem_231: "f32[1, 1024, 1, 1]" = var_mean_114[1];  var_mean_114 = None
        add_607: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_230, 1e-05)
        rsqrt_114: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_607);  add_607 = None
        sub_114: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_114, getitem_231)
        mul_798: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_114);  sub_114 = None
        squeeze_342: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_231, [0, 2, 3]);  getitem_231 = None
        squeeze_343: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_114, [0, 2, 3]);  rsqrt_114 = None
        mul_799: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_342, 0.1)
        mul_800: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_688, 0.9)
        add_608: "f32[1024]" = torch.ops.aten.add.Tensor(mul_799, mul_800);  mul_799 = mul_800 = None
        squeeze_344: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_230, [0, 2, 3]);  getitem_230 = None
        mul_801: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_344, 1.0001594642002871);  squeeze_344 = None
        mul_802: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_801, 0.1);  mul_801 = None
        mul_803: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_689, 0.9)
        add_609: "f32[1024]" = torch.ops.aten.add.Tensor(mul_802, mul_803);  mul_802 = mul_803 = None
        unsqueeze_456: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_690, -1)
        unsqueeze_457: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        mul_804: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_798, unsqueeze_457);  mul_798 = unsqueeze_457 = None
        unsqueeze_458: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_691, -1);  primals_691 = None
        unsqueeze_459: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        add_610: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_804, unsqueeze_459);  mul_804 = unsqueeze_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_611: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_610, relu_108);  add_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_111: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_611);  add_611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_115: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_111, primals_692, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_612: "i64[]" = torch.ops.aten.add.Tensor(primals_693, 1)
        var_mean_115 = torch.ops.aten.var_mean.correction(convolution_115, [0, 2, 3], correction = 0, keepdim = True)
        getitem_232: "f32[1, 256, 1, 1]" = var_mean_115[0]
        getitem_233: "f32[1, 256, 1, 1]" = var_mean_115[1];  var_mean_115 = None
        add_613: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_232, 1e-05)
        rsqrt_115: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_613);  add_613 = None
        sub_115: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_115, getitem_233)
        mul_805: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_115);  sub_115 = None
        squeeze_345: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_233, [0, 2, 3]);  getitem_233 = None
        squeeze_346: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_115, [0, 2, 3]);  rsqrt_115 = None
        mul_806: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_345, 0.1)
        mul_807: "f32[256]" = torch.ops.aten.mul.Tensor(primals_694, 0.9)
        add_614: "f32[256]" = torch.ops.aten.add.Tensor(mul_806, mul_807);  mul_806 = mul_807 = None
        squeeze_347: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_232, [0, 2, 3]);  getitem_232 = None
        mul_808: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_347, 1.0001594642002871);  squeeze_347 = None
        mul_809: "f32[256]" = torch.ops.aten.mul.Tensor(mul_808, 0.1);  mul_808 = None
        mul_810: "f32[256]" = torch.ops.aten.mul.Tensor(primals_695, 0.9)
        add_615: "f32[256]" = torch.ops.aten.add.Tensor(mul_809, mul_810);  mul_809 = mul_810 = None
        unsqueeze_460: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_696, -1)
        unsqueeze_461: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_811: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_805, unsqueeze_461);  mul_805 = unsqueeze_461 = None
        unsqueeze_462: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_697, -1);  primals_697 = None
        unsqueeze_463: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_616: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_811, unsqueeze_463);  mul_811 = unsqueeze_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_112: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_616);  add_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_116: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_112, primals_698, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_617: "i64[]" = torch.ops.aten.add.Tensor(primals_699, 1)
        var_mean_116 = torch.ops.aten.var_mean.correction(convolution_116, [0, 2, 3], correction = 0, keepdim = True)
        getitem_234: "f32[1, 256, 1, 1]" = var_mean_116[0]
        getitem_235: "f32[1, 256, 1, 1]" = var_mean_116[1];  var_mean_116 = None
        add_618: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_234, 1e-05)
        rsqrt_116: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_618);  add_618 = None
        sub_116: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_116, getitem_235)
        mul_812: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_116);  sub_116 = None
        squeeze_348: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_235, [0, 2, 3]);  getitem_235 = None
        squeeze_349: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_116, [0, 2, 3]);  rsqrt_116 = None
        mul_813: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_348, 0.1)
        mul_814: "f32[256]" = torch.ops.aten.mul.Tensor(primals_700, 0.9)
        add_619: "f32[256]" = torch.ops.aten.add.Tensor(mul_813, mul_814);  mul_813 = mul_814 = None
        squeeze_350: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_234, [0, 2, 3]);  getitem_234 = None
        mul_815: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_350, 1.0001594642002871);  squeeze_350 = None
        mul_816: "f32[256]" = torch.ops.aten.mul.Tensor(mul_815, 0.1);  mul_815 = None
        mul_817: "f32[256]" = torch.ops.aten.mul.Tensor(primals_701, 0.9)
        add_620: "f32[256]" = torch.ops.aten.add.Tensor(mul_816, mul_817);  mul_816 = mul_817 = None
        unsqueeze_464: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_702, -1)
        unsqueeze_465: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        mul_818: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_812, unsqueeze_465);  mul_812 = unsqueeze_465 = None
        unsqueeze_466: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_703, -1);  primals_703 = None
        unsqueeze_467: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        add_621: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_818, unsqueeze_467);  mul_818 = unsqueeze_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_113: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_621);  add_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_117: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_113, primals_704, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_622: "i64[]" = torch.ops.aten.add.Tensor(primals_705, 1)
        var_mean_117 = torch.ops.aten.var_mean.correction(convolution_117, [0, 2, 3], correction = 0, keepdim = True)
        getitem_236: "f32[1, 1024, 1, 1]" = var_mean_117[0]
        getitem_237: "f32[1, 1024, 1, 1]" = var_mean_117[1];  var_mean_117 = None
        add_623: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_236, 1e-05)
        rsqrt_117: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_623);  add_623 = None
        sub_117: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_117, getitem_237)
        mul_819: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_117);  sub_117 = None
        squeeze_351: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_237, [0, 2, 3]);  getitem_237 = None
        squeeze_352: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_117, [0, 2, 3]);  rsqrt_117 = None
        mul_820: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_351, 0.1)
        mul_821: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_706, 0.9)
        add_624: "f32[1024]" = torch.ops.aten.add.Tensor(mul_820, mul_821);  mul_820 = mul_821 = None
        squeeze_353: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_236, [0, 2, 3]);  getitem_236 = None
        mul_822: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_353, 1.0001594642002871);  squeeze_353 = None
        mul_823: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_822, 0.1);  mul_822 = None
        mul_824: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_707, 0.9)
        add_625: "f32[1024]" = torch.ops.aten.add.Tensor(mul_823, mul_824);  mul_823 = mul_824 = None
        unsqueeze_468: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_708, -1)
        unsqueeze_469: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_825: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_819, unsqueeze_469);  mul_819 = unsqueeze_469 = None
        unsqueeze_470: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_709, -1);  primals_709 = None
        unsqueeze_471: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_626: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_825, unsqueeze_471);  mul_825 = unsqueeze_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_627: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_626, relu_111);  add_626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_114: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_627);  add_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_118: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_114, primals_710, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_628: "i64[]" = torch.ops.aten.add.Tensor(primals_711, 1)
        var_mean_118 = torch.ops.aten.var_mean.correction(convolution_118, [0, 2, 3], correction = 0, keepdim = True)
        getitem_238: "f32[1, 256, 1, 1]" = var_mean_118[0]
        getitem_239: "f32[1, 256, 1, 1]" = var_mean_118[1];  var_mean_118 = None
        add_629: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_238, 1e-05)
        rsqrt_118: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_629);  add_629 = None
        sub_118: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_118, getitem_239)
        mul_826: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_118);  sub_118 = None
        squeeze_354: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_239, [0, 2, 3]);  getitem_239 = None
        squeeze_355: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_118, [0, 2, 3]);  rsqrt_118 = None
        mul_827: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_354, 0.1)
        mul_828: "f32[256]" = torch.ops.aten.mul.Tensor(primals_712, 0.9)
        add_630: "f32[256]" = torch.ops.aten.add.Tensor(mul_827, mul_828);  mul_827 = mul_828 = None
        squeeze_356: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_238, [0, 2, 3]);  getitem_238 = None
        mul_829: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_356, 1.0001594642002871);  squeeze_356 = None
        mul_830: "f32[256]" = torch.ops.aten.mul.Tensor(mul_829, 0.1);  mul_829 = None
        mul_831: "f32[256]" = torch.ops.aten.mul.Tensor(primals_713, 0.9)
        add_631: "f32[256]" = torch.ops.aten.add.Tensor(mul_830, mul_831);  mul_830 = mul_831 = None
        unsqueeze_472: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_714, -1)
        unsqueeze_473: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        mul_832: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_826, unsqueeze_473);  mul_826 = unsqueeze_473 = None
        unsqueeze_474: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_715, -1);  primals_715 = None
        unsqueeze_475: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        add_632: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_832, unsqueeze_475);  mul_832 = unsqueeze_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_115: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_632);  add_632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_119: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_115, primals_716, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_633: "i64[]" = torch.ops.aten.add.Tensor(primals_717, 1)
        var_mean_119 = torch.ops.aten.var_mean.correction(convolution_119, [0, 2, 3], correction = 0, keepdim = True)
        getitem_240: "f32[1, 256, 1, 1]" = var_mean_119[0]
        getitem_241: "f32[1, 256, 1, 1]" = var_mean_119[1];  var_mean_119 = None
        add_634: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_240, 1e-05)
        rsqrt_119: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_634);  add_634 = None
        sub_119: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_119, getitem_241)
        mul_833: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_119);  sub_119 = None
        squeeze_357: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_241, [0, 2, 3]);  getitem_241 = None
        squeeze_358: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_119, [0, 2, 3]);  rsqrt_119 = None
        mul_834: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_357, 0.1)
        mul_835: "f32[256]" = torch.ops.aten.mul.Tensor(primals_718, 0.9)
        add_635: "f32[256]" = torch.ops.aten.add.Tensor(mul_834, mul_835);  mul_834 = mul_835 = None
        squeeze_359: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_240, [0, 2, 3]);  getitem_240 = None
        mul_836: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_359, 1.0001594642002871);  squeeze_359 = None
        mul_837: "f32[256]" = torch.ops.aten.mul.Tensor(mul_836, 0.1);  mul_836 = None
        mul_838: "f32[256]" = torch.ops.aten.mul.Tensor(primals_719, 0.9)
        add_636: "f32[256]" = torch.ops.aten.add.Tensor(mul_837, mul_838);  mul_837 = mul_838 = None
        unsqueeze_476: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_720, -1)
        unsqueeze_477: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_839: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_833, unsqueeze_477);  mul_833 = unsqueeze_477 = None
        unsqueeze_478: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_721, -1);  primals_721 = None
        unsqueeze_479: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_637: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_839, unsqueeze_479);  mul_839 = unsqueeze_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_116: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_637);  add_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_120: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_116, primals_722, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_638: "i64[]" = torch.ops.aten.add.Tensor(primals_723, 1)
        var_mean_120 = torch.ops.aten.var_mean.correction(convolution_120, [0, 2, 3], correction = 0, keepdim = True)
        getitem_242: "f32[1, 1024, 1, 1]" = var_mean_120[0]
        getitem_243: "f32[1, 1024, 1, 1]" = var_mean_120[1];  var_mean_120 = None
        add_639: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-05)
        rsqrt_120: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_639);  add_639 = None
        sub_120: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_120, getitem_243)
        mul_840: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_120, rsqrt_120);  sub_120 = None
        squeeze_360: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_243, [0, 2, 3]);  getitem_243 = None
        squeeze_361: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_120, [0, 2, 3]);  rsqrt_120 = None
        mul_841: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_360, 0.1)
        mul_842: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_724, 0.9)
        add_640: "f32[1024]" = torch.ops.aten.add.Tensor(mul_841, mul_842);  mul_841 = mul_842 = None
        squeeze_362: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_242, [0, 2, 3]);  getitem_242 = None
        mul_843: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_362, 1.0001594642002871);  squeeze_362 = None
        mul_844: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_843, 0.1);  mul_843 = None
        mul_845: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_725, 0.9)
        add_641: "f32[1024]" = torch.ops.aten.add.Tensor(mul_844, mul_845);  mul_844 = mul_845 = None
        unsqueeze_480: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_726, -1)
        unsqueeze_481: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        mul_846: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_840, unsqueeze_481);  mul_840 = unsqueeze_481 = None
        unsqueeze_482: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_727, -1);  primals_727 = None
        unsqueeze_483: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        add_642: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_846, unsqueeze_483);  mul_846 = unsqueeze_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_643: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_642, relu_114);  add_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_117: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_643);  add_643 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_121: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_117, primals_728, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_644: "i64[]" = torch.ops.aten.add.Tensor(primals_729, 1)
        var_mean_121 = torch.ops.aten.var_mean.correction(convolution_121, [0, 2, 3], correction = 0, keepdim = True)
        getitem_244: "f32[1, 256, 1, 1]" = var_mean_121[0]
        getitem_245: "f32[1, 256, 1, 1]" = var_mean_121[1];  var_mean_121 = None
        add_645: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_244, 1e-05)
        rsqrt_121: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_645);  add_645 = None
        sub_121: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_121, getitem_245)
        mul_847: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_121);  sub_121 = None
        squeeze_363: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_245, [0, 2, 3]);  getitem_245 = None
        squeeze_364: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_121, [0, 2, 3]);  rsqrt_121 = None
        mul_848: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_363, 0.1)
        mul_849: "f32[256]" = torch.ops.aten.mul.Tensor(primals_730, 0.9)
        add_646: "f32[256]" = torch.ops.aten.add.Tensor(mul_848, mul_849);  mul_848 = mul_849 = None
        squeeze_365: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_244, [0, 2, 3]);  getitem_244 = None
        mul_850: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_365, 1.0001594642002871);  squeeze_365 = None
        mul_851: "f32[256]" = torch.ops.aten.mul.Tensor(mul_850, 0.1);  mul_850 = None
        mul_852: "f32[256]" = torch.ops.aten.mul.Tensor(primals_731, 0.9)
        add_647: "f32[256]" = torch.ops.aten.add.Tensor(mul_851, mul_852);  mul_851 = mul_852 = None
        unsqueeze_484: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_732, -1)
        unsqueeze_485: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_853: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_847, unsqueeze_485);  mul_847 = unsqueeze_485 = None
        unsqueeze_486: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_733, -1);  primals_733 = None
        unsqueeze_487: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_648: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_853, unsqueeze_487);  mul_853 = unsqueeze_487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_118: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_648);  add_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_122: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_118, primals_734, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_649: "i64[]" = torch.ops.aten.add.Tensor(primals_735, 1)
        var_mean_122 = torch.ops.aten.var_mean.correction(convolution_122, [0, 2, 3], correction = 0, keepdim = True)
        getitem_246: "f32[1, 256, 1, 1]" = var_mean_122[0]
        getitem_247: "f32[1, 256, 1, 1]" = var_mean_122[1];  var_mean_122 = None
        add_650: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_246, 1e-05)
        rsqrt_122: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_650);  add_650 = None
        sub_122: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_122, getitem_247)
        mul_854: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_122);  sub_122 = None
        squeeze_366: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_247, [0, 2, 3]);  getitem_247 = None
        squeeze_367: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_122, [0, 2, 3]);  rsqrt_122 = None
        mul_855: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_366, 0.1)
        mul_856: "f32[256]" = torch.ops.aten.mul.Tensor(primals_736, 0.9)
        add_651: "f32[256]" = torch.ops.aten.add.Tensor(mul_855, mul_856);  mul_855 = mul_856 = None
        squeeze_368: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_246, [0, 2, 3]);  getitem_246 = None
        mul_857: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_368, 1.0001594642002871);  squeeze_368 = None
        mul_858: "f32[256]" = torch.ops.aten.mul.Tensor(mul_857, 0.1);  mul_857 = None
        mul_859: "f32[256]" = torch.ops.aten.mul.Tensor(primals_737, 0.9)
        add_652: "f32[256]" = torch.ops.aten.add.Tensor(mul_858, mul_859);  mul_858 = mul_859 = None
        unsqueeze_488: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_738, -1)
        unsqueeze_489: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        mul_860: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_854, unsqueeze_489);  mul_854 = unsqueeze_489 = None
        unsqueeze_490: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_739, -1);  primals_739 = None
        unsqueeze_491: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        add_653: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_860, unsqueeze_491);  mul_860 = unsqueeze_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_119: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_653);  add_653 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_123: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_119, primals_740, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_654: "i64[]" = torch.ops.aten.add.Tensor(primals_741, 1)
        var_mean_123 = torch.ops.aten.var_mean.correction(convolution_123, [0, 2, 3], correction = 0, keepdim = True)
        getitem_248: "f32[1, 1024, 1, 1]" = var_mean_123[0]
        getitem_249: "f32[1, 1024, 1, 1]" = var_mean_123[1];  var_mean_123 = None
        add_655: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_248, 1e-05)
        rsqrt_123: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_655);  add_655 = None
        sub_123: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_123, getitem_249)
        mul_861: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_123, rsqrt_123);  sub_123 = None
        squeeze_369: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_249, [0, 2, 3]);  getitem_249 = None
        squeeze_370: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_123, [0, 2, 3]);  rsqrt_123 = None
        mul_862: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_369, 0.1)
        mul_863: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_742, 0.9)
        add_656: "f32[1024]" = torch.ops.aten.add.Tensor(mul_862, mul_863);  mul_862 = mul_863 = None
        squeeze_371: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_248, [0, 2, 3]);  getitem_248 = None
        mul_864: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_371, 1.0001594642002871);  squeeze_371 = None
        mul_865: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_864, 0.1);  mul_864 = None
        mul_866: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_743, 0.9)
        add_657: "f32[1024]" = torch.ops.aten.add.Tensor(mul_865, mul_866);  mul_865 = mul_866 = None
        unsqueeze_492: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_744, -1)
        unsqueeze_493: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_867: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_861, unsqueeze_493);  mul_861 = unsqueeze_493 = None
        unsqueeze_494: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_745, -1);  primals_745 = None
        unsqueeze_495: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_658: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_867, unsqueeze_495);  mul_867 = unsqueeze_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_659: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_658, relu_117);  add_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_120: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_659);  add_659 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_124: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_120, primals_746, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_660: "i64[]" = torch.ops.aten.add.Tensor(primals_747, 1)
        var_mean_124 = torch.ops.aten.var_mean.correction(convolution_124, [0, 2, 3], correction = 0, keepdim = True)
        getitem_250: "f32[1, 256, 1, 1]" = var_mean_124[0]
        getitem_251: "f32[1, 256, 1, 1]" = var_mean_124[1];  var_mean_124 = None
        add_661: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_250, 1e-05)
        rsqrt_124: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_661);  add_661 = None
        sub_124: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_124, getitem_251)
        mul_868: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_124);  sub_124 = None
        squeeze_372: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_251, [0, 2, 3]);  getitem_251 = None
        squeeze_373: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_124, [0, 2, 3]);  rsqrt_124 = None
        mul_869: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_372, 0.1)
        mul_870: "f32[256]" = torch.ops.aten.mul.Tensor(primals_748, 0.9)
        add_662: "f32[256]" = torch.ops.aten.add.Tensor(mul_869, mul_870);  mul_869 = mul_870 = None
        squeeze_374: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_250, [0, 2, 3]);  getitem_250 = None
        mul_871: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_374, 1.0001594642002871);  squeeze_374 = None
        mul_872: "f32[256]" = torch.ops.aten.mul.Tensor(mul_871, 0.1);  mul_871 = None
        mul_873: "f32[256]" = torch.ops.aten.mul.Tensor(primals_749, 0.9)
        add_663: "f32[256]" = torch.ops.aten.add.Tensor(mul_872, mul_873);  mul_872 = mul_873 = None
        unsqueeze_496: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_750, -1)
        unsqueeze_497: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        mul_874: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_868, unsqueeze_497);  mul_868 = unsqueeze_497 = None
        unsqueeze_498: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_751, -1);  primals_751 = None
        unsqueeze_499: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        add_664: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_874, unsqueeze_499);  mul_874 = unsqueeze_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_121: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_664);  add_664 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_125: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_121, primals_752, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_665: "i64[]" = torch.ops.aten.add.Tensor(primals_753, 1)
        var_mean_125 = torch.ops.aten.var_mean.correction(convolution_125, [0, 2, 3], correction = 0, keepdim = True)
        getitem_252: "f32[1, 256, 1, 1]" = var_mean_125[0]
        getitem_253: "f32[1, 256, 1, 1]" = var_mean_125[1];  var_mean_125 = None
        add_666: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_252, 1e-05)
        rsqrt_125: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_666);  add_666 = None
        sub_125: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_125, getitem_253)
        mul_875: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_125);  sub_125 = None
        squeeze_375: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_253, [0, 2, 3]);  getitem_253 = None
        squeeze_376: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_125, [0, 2, 3]);  rsqrt_125 = None
        mul_876: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_375, 0.1)
        mul_877: "f32[256]" = torch.ops.aten.mul.Tensor(primals_754, 0.9)
        add_667: "f32[256]" = torch.ops.aten.add.Tensor(mul_876, mul_877);  mul_876 = mul_877 = None
        squeeze_377: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_252, [0, 2, 3]);  getitem_252 = None
        mul_878: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_377, 1.0001594642002871);  squeeze_377 = None
        mul_879: "f32[256]" = torch.ops.aten.mul.Tensor(mul_878, 0.1);  mul_878 = None
        mul_880: "f32[256]" = torch.ops.aten.mul.Tensor(primals_755, 0.9)
        add_668: "f32[256]" = torch.ops.aten.add.Tensor(mul_879, mul_880);  mul_879 = mul_880 = None
        unsqueeze_500: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_756, -1)
        unsqueeze_501: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_881: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_875, unsqueeze_501);  mul_875 = unsqueeze_501 = None
        unsqueeze_502: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_757, -1);  primals_757 = None
        unsqueeze_503: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_669: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_881, unsqueeze_503);  mul_881 = unsqueeze_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_122: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_669);  add_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_126: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_122, primals_758, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_670: "i64[]" = torch.ops.aten.add.Tensor(primals_759, 1)
        var_mean_126 = torch.ops.aten.var_mean.correction(convolution_126, [0, 2, 3], correction = 0, keepdim = True)
        getitem_254: "f32[1, 1024, 1, 1]" = var_mean_126[0]
        getitem_255: "f32[1, 1024, 1, 1]" = var_mean_126[1];  var_mean_126 = None
        add_671: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_254, 1e-05)
        rsqrt_126: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_671);  add_671 = None
        sub_126: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_126, getitem_255)
        mul_882: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_126, rsqrt_126);  sub_126 = None
        squeeze_378: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_255, [0, 2, 3]);  getitem_255 = None
        squeeze_379: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_126, [0, 2, 3]);  rsqrt_126 = None
        mul_883: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_378, 0.1)
        mul_884: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_760, 0.9)
        add_672: "f32[1024]" = torch.ops.aten.add.Tensor(mul_883, mul_884);  mul_883 = mul_884 = None
        squeeze_380: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_254, [0, 2, 3]);  getitem_254 = None
        mul_885: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_380, 1.0001594642002871);  squeeze_380 = None
        mul_886: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_885, 0.1);  mul_885 = None
        mul_887: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_761, 0.9)
        add_673: "f32[1024]" = torch.ops.aten.add.Tensor(mul_886, mul_887);  mul_886 = mul_887 = None
        unsqueeze_504: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_762, -1)
        unsqueeze_505: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        mul_888: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_882, unsqueeze_505);  mul_882 = unsqueeze_505 = None
        unsqueeze_506: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_763, -1);  primals_763 = None
        unsqueeze_507: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        add_674: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_888, unsqueeze_507);  mul_888 = unsqueeze_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_675: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_674, relu_120);  add_674 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_123: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_675);  add_675 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_127: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_123, primals_764, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_676: "i64[]" = torch.ops.aten.add.Tensor(primals_765, 1)
        var_mean_127 = torch.ops.aten.var_mean.correction(convolution_127, [0, 2, 3], correction = 0, keepdim = True)
        getitem_256: "f32[1, 256, 1, 1]" = var_mean_127[0]
        getitem_257: "f32[1, 256, 1, 1]" = var_mean_127[1];  var_mean_127 = None
        add_677: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_256, 1e-05)
        rsqrt_127: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_677);  add_677 = None
        sub_127: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_127, getitem_257)
        mul_889: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_127);  sub_127 = None
        squeeze_381: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_257, [0, 2, 3]);  getitem_257 = None
        squeeze_382: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_127, [0, 2, 3]);  rsqrt_127 = None
        mul_890: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_381, 0.1)
        mul_891: "f32[256]" = torch.ops.aten.mul.Tensor(primals_766, 0.9)
        add_678: "f32[256]" = torch.ops.aten.add.Tensor(mul_890, mul_891);  mul_890 = mul_891 = None
        squeeze_383: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_256, [0, 2, 3]);  getitem_256 = None
        mul_892: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_383, 1.0001594642002871);  squeeze_383 = None
        mul_893: "f32[256]" = torch.ops.aten.mul.Tensor(mul_892, 0.1);  mul_892 = None
        mul_894: "f32[256]" = torch.ops.aten.mul.Tensor(primals_767, 0.9)
        add_679: "f32[256]" = torch.ops.aten.add.Tensor(mul_893, mul_894);  mul_893 = mul_894 = None
        unsqueeze_508: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_768, -1)
        unsqueeze_509: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_895: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_889, unsqueeze_509);  mul_889 = unsqueeze_509 = None
        unsqueeze_510: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_769, -1);  primals_769 = None
        unsqueeze_511: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_680: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_895, unsqueeze_511);  mul_895 = unsqueeze_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_124: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_680);  add_680 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_128: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_124, primals_770, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_681: "i64[]" = torch.ops.aten.add.Tensor(primals_771, 1)
        var_mean_128 = torch.ops.aten.var_mean.correction(convolution_128, [0, 2, 3], correction = 0, keepdim = True)
        getitem_258: "f32[1, 256, 1, 1]" = var_mean_128[0]
        getitem_259: "f32[1, 256, 1, 1]" = var_mean_128[1];  var_mean_128 = None
        add_682: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_258, 1e-05)
        rsqrt_128: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_682);  add_682 = None
        sub_128: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_128, getitem_259)
        mul_896: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_128, rsqrt_128);  sub_128 = None
        squeeze_384: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_259, [0, 2, 3]);  getitem_259 = None
        squeeze_385: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_128, [0, 2, 3]);  rsqrt_128 = None
        mul_897: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_384, 0.1)
        mul_898: "f32[256]" = torch.ops.aten.mul.Tensor(primals_772, 0.9)
        add_683: "f32[256]" = torch.ops.aten.add.Tensor(mul_897, mul_898);  mul_897 = mul_898 = None
        squeeze_386: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_258, [0, 2, 3]);  getitem_258 = None
        mul_899: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_386, 1.0001594642002871);  squeeze_386 = None
        mul_900: "f32[256]" = torch.ops.aten.mul.Tensor(mul_899, 0.1);  mul_899 = None
        mul_901: "f32[256]" = torch.ops.aten.mul.Tensor(primals_773, 0.9)
        add_684: "f32[256]" = torch.ops.aten.add.Tensor(mul_900, mul_901);  mul_900 = mul_901 = None
        unsqueeze_512: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_774, -1)
        unsqueeze_513: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        mul_902: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_896, unsqueeze_513);  mul_896 = unsqueeze_513 = None
        unsqueeze_514: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_775, -1);  primals_775 = None
        unsqueeze_515: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        add_685: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_902, unsqueeze_515);  mul_902 = unsqueeze_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_125: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_685);  add_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_129: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_125, primals_776, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_686: "i64[]" = torch.ops.aten.add.Tensor(primals_777, 1)
        var_mean_129 = torch.ops.aten.var_mean.correction(convolution_129, [0, 2, 3], correction = 0, keepdim = True)
        getitem_260: "f32[1, 1024, 1, 1]" = var_mean_129[0]
        getitem_261: "f32[1, 1024, 1, 1]" = var_mean_129[1];  var_mean_129 = None
        add_687: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_260, 1e-05)
        rsqrt_129: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_687);  add_687 = None
        sub_129: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_129, getitem_261)
        mul_903: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_129, rsqrt_129);  sub_129 = None
        squeeze_387: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_261, [0, 2, 3]);  getitem_261 = None
        squeeze_388: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_129, [0, 2, 3]);  rsqrt_129 = None
        mul_904: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_387, 0.1)
        mul_905: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_778, 0.9)
        add_688: "f32[1024]" = torch.ops.aten.add.Tensor(mul_904, mul_905);  mul_904 = mul_905 = None
        squeeze_389: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_260, [0, 2, 3]);  getitem_260 = None
        mul_906: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_389, 1.0001594642002871);  squeeze_389 = None
        mul_907: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_906, 0.1);  mul_906 = None
        mul_908: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_779, 0.9)
        add_689: "f32[1024]" = torch.ops.aten.add.Tensor(mul_907, mul_908);  mul_907 = mul_908 = None
        unsqueeze_516: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_780, -1)
        unsqueeze_517: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_909: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_903, unsqueeze_517);  mul_903 = unsqueeze_517 = None
        unsqueeze_518: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_781, -1);  primals_781 = None
        unsqueeze_519: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_690: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_909, unsqueeze_519);  mul_909 = unsqueeze_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_691: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_690, relu_123);  add_690 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_126: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_691);  add_691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_130: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_126, primals_782, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_692: "i64[]" = torch.ops.aten.add.Tensor(primals_783, 1)
        var_mean_130 = torch.ops.aten.var_mean.correction(convolution_130, [0, 2, 3], correction = 0, keepdim = True)
        getitem_262: "f32[1, 256, 1, 1]" = var_mean_130[0]
        getitem_263: "f32[1, 256, 1, 1]" = var_mean_130[1];  var_mean_130 = None
        add_693: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_262, 1e-05)
        rsqrt_130: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_693);  add_693 = None
        sub_130: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_130, getitem_263)
        mul_910: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_130, rsqrt_130);  sub_130 = None
        squeeze_390: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_263, [0, 2, 3]);  getitem_263 = None
        squeeze_391: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_130, [0, 2, 3]);  rsqrt_130 = None
        mul_911: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_390, 0.1)
        mul_912: "f32[256]" = torch.ops.aten.mul.Tensor(primals_784, 0.9)
        add_694: "f32[256]" = torch.ops.aten.add.Tensor(mul_911, mul_912);  mul_911 = mul_912 = None
        squeeze_392: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_262, [0, 2, 3]);  getitem_262 = None
        mul_913: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_392, 1.0001594642002871);  squeeze_392 = None
        mul_914: "f32[256]" = torch.ops.aten.mul.Tensor(mul_913, 0.1);  mul_913 = None
        mul_915: "f32[256]" = torch.ops.aten.mul.Tensor(primals_785, 0.9)
        add_695: "f32[256]" = torch.ops.aten.add.Tensor(mul_914, mul_915);  mul_914 = mul_915 = None
        unsqueeze_520: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_786, -1)
        unsqueeze_521: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        mul_916: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_910, unsqueeze_521);  mul_910 = unsqueeze_521 = None
        unsqueeze_522: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_787, -1);  primals_787 = None
        unsqueeze_523: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        add_696: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_916, unsqueeze_523);  mul_916 = unsqueeze_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_127: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_696);  add_696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_131: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_127, primals_788, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_697: "i64[]" = torch.ops.aten.add.Tensor(primals_789, 1)
        var_mean_131 = torch.ops.aten.var_mean.correction(convolution_131, [0, 2, 3], correction = 0, keepdim = True)
        getitem_264: "f32[1, 256, 1, 1]" = var_mean_131[0]
        getitem_265: "f32[1, 256, 1, 1]" = var_mean_131[1];  var_mean_131 = None
        add_698: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_264, 1e-05)
        rsqrt_131: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_698);  add_698 = None
        sub_131: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_131, getitem_265)
        mul_917: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_131, rsqrt_131);  sub_131 = None
        squeeze_393: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_265, [0, 2, 3]);  getitem_265 = None
        squeeze_394: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_131, [0, 2, 3]);  rsqrt_131 = None
        mul_918: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_393, 0.1)
        mul_919: "f32[256]" = torch.ops.aten.mul.Tensor(primals_790, 0.9)
        add_699: "f32[256]" = torch.ops.aten.add.Tensor(mul_918, mul_919);  mul_918 = mul_919 = None
        squeeze_395: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_264, [0, 2, 3]);  getitem_264 = None
        mul_920: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_395, 1.0001594642002871);  squeeze_395 = None
        mul_921: "f32[256]" = torch.ops.aten.mul.Tensor(mul_920, 0.1);  mul_920 = None
        mul_922: "f32[256]" = torch.ops.aten.mul.Tensor(primals_791, 0.9)
        add_700: "f32[256]" = torch.ops.aten.add.Tensor(mul_921, mul_922);  mul_921 = mul_922 = None
        unsqueeze_524: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_792, -1)
        unsqueeze_525: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_923: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_917, unsqueeze_525);  mul_917 = unsqueeze_525 = None
        unsqueeze_526: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_793, -1);  primals_793 = None
        unsqueeze_527: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_701: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_923, unsqueeze_527);  mul_923 = unsqueeze_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_128: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_701);  add_701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_132: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_128, primals_794, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_702: "i64[]" = torch.ops.aten.add.Tensor(primals_795, 1)
        var_mean_132 = torch.ops.aten.var_mean.correction(convolution_132, [0, 2, 3], correction = 0, keepdim = True)
        getitem_266: "f32[1, 1024, 1, 1]" = var_mean_132[0]
        getitem_267: "f32[1, 1024, 1, 1]" = var_mean_132[1];  var_mean_132 = None
        add_703: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_266, 1e-05)
        rsqrt_132: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_703);  add_703 = None
        sub_132: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_132, getitem_267)
        mul_924: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_132, rsqrt_132);  sub_132 = None
        squeeze_396: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_267, [0, 2, 3]);  getitem_267 = None
        squeeze_397: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_132, [0, 2, 3]);  rsqrt_132 = None
        mul_925: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_396, 0.1)
        mul_926: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_796, 0.9)
        add_704: "f32[1024]" = torch.ops.aten.add.Tensor(mul_925, mul_926);  mul_925 = mul_926 = None
        squeeze_398: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_266, [0, 2, 3]);  getitem_266 = None
        mul_927: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_398, 1.0001594642002871);  squeeze_398 = None
        mul_928: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_927, 0.1);  mul_927 = None
        mul_929: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_797, 0.9)
        add_705: "f32[1024]" = torch.ops.aten.add.Tensor(mul_928, mul_929);  mul_928 = mul_929 = None
        unsqueeze_528: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_798, -1)
        unsqueeze_529: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        mul_930: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_924, unsqueeze_529);  mul_924 = unsqueeze_529 = None
        unsqueeze_530: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_799, -1);  primals_799 = None
        unsqueeze_531: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        add_706: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_930, unsqueeze_531);  mul_930 = unsqueeze_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_707: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_706, relu_126);  add_706 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_129: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_707);  add_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_133: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_129, primals_800, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_708: "i64[]" = torch.ops.aten.add.Tensor(primals_801, 1)
        var_mean_133 = torch.ops.aten.var_mean.correction(convolution_133, [0, 2, 3], correction = 0, keepdim = True)
        getitem_268: "f32[1, 256, 1, 1]" = var_mean_133[0]
        getitem_269: "f32[1, 256, 1, 1]" = var_mean_133[1];  var_mean_133 = None
        add_709: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_268, 1e-05)
        rsqrt_133: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_709);  add_709 = None
        sub_133: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_133, getitem_269)
        mul_931: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_133, rsqrt_133);  sub_133 = None
        squeeze_399: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_269, [0, 2, 3]);  getitem_269 = None
        squeeze_400: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_133, [0, 2, 3]);  rsqrt_133 = None
        mul_932: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_399, 0.1)
        mul_933: "f32[256]" = torch.ops.aten.mul.Tensor(primals_802, 0.9)
        add_710: "f32[256]" = torch.ops.aten.add.Tensor(mul_932, mul_933);  mul_932 = mul_933 = None
        squeeze_401: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_268, [0, 2, 3]);  getitem_268 = None
        mul_934: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_401, 1.0001594642002871);  squeeze_401 = None
        mul_935: "f32[256]" = torch.ops.aten.mul.Tensor(mul_934, 0.1);  mul_934 = None
        mul_936: "f32[256]" = torch.ops.aten.mul.Tensor(primals_803, 0.9)
        add_711: "f32[256]" = torch.ops.aten.add.Tensor(mul_935, mul_936);  mul_935 = mul_936 = None
        unsqueeze_532: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_804, -1)
        unsqueeze_533: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_937: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_931, unsqueeze_533);  mul_931 = unsqueeze_533 = None
        unsqueeze_534: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_805, -1);  primals_805 = None
        unsqueeze_535: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_712: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_937, unsqueeze_535);  mul_937 = unsqueeze_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_130: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_712);  add_712 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_134: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_130, primals_806, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_713: "i64[]" = torch.ops.aten.add.Tensor(primals_807, 1)
        var_mean_134 = torch.ops.aten.var_mean.correction(convolution_134, [0, 2, 3], correction = 0, keepdim = True)
        getitem_270: "f32[1, 256, 1, 1]" = var_mean_134[0]
        getitem_271: "f32[1, 256, 1, 1]" = var_mean_134[1];  var_mean_134 = None
        add_714: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_270, 1e-05)
        rsqrt_134: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_714);  add_714 = None
        sub_134: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_134, getitem_271)
        mul_938: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_134, rsqrt_134);  sub_134 = None
        squeeze_402: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_271, [0, 2, 3]);  getitem_271 = None
        squeeze_403: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_134, [0, 2, 3]);  rsqrt_134 = None
        mul_939: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_402, 0.1)
        mul_940: "f32[256]" = torch.ops.aten.mul.Tensor(primals_808, 0.9)
        add_715: "f32[256]" = torch.ops.aten.add.Tensor(mul_939, mul_940);  mul_939 = mul_940 = None
        squeeze_404: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_270, [0, 2, 3]);  getitem_270 = None
        mul_941: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_404, 1.0001594642002871);  squeeze_404 = None
        mul_942: "f32[256]" = torch.ops.aten.mul.Tensor(mul_941, 0.1);  mul_941 = None
        mul_943: "f32[256]" = torch.ops.aten.mul.Tensor(primals_809, 0.9)
        add_716: "f32[256]" = torch.ops.aten.add.Tensor(mul_942, mul_943);  mul_942 = mul_943 = None
        unsqueeze_536: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_810, -1)
        unsqueeze_537: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        mul_944: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_938, unsqueeze_537);  mul_938 = unsqueeze_537 = None
        unsqueeze_538: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_811, -1);  primals_811 = None
        unsqueeze_539: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        add_717: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_944, unsqueeze_539);  mul_944 = unsqueeze_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_131: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_717);  add_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_135: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_131, primals_812, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_718: "i64[]" = torch.ops.aten.add.Tensor(primals_813, 1)
        var_mean_135 = torch.ops.aten.var_mean.correction(convolution_135, [0, 2, 3], correction = 0, keepdim = True)
        getitem_272: "f32[1, 1024, 1, 1]" = var_mean_135[0]
        getitem_273: "f32[1, 1024, 1, 1]" = var_mean_135[1];  var_mean_135 = None
        add_719: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_272, 1e-05)
        rsqrt_135: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_719);  add_719 = None
        sub_135: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_135, getitem_273)
        mul_945: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_135, rsqrt_135);  sub_135 = None
        squeeze_405: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_273, [0, 2, 3]);  getitem_273 = None
        squeeze_406: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_135, [0, 2, 3]);  rsqrt_135 = None
        mul_946: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_405, 0.1)
        mul_947: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_814, 0.9)
        add_720: "f32[1024]" = torch.ops.aten.add.Tensor(mul_946, mul_947);  mul_946 = mul_947 = None
        squeeze_407: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_272, [0, 2, 3]);  getitem_272 = None
        mul_948: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_407, 1.0001594642002871);  squeeze_407 = None
        mul_949: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_948, 0.1);  mul_948 = None
        mul_950: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_815, 0.9)
        add_721: "f32[1024]" = torch.ops.aten.add.Tensor(mul_949, mul_950);  mul_949 = mul_950 = None
        unsqueeze_540: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_816, -1)
        unsqueeze_541: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_951: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_945, unsqueeze_541);  mul_945 = unsqueeze_541 = None
        unsqueeze_542: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_817, -1);  primals_817 = None
        unsqueeze_543: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_722: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_951, unsqueeze_543);  mul_951 = unsqueeze_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_723: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_722, relu_129);  add_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_132: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_723);  add_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_136: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_132, primals_818, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_724: "i64[]" = torch.ops.aten.add.Tensor(primals_819, 1)
        var_mean_136 = torch.ops.aten.var_mean.correction(convolution_136, [0, 2, 3], correction = 0, keepdim = True)
        getitem_274: "f32[1, 256, 1, 1]" = var_mean_136[0]
        getitem_275: "f32[1, 256, 1, 1]" = var_mean_136[1];  var_mean_136 = None
        add_725: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_274, 1e-05)
        rsqrt_136: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_725);  add_725 = None
        sub_136: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_136, getitem_275)
        mul_952: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_136, rsqrt_136);  sub_136 = None
        squeeze_408: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_275, [0, 2, 3]);  getitem_275 = None
        squeeze_409: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_136, [0, 2, 3]);  rsqrt_136 = None
        mul_953: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_408, 0.1)
        mul_954: "f32[256]" = torch.ops.aten.mul.Tensor(primals_820, 0.9)
        add_726: "f32[256]" = torch.ops.aten.add.Tensor(mul_953, mul_954);  mul_953 = mul_954 = None
        squeeze_410: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_274, [0, 2, 3]);  getitem_274 = None
        mul_955: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_410, 1.0001594642002871);  squeeze_410 = None
        mul_956: "f32[256]" = torch.ops.aten.mul.Tensor(mul_955, 0.1);  mul_955 = None
        mul_957: "f32[256]" = torch.ops.aten.mul.Tensor(primals_821, 0.9)
        add_727: "f32[256]" = torch.ops.aten.add.Tensor(mul_956, mul_957);  mul_956 = mul_957 = None
        unsqueeze_544: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_822, -1)
        unsqueeze_545: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        mul_958: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_952, unsqueeze_545);  mul_952 = unsqueeze_545 = None
        unsqueeze_546: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_823, -1);  primals_823 = None
        unsqueeze_547: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        add_728: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_958, unsqueeze_547);  mul_958 = unsqueeze_547 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_133: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_728);  add_728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_137: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_133, primals_824, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_729: "i64[]" = torch.ops.aten.add.Tensor(primals_825, 1)
        var_mean_137 = torch.ops.aten.var_mean.correction(convolution_137, [0, 2, 3], correction = 0, keepdim = True)
        getitem_276: "f32[1, 256, 1, 1]" = var_mean_137[0]
        getitem_277: "f32[1, 256, 1, 1]" = var_mean_137[1];  var_mean_137 = None
        add_730: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_276, 1e-05)
        rsqrt_137: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_730);  add_730 = None
        sub_137: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_137, getitem_277)
        mul_959: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_137);  sub_137 = None
        squeeze_411: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_277, [0, 2, 3]);  getitem_277 = None
        squeeze_412: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_137, [0, 2, 3]);  rsqrt_137 = None
        mul_960: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_411, 0.1)
        mul_961: "f32[256]" = torch.ops.aten.mul.Tensor(primals_826, 0.9)
        add_731: "f32[256]" = torch.ops.aten.add.Tensor(mul_960, mul_961);  mul_960 = mul_961 = None
        squeeze_413: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_276, [0, 2, 3]);  getitem_276 = None
        mul_962: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_413, 1.0001594642002871);  squeeze_413 = None
        mul_963: "f32[256]" = torch.ops.aten.mul.Tensor(mul_962, 0.1);  mul_962 = None
        mul_964: "f32[256]" = torch.ops.aten.mul.Tensor(primals_827, 0.9)
        add_732: "f32[256]" = torch.ops.aten.add.Tensor(mul_963, mul_964);  mul_963 = mul_964 = None
        unsqueeze_548: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_828, -1)
        unsqueeze_549: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_965: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_959, unsqueeze_549);  mul_959 = unsqueeze_549 = None
        unsqueeze_550: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_829, -1);  primals_829 = None
        unsqueeze_551: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_733: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_965, unsqueeze_551);  mul_965 = unsqueeze_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_134: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_733);  add_733 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_138: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_134, primals_830, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_734: "i64[]" = torch.ops.aten.add.Tensor(primals_831, 1)
        var_mean_138 = torch.ops.aten.var_mean.correction(convolution_138, [0, 2, 3], correction = 0, keepdim = True)
        getitem_278: "f32[1, 1024, 1, 1]" = var_mean_138[0]
        getitem_279: "f32[1, 1024, 1, 1]" = var_mean_138[1];  var_mean_138 = None
        add_735: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_278, 1e-05)
        rsqrt_138: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_735);  add_735 = None
        sub_138: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_138, getitem_279)
        mul_966: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_138, rsqrt_138);  sub_138 = None
        squeeze_414: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_279, [0, 2, 3]);  getitem_279 = None
        squeeze_415: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_138, [0, 2, 3]);  rsqrt_138 = None
        mul_967: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_414, 0.1)
        mul_968: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_832, 0.9)
        add_736: "f32[1024]" = torch.ops.aten.add.Tensor(mul_967, mul_968);  mul_967 = mul_968 = None
        squeeze_416: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_278, [0, 2, 3]);  getitem_278 = None
        mul_969: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_416, 1.0001594642002871);  squeeze_416 = None
        mul_970: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_969, 0.1);  mul_969 = None
        mul_971: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_833, 0.9)
        add_737: "f32[1024]" = torch.ops.aten.add.Tensor(mul_970, mul_971);  mul_970 = mul_971 = None
        unsqueeze_552: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_834, -1)
        unsqueeze_553: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        mul_972: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_966, unsqueeze_553);  mul_966 = unsqueeze_553 = None
        unsqueeze_554: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_835, -1);  primals_835 = None
        unsqueeze_555: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        add_738: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_972, unsqueeze_555);  mul_972 = unsqueeze_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_739: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_738, relu_132);  add_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_135: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_739);  add_739 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_139: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_135, primals_836, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_740: "i64[]" = torch.ops.aten.add.Tensor(primals_837, 1)
        var_mean_139 = torch.ops.aten.var_mean.correction(convolution_139, [0, 2, 3], correction = 0, keepdim = True)
        getitem_280: "f32[1, 256, 1, 1]" = var_mean_139[0]
        getitem_281: "f32[1, 256, 1, 1]" = var_mean_139[1];  var_mean_139 = None
        add_741: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_280, 1e-05)
        rsqrt_139: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_741);  add_741 = None
        sub_139: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_139, getitem_281)
        mul_973: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_139, rsqrt_139);  sub_139 = None
        squeeze_417: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_281, [0, 2, 3]);  getitem_281 = None
        squeeze_418: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_139, [0, 2, 3]);  rsqrt_139 = None
        mul_974: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_417, 0.1)
        mul_975: "f32[256]" = torch.ops.aten.mul.Tensor(primals_838, 0.9)
        add_742: "f32[256]" = torch.ops.aten.add.Tensor(mul_974, mul_975);  mul_974 = mul_975 = None
        squeeze_419: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_280, [0, 2, 3]);  getitem_280 = None
        mul_976: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_419, 1.0001594642002871);  squeeze_419 = None
        mul_977: "f32[256]" = torch.ops.aten.mul.Tensor(mul_976, 0.1);  mul_976 = None
        mul_978: "f32[256]" = torch.ops.aten.mul.Tensor(primals_839, 0.9)
        add_743: "f32[256]" = torch.ops.aten.add.Tensor(mul_977, mul_978);  mul_977 = mul_978 = None
        unsqueeze_556: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_840, -1)
        unsqueeze_557: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_979: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_973, unsqueeze_557);  mul_973 = unsqueeze_557 = None
        unsqueeze_558: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_841, -1);  primals_841 = None
        unsqueeze_559: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_744: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_979, unsqueeze_559);  mul_979 = unsqueeze_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_136: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_744);  add_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_140: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_136, primals_842, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_745: "i64[]" = torch.ops.aten.add.Tensor(primals_843, 1)
        var_mean_140 = torch.ops.aten.var_mean.correction(convolution_140, [0, 2, 3], correction = 0, keepdim = True)
        getitem_282: "f32[1, 256, 1, 1]" = var_mean_140[0]
        getitem_283: "f32[1, 256, 1, 1]" = var_mean_140[1];  var_mean_140 = None
        add_746: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_282, 1e-05)
        rsqrt_140: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_746);  add_746 = None
        sub_140: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_140, getitem_283)
        mul_980: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_140, rsqrt_140);  sub_140 = None
        squeeze_420: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_283, [0, 2, 3]);  getitem_283 = None
        squeeze_421: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_140, [0, 2, 3]);  rsqrt_140 = None
        mul_981: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_420, 0.1)
        mul_982: "f32[256]" = torch.ops.aten.mul.Tensor(primals_844, 0.9)
        add_747: "f32[256]" = torch.ops.aten.add.Tensor(mul_981, mul_982);  mul_981 = mul_982 = None
        squeeze_422: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_282, [0, 2, 3]);  getitem_282 = None
        mul_983: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_422, 1.0001594642002871);  squeeze_422 = None
        mul_984: "f32[256]" = torch.ops.aten.mul.Tensor(mul_983, 0.1);  mul_983 = None
        mul_985: "f32[256]" = torch.ops.aten.mul.Tensor(primals_845, 0.9)
        add_748: "f32[256]" = torch.ops.aten.add.Tensor(mul_984, mul_985);  mul_984 = mul_985 = None
        unsqueeze_560: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_846, -1)
        unsqueeze_561: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        mul_986: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_980, unsqueeze_561);  mul_980 = unsqueeze_561 = None
        unsqueeze_562: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_847, -1);  primals_847 = None
        unsqueeze_563: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        add_749: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_986, unsqueeze_563);  mul_986 = unsqueeze_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_137: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_749);  add_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_141: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_137, primals_848, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_750: "i64[]" = torch.ops.aten.add.Tensor(primals_849, 1)
        var_mean_141 = torch.ops.aten.var_mean.correction(convolution_141, [0, 2, 3], correction = 0, keepdim = True)
        getitem_284: "f32[1, 1024, 1, 1]" = var_mean_141[0]
        getitem_285: "f32[1, 1024, 1, 1]" = var_mean_141[1];  var_mean_141 = None
        add_751: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_284, 1e-05)
        rsqrt_141: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_751);  add_751 = None
        sub_141: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_141, getitem_285)
        mul_987: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_141, rsqrt_141);  sub_141 = None
        squeeze_423: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_285, [0, 2, 3]);  getitem_285 = None
        squeeze_424: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_141, [0, 2, 3]);  rsqrt_141 = None
        mul_988: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_423, 0.1)
        mul_989: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_850, 0.9)
        add_752: "f32[1024]" = torch.ops.aten.add.Tensor(mul_988, mul_989);  mul_988 = mul_989 = None
        squeeze_425: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_284, [0, 2, 3]);  getitem_284 = None
        mul_990: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_425, 1.0001594642002871);  squeeze_425 = None
        mul_991: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_990, 0.1);  mul_990 = None
        mul_992: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_851, 0.9)
        add_753: "f32[1024]" = torch.ops.aten.add.Tensor(mul_991, mul_992);  mul_991 = mul_992 = None
        unsqueeze_564: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_852, -1)
        unsqueeze_565: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_993: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_987, unsqueeze_565);  mul_987 = unsqueeze_565 = None
        unsqueeze_566: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_853, -1);  primals_853 = None
        unsqueeze_567: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_754: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_993, unsqueeze_567);  mul_993 = unsqueeze_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_755: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_754, relu_135);  add_754 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_138: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_755);  add_755 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_142: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_138, primals_854, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_756: "i64[]" = torch.ops.aten.add.Tensor(primals_855, 1)
        var_mean_142 = torch.ops.aten.var_mean.correction(convolution_142, [0, 2, 3], correction = 0, keepdim = True)
        getitem_286: "f32[1, 256, 1, 1]" = var_mean_142[0]
        getitem_287: "f32[1, 256, 1, 1]" = var_mean_142[1];  var_mean_142 = None
        add_757: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_286, 1e-05)
        rsqrt_142: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_757);  add_757 = None
        sub_142: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_142, getitem_287)
        mul_994: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_142, rsqrt_142);  sub_142 = None
        squeeze_426: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_287, [0, 2, 3]);  getitem_287 = None
        squeeze_427: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_142, [0, 2, 3]);  rsqrt_142 = None
        mul_995: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_426, 0.1)
        mul_996: "f32[256]" = torch.ops.aten.mul.Tensor(primals_856, 0.9)
        add_758: "f32[256]" = torch.ops.aten.add.Tensor(mul_995, mul_996);  mul_995 = mul_996 = None
        squeeze_428: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_286, [0, 2, 3]);  getitem_286 = None
        mul_997: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_428, 1.0001594642002871);  squeeze_428 = None
        mul_998: "f32[256]" = torch.ops.aten.mul.Tensor(mul_997, 0.1);  mul_997 = None
        mul_999: "f32[256]" = torch.ops.aten.mul.Tensor(primals_857, 0.9)
        add_759: "f32[256]" = torch.ops.aten.add.Tensor(mul_998, mul_999);  mul_998 = mul_999 = None
        unsqueeze_568: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_858, -1)
        unsqueeze_569: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        mul_1000: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_994, unsqueeze_569);  mul_994 = unsqueeze_569 = None
        unsqueeze_570: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_859, -1);  primals_859 = None
        unsqueeze_571: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        add_760: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_1000, unsqueeze_571);  mul_1000 = unsqueeze_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_139: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_760);  add_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_143: "f32[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_139, primals_860, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_761: "i64[]" = torch.ops.aten.add.Tensor(primals_861, 1)
        var_mean_143 = torch.ops.aten.var_mean.correction(convolution_143, [0, 2, 3], correction = 0, keepdim = True)
        getitem_288: "f32[1, 256, 1, 1]" = var_mean_143[0]
        getitem_289: "f32[1, 256, 1, 1]" = var_mean_143[1];  var_mean_143 = None
        add_762: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_288, 1e-05)
        rsqrt_143: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_762);  add_762 = None
        sub_143: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_143, getitem_289)
        mul_1001: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_143, rsqrt_143);  sub_143 = None
        squeeze_429: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_289, [0, 2, 3]);  getitem_289 = None
        squeeze_430: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_143, [0, 2, 3]);  rsqrt_143 = None
        mul_1002: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_429, 0.1)
        mul_1003: "f32[256]" = torch.ops.aten.mul.Tensor(primals_862, 0.9)
        add_763: "f32[256]" = torch.ops.aten.add.Tensor(mul_1002, mul_1003);  mul_1002 = mul_1003 = None
        squeeze_431: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_288, [0, 2, 3]);  getitem_288 = None
        mul_1004: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_431, 1.0001594642002871);  squeeze_431 = None
        mul_1005: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1004, 0.1);  mul_1004 = None
        mul_1006: "f32[256]" = torch.ops.aten.mul.Tensor(primals_863, 0.9)
        add_764: "f32[256]" = torch.ops.aten.add.Tensor(mul_1005, mul_1006);  mul_1005 = mul_1006 = None
        unsqueeze_572: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_864, -1)
        unsqueeze_573: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_1007: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1001, unsqueeze_573);  mul_1001 = unsqueeze_573 = None
        unsqueeze_574: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_865, -1);  primals_865 = None
        unsqueeze_575: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_765: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_1007, unsqueeze_575);  mul_1007 = unsqueeze_575 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_140: "f32[32, 256, 14, 14]" = torch.ops.aten.relu.default(add_765);  add_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_144: "f32[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_140, primals_866, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_766: "i64[]" = torch.ops.aten.add.Tensor(primals_867, 1)
        var_mean_144 = torch.ops.aten.var_mean.correction(convolution_144, [0, 2, 3], correction = 0, keepdim = True)
        getitem_290: "f32[1, 1024, 1, 1]" = var_mean_144[0]
        getitem_291: "f32[1, 1024, 1, 1]" = var_mean_144[1];  var_mean_144 = None
        add_767: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_290, 1e-05)
        rsqrt_144: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_767);  add_767 = None
        sub_144: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_144, getitem_291)
        mul_1008: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, rsqrt_144);  sub_144 = None
        squeeze_432: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_291, [0, 2, 3]);  getitem_291 = None
        squeeze_433: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_144, [0, 2, 3]);  rsqrt_144 = None
        mul_1009: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_432, 0.1)
        mul_1010: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_868, 0.9)
        add_768: "f32[1024]" = torch.ops.aten.add.Tensor(mul_1009, mul_1010);  mul_1009 = mul_1010 = None
        squeeze_434: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_290, [0, 2, 3]);  getitem_290 = None
        mul_1011: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_434, 1.0001594642002871);  squeeze_434 = None
        mul_1012: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1011, 0.1);  mul_1011 = None
        mul_1013: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_869, 0.9)
        add_769: "f32[1024]" = torch.ops.aten.add.Tensor(mul_1012, mul_1013);  mul_1012 = mul_1013 = None
        unsqueeze_576: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_870, -1)
        unsqueeze_577: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        mul_1014: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1008, unsqueeze_577);  mul_1008 = unsqueeze_577 = None
        unsqueeze_578: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_871, -1);  primals_871 = None
        unsqueeze_579: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        add_770: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_1014, unsqueeze_579);  mul_1014 = unsqueeze_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_771: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_770, relu_138);  add_770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_141: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_771);  add_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_145: "f32[32, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_141, primals_872, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_772: "i64[]" = torch.ops.aten.add.Tensor(primals_873, 1)
        var_mean_145 = torch.ops.aten.var_mean.correction(convolution_145, [0, 2, 3], correction = 0, keepdim = True)
        getitem_292: "f32[1, 512, 1, 1]" = var_mean_145[0]
        getitem_293: "f32[1, 512, 1, 1]" = var_mean_145[1];  var_mean_145 = None
        add_773: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_292, 1e-05)
        rsqrt_145: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_773);  add_773 = None
        sub_145: "f32[32, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_145, getitem_293)
        mul_1015: "f32[32, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_145, rsqrt_145);  sub_145 = None
        squeeze_435: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_293, [0, 2, 3]);  getitem_293 = None
        squeeze_436: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_145, [0, 2, 3]);  rsqrt_145 = None
        mul_1016: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_435, 0.1)
        mul_1017: "f32[512]" = torch.ops.aten.mul.Tensor(primals_874, 0.9)
        add_774: "f32[512]" = torch.ops.aten.add.Tensor(mul_1016, mul_1017);  mul_1016 = mul_1017 = None
        squeeze_437: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_292, [0, 2, 3]);  getitem_292 = None
        mul_1018: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_437, 1.0001594642002871);  squeeze_437 = None
        mul_1019: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1018, 0.1);  mul_1018 = None
        mul_1020: "f32[512]" = torch.ops.aten.mul.Tensor(primals_875, 0.9)
        add_775: "f32[512]" = torch.ops.aten.add.Tensor(mul_1019, mul_1020);  mul_1019 = mul_1020 = None
        unsqueeze_580: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_876, -1)
        unsqueeze_581: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_1021: "f32[32, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1015, unsqueeze_581);  mul_1015 = unsqueeze_581 = None
        unsqueeze_582: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_877, -1);  primals_877 = None
        unsqueeze_583: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_776: "f32[32, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_1021, unsqueeze_583);  mul_1021 = unsqueeze_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_142: "f32[32, 512, 14, 14]" = torch.ops.aten.relu.default(add_776);  add_776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_146: "f32[32, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_142, primals_878, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_777: "i64[]" = torch.ops.aten.add.Tensor(primals_879, 1)
        var_mean_146 = torch.ops.aten.var_mean.correction(convolution_146, [0, 2, 3], correction = 0, keepdim = True)
        getitem_294: "f32[1, 512, 1, 1]" = var_mean_146[0]
        getitem_295: "f32[1, 512, 1, 1]" = var_mean_146[1];  var_mean_146 = None
        add_778: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_294, 1e-05)
        rsqrt_146: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_778);  add_778 = None
        sub_146: "f32[32, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_146, getitem_295)
        mul_1022: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_146, rsqrt_146);  sub_146 = None
        squeeze_438: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_295, [0, 2, 3]);  getitem_295 = None
        squeeze_439: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_146, [0, 2, 3]);  rsqrt_146 = None
        mul_1023: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_438, 0.1)
        mul_1024: "f32[512]" = torch.ops.aten.mul.Tensor(primals_880, 0.9)
        add_779: "f32[512]" = torch.ops.aten.add.Tensor(mul_1023, mul_1024);  mul_1023 = mul_1024 = None
        squeeze_440: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_294, [0, 2, 3]);  getitem_294 = None
        mul_1025: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_440, 1.0006381620931717);  squeeze_440 = None
        mul_1026: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1025, 0.1);  mul_1025 = None
        mul_1027: "f32[512]" = torch.ops.aten.mul.Tensor(primals_881, 0.9)
        add_780: "f32[512]" = torch.ops.aten.add.Tensor(mul_1026, mul_1027);  mul_1026 = mul_1027 = None
        unsqueeze_584: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_882, -1)
        unsqueeze_585: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        mul_1028: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1022, unsqueeze_585);  mul_1022 = unsqueeze_585 = None
        unsqueeze_586: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_883, -1);  primals_883 = None
        unsqueeze_587: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        add_781: "f32[32, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_1028, unsqueeze_587);  mul_1028 = unsqueeze_587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_143: "f32[32, 512, 7, 7]" = torch.ops.aten.relu.default(add_781);  add_781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_147: "f32[32, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_143, primals_884, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_782: "i64[]" = torch.ops.aten.add.Tensor(primals_885, 1)
        var_mean_147 = torch.ops.aten.var_mean.correction(convolution_147, [0, 2, 3], correction = 0, keepdim = True)
        getitem_296: "f32[1, 2048, 1, 1]" = var_mean_147[0]
        getitem_297: "f32[1, 2048, 1, 1]" = var_mean_147[1];  var_mean_147 = None
        add_783: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_296, 1e-05)
        rsqrt_147: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_783);  add_783 = None
        sub_147: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_147, getitem_297)
        mul_1029: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_147, rsqrt_147);  sub_147 = None
        squeeze_441: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_297, [0, 2, 3]);  getitem_297 = None
        squeeze_442: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_147, [0, 2, 3]);  rsqrt_147 = None
        mul_1030: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_441, 0.1)
        mul_1031: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_886, 0.9)
        add_784: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1030, mul_1031);  mul_1030 = mul_1031 = None
        squeeze_443: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_296, [0, 2, 3]);  getitem_296 = None
        mul_1032: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_443, 1.0006381620931717);  squeeze_443 = None
        mul_1033: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1032, 0.1);  mul_1032 = None
        mul_1034: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_887, 0.9)
        add_785: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1033, mul_1034);  mul_1033 = mul_1034 = None
        unsqueeze_588: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_888, -1)
        unsqueeze_589: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_1035: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1029, unsqueeze_589);  mul_1029 = unsqueeze_589 = None
        unsqueeze_590: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_889, -1);  primals_889 = None
        unsqueeze_591: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_786: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_1035, unsqueeze_591);  mul_1035 = unsqueeze_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convolution_148: "f32[32, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_141, primals_890, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add_787: "i64[]" = torch.ops.aten.add.Tensor(primals_891, 1)
        var_mean_148 = torch.ops.aten.var_mean.correction(convolution_148, [0, 2, 3], correction = 0, keepdim = True)
        getitem_298: "f32[1, 2048, 1, 1]" = var_mean_148[0]
        getitem_299: "f32[1, 2048, 1, 1]" = var_mean_148[1];  var_mean_148 = None
        add_788: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_298, 1e-05)
        rsqrt_148: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_788);  add_788 = None
        sub_148: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_148, getitem_299)
        mul_1036: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_148, rsqrt_148);  sub_148 = None
        squeeze_444: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_299, [0, 2, 3]);  getitem_299 = None
        squeeze_445: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_148, [0, 2, 3]);  rsqrt_148 = None
        mul_1037: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_444, 0.1)
        mul_1038: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_892, 0.9)
        add_789: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1037, mul_1038);  mul_1037 = mul_1038 = None
        squeeze_446: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_298, [0, 2, 3]);  getitem_298 = None
        mul_1039: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_446, 1.0006381620931717);  squeeze_446 = None
        mul_1040: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1039, 0.1);  mul_1039 = None
        mul_1041: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_893, 0.9)
        add_790: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1040, mul_1041);  mul_1040 = mul_1041 = None
        unsqueeze_592: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_894, -1)
        unsqueeze_593: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        mul_1042: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1036, unsqueeze_593);  mul_1036 = unsqueeze_593 = None
        unsqueeze_594: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_895, -1);  primals_895 = None
        unsqueeze_595: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        add_791: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_1042, unsqueeze_595);  mul_1042 = unsqueeze_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_792: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_786, add_791);  add_786 = add_791 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_144: "f32[32, 2048, 7, 7]" = torch.ops.aten.relu.default(add_792);  add_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_149: "f32[32, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_144, primals_896, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_793: "i64[]" = torch.ops.aten.add.Tensor(primals_897, 1)
        var_mean_149 = torch.ops.aten.var_mean.correction(convolution_149, [0, 2, 3], correction = 0, keepdim = True)
        getitem_300: "f32[1, 512, 1, 1]" = var_mean_149[0]
        getitem_301: "f32[1, 512, 1, 1]" = var_mean_149[1];  var_mean_149 = None
        add_794: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_300, 1e-05)
        rsqrt_149: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_794);  add_794 = None
        sub_149: "f32[32, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_149, getitem_301)
        mul_1043: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_149, rsqrt_149);  sub_149 = None
        squeeze_447: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_301, [0, 2, 3]);  getitem_301 = None
        squeeze_448: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_149, [0, 2, 3]);  rsqrt_149 = None
        mul_1044: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_447, 0.1)
        mul_1045: "f32[512]" = torch.ops.aten.mul.Tensor(primals_898, 0.9)
        add_795: "f32[512]" = torch.ops.aten.add.Tensor(mul_1044, mul_1045);  mul_1044 = mul_1045 = None
        squeeze_449: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_300, [0, 2, 3]);  getitem_300 = None
        mul_1046: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_449, 1.0006381620931717);  squeeze_449 = None
        mul_1047: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1046, 0.1);  mul_1046 = None
        mul_1048: "f32[512]" = torch.ops.aten.mul.Tensor(primals_899, 0.9)
        add_796: "f32[512]" = torch.ops.aten.add.Tensor(mul_1047, mul_1048);  mul_1047 = mul_1048 = None
        unsqueeze_596: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_900, -1)
        unsqueeze_597: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_1049: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1043, unsqueeze_597);  mul_1043 = unsqueeze_597 = None
        unsqueeze_598: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_901, -1);  primals_901 = None
        unsqueeze_599: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_797: "f32[32, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_1049, unsqueeze_599);  mul_1049 = unsqueeze_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_145: "f32[32, 512, 7, 7]" = torch.ops.aten.relu.default(add_797);  add_797 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_150: "f32[32, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_145, primals_902, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_798: "i64[]" = torch.ops.aten.add.Tensor(primals_903, 1)
        var_mean_150 = torch.ops.aten.var_mean.correction(convolution_150, [0, 2, 3], correction = 0, keepdim = True)
        getitem_302: "f32[1, 512, 1, 1]" = var_mean_150[0]
        getitem_303: "f32[1, 512, 1, 1]" = var_mean_150[1];  var_mean_150 = None
        add_799: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_302, 1e-05)
        rsqrt_150: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_799);  add_799 = None
        sub_150: "f32[32, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_150, getitem_303)
        mul_1050: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_150, rsqrt_150);  sub_150 = None
        squeeze_450: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_303, [0, 2, 3]);  getitem_303 = None
        squeeze_451: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_150, [0, 2, 3]);  rsqrt_150 = None
        mul_1051: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_450, 0.1)
        mul_1052: "f32[512]" = torch.ops.aten.mul.Tensor(primals_904, 0.9)
        add_800: "f32[512]" = torch.ops.aten.add.Tensor(mul_1051, mul_1052);  mul_1051 = mul_1052 = None
        squeeze_452: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_302, [0, 2, 3]);  getitem_302 = None
        mul_1053: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_452, 1.0006381620931717);  squeeze_452 = None
        mul_1054: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1053, 0.1);  mul_1053 = None
        mul_1055: "f32[512]" = torch.ops.aten.mul.Tensor(primals_905, 0.9)
        add_801: "f32[512]" = torch.ops.aten.add.Tensor(mul_1054, mul_1055);  mul_1054 = mul_1055 = None
        unsqueeze_600: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_906, -1)
        unsqueeze_601: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        mul_1056: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1050, unsqueeze_601);  mul_1050 = unsqueeze_601 = None
        unsqueeze_602: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_907, -1);  primals_907 = None
        unsqueeze_603: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        add_802: "f32[32, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_1056, unsqueeze_603);  mul_1056 = unsqueeze_603 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_146: "f32[32, 512, 7, 7]" = torch.ops.aten.relu.default(add_802);  add_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_151: "f32[32, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_146, primals_908, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_803: "i64[]" = torch.ops.aten.add.Tensor(primals_909, 1)
        var_mean_151 = torch.ops.aten.var_mean.correction(convolution_151, [0, 2, 3], correction = 0, keepdim = True)
        getitem_304: "f32[1, 2048, 1, 1]" = var_mean_151[0]
        getitem_305: "f32[1, 2048, 1, 1]" = var_mean_151[1];  var_mean_151 = None
        add_804: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_304, 1e-05)
        rsqrt_151: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_804);  add_804 = None
        sub_151: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_151, getitem_305)
        mul_1057: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_151, rsqrt_151);  sub_151 = None
        squeeze_453: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_305, [0, 2, 3]);  getitem_305 = None
        squeeze_454: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_151, [0, 2, 3]);  rsqrt_151 = None
        mul_1058: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_453, 0.1)
        mul_1059: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_910, 0.9)
        add_805: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1058, mul_1059);  mul_1058 = mul_1059 = None
        squeeze_455: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_304, [0, 2, 3]);  getitem_304 = None
        mul_1060: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_455, 1.0006381620931717);  squeeze_455 = None
        mul_1061: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1060, 0.1);  mul_1060 = None
        mul_1062: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_911, 0.9)
        add_806: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1061, mul_1062);  mul_1061 = mul_1062 = None
        unsqueeze_604: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_912, -1)
        unsqueeze_605: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_1063: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1057, unsqueeze_605);  mul_1057 = unsqueeze_605 = None
        unsqueeze_606: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_913, -1);  primals_913 = None
        unsqueeze_607: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_807: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_1063, unsqueeze_607);  mul_1063 = unsqueeze_607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_808: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_807, relu_144);  add_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_147: "f32[32, 2048, 7, 7]" = torch.ops.aten.relu.default(add_808);  add_808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_152: "f32[32, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_147, primals_914, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        add_809: "i64[]" = torch.ops.aten.add.Tensor(primals_915, 1)
        var_mean_152 = torch.ops.aten.var_mean.correction(convolution_152, [0, 2, 3], correction = 0, keepdim = True)
        getitem_306: "f32[1, 512, 1, 1]" = var_mean_152[0]
        getitem_307: "f32[1, 512, 1, 1]" = var_mean_152[1];  var_mean_152 = None
        add_810: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_306, 1e-05)
        rsqrt_152: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_810);  add_810 = None
        sub_152: "f32[32, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_152, getitem_307)
        mul_1064: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_152, rsqrt_152);  sub_152 = None
        squeeze_456: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_307, [0, 2, 3]);  getitem_307 = None
        squeeze_457: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_152, [0, 2, 3]);  rsqrt_152 = None
        mul_1065: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_456, 0.1)
        mul_1066: "f32[512]" = torch.ops.aten.mul.Tensor(primals_916, 0.9)
        add_811: "f32[512]" = torch.ops.aten.add.Tensor(mul_1065, mul_1066);  mul_1065 = mul_1066 = None
        squeeze_458: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_306, [0, 2, 3]);  getitem_306 = None
        mul_1067: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_458, 1.0006381620931717);  squeeze_458 = None
        mul_1068: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1067, 0.1);  mul_1067 = None
        mul_1069: "f32[512]" = torch.ops.aten.mul.Tensor(primals_917, 0.9)
        add_812: "f32[512]" = torch.ops.aten.add.Tensor(mul_1068, mul_1069);  mul_1068 = mul_1069 = None
        unsqueeze_608: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_918, -1)
        unsqueeze_609: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        mul_1070: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1064, unsqueeze_609);  mul_1064 = unsqueeze_609 = None
        unsqueeze_610: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_919, -1);  primals_919 = None
        unsqueeze_611: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        add_813: "f32[32, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_1070, unsqueeze_611);  mul_1070 = unsqueeze_611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_148: "f32[32, 512, 7, 7]" = torch.ops.aten.relu.default(add_813);  add_813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_153: "f32[32, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_148, primals_920, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        add_814: "i64[]" = torch.ops.aten.add.Tensor(primals_921, 1)
        var_mean_153 = torch.ops.aten.var_mean.correction(convolution_153, [0, 2, 3], correction = 0, keepdim = True)
        getitem_308: "f32[1, 512, 1, 1]" = var_mean_153[0]
        getitem_309: "f32[1, 512, 1, 1]" = var_mean_153[1];  var_mean_153 = None
        add_815: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_308, 1e-05)
        rsqrt_153: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_815);  add_815 = None
        sub_153: "f32[32, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_153, getitem_309)
        mul_1071: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_153, rsqrt_153);  sub_153 = None
        squeeze_459: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_309, [0, 2, 3]);  getitem_309 = None
        squeeze_460: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_153, [0, 2, 3]);  rsqrt_153 = None
        mul_1072: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_459, 0.1)
        mul_1073: "f32[512]" = torch.ops.aten.mul.Tensor(primals_922, 0.9)
        add_816: "f32[512]" = torch.ops.aten.add.Tensor(mul_1072, mul_1073);  mul_1072 = mul_1073 = None
        squeeze_461: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_308, [0, 2, 3]);  getitem_308 = None
        mul_1074: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_461, 1.0006381620931717);  squeeze_461 = None
        mul_1075: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1074, 0.1);  mul_1074 = None
        mul_1076: "f32[512]" = torch.ops.aten.mul.Tensor(primals_923, 0.9)
        add_817: "f32[512]" = torch.ops.aten.add.Tensor(mul_1075, mul_1076);  mul_1075 = mul_1076 = None
        unsqueeze_612: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_924, -1)
        unsqueeze_613: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_1077: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1071, unsqueeze_613);  mul_1071 = unsqueeze_613 = None
        unsqueeze_614: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_925, -1);  primals_925 = None
        unsqueeze_615: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_818: "f32[32, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_1077, unsqueeze_615);  mul_1077 = unsqueeze_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_149: "f32[32, 512, 7, 7]" = torch.ops.aten.relu.default(add_818);  add_818 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_154: "f32[32, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_149, primals_926, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_819: "i64[]" = torch.ops.aten.add.Tensor(primals_927, 1)
        var_mean_154 = torch.ops.aten.var_mean.correction(convolution_154, [0, 2, 3], correction = 0, keepdim = True)
        getitem_310: "f32[1, 2048, 1, 1]" = var_mean_154[0]
        getitem_311: "f32[1, 2048, 1, 1]" = var_mean_154[1];  var_mean_154 = None
        add_820: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_310, 1e-05)
        rsqrt_154: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_820);  add_820 = None
        sub_154: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_154, getitem_311)
        mul_1078: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_154, rsqrt_154);  sub_154 = None
        squeeze_462: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_311, [0, 2, 3]);  getitem_311 = None
        squeeze_463: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_154, [0, 2, 3]);  rsqrt_154 = None
        mul_1079: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_462, 0.1)
        mul_1080: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_928, 0.9)
        add_821: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1079, mul_1080);  mul_1079 = mul_1080 = None
        squeeze_464: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_310, [0, 2, 3]);  getitem_310 = None
        mul_1081: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_464, 1.0006381620931717);  squeeze_464 = None
        mul_1082: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1081, 0.1);  mul_1081 = None
        mul_1083: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_929, 0.9)
        add_822: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1082, mul_1083);  mul_1082 = mul_1083 = None
        unsqueeze_616: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_930, -1)
        unsqueeze_617: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        mul_1084: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1078, unsqueeze_617);  mul_1078 = unsqueeze_617 = None
        unsqueeze_618: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_931, -1);  primals_931 = None
        unsqueeze_619: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        add_823: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_1084, unsqueeze_619);  mul_1084 = unsqueeze_619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_824: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_823, relu_147);  add_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_150: "f32[32, 2048, 7, 7]" = torch.ops.aten.relu.default(add_824);  add_824 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in _forward_impl, code: x = self.avgpool(x)
        mean: "f32[32, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_150, [-1, -2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in _forward_impl, code: x = torch.flatten(x, 1)
        view: "f32[32, 2048]" = torch.ops.aten.reshape.default(mean, [32, 2048]);  mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:280 in _forward_impl, code: x = self.fc(x)
        permute: "f32[2048, 1000]" = torch.ops.aten.permute.default(primals_932, [1, 0])
        addmm: "f32[32, 1000]" = torch.ops.aten.addmm.default(primals_933, view, permute);  primals_933 = permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le: "b8[32, 2048, 7, 7]" = torch.ops.aten.le.Scalar(relu_150, 0);  relu_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_620: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_462, 0);  squeeze_462 = None
        unsqueeze_621: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
        unsqueeze_622: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_632: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_459, 0);  squeeze_459 = None
        unsqueeze_633: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
        unsqueeze_634: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_644: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_456, 0);  squeeze_456 = None
        unsqueeze_645: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
        unsqueeze_646: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_656: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_453, 0);  squeeze_453 = None
        unsqueeze_657: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
        unsqueeze_658: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_668: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_450, 0);  squeeze_450 = None
        unsqueeze_669: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
        unsqueeze_670: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_680: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_447, 0);  squeeze_447 = None
        unsqueeze_681: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
        unsqueeze_682: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        unsqueeze_692: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_444, 0);  squeeze_444 = None
        unsqueeze_693: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
        unsqueeze_694: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_704: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_441, 0);  squeeze_441 = None
        unsqueeze_705: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
        unsqueeze_706: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_716: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_438, 0);  squeeze_438 = None
        unsqueeze_717: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
        unsqueeze_718: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_728: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_435, 0);  squeeze_435 = None
        unsqueeze_729: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
        unsqueeze_730: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_740: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_432, 0);  squeeze_432 = None
        unsqueeze_741: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
        unsqueeze_742: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_752: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_429, 0);  squeeze_429 = None
        unsqueeze_753: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
        unsqueeze_754: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_764: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_426, 0);  squeeze_426 = None
        unsqueeze_765: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
        unsqueeze_766: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_776: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_423, 0);  squeeze_423 = None
        unsqueeze_777: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
        unsqueeze_778: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_788: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_420, 0);  squeeze_420 = None
        unsqueeze_789: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
        unsqueeze_790: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_800: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_417, 0);  squeeze_417 = None
        unsqueeze_801: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
        unsqueeze_802: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_812: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_414, 0);  squeeze_414 = None
        unsqueeze_813: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
        unsqueeze_814: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_824: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_411, 0);  squeeze_411 = None
        unsqueeze_825: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
        unsqueeze_826: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_836: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_408, 0);  squeeze_408 = None
        unsqueeze_837: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
        unsqueeze_838: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_848: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_405, 0);  squeeze_405 = None
        unsqueeze_849: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 2);  unsqueeze_848 = None
        unsqueeze_850: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 3);  unsqueeze_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_860: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_402, 0);  squeeze_402 = None
        unsqueeze_861: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 2);  unsqueeze_860 = None
        unsqueeze_862: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 3);  unsqueeze_861 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_872: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_399, 0);  squeeze_399 = None
        unsqueeze_873: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 2);  unsqueeze_872 = None
        unsqueeze_874: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 3);  unsqueeze_873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_884: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_396, 0);  squeeze_396 = None
        unsqueeze_885: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 2);  unsqueeze_884 = None
        unsqueeze_886: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 3);  unsqueeze_885 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_896: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_393, 0);  squeeze_393 = None
        unsqueeze_897: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 2);  unsqueeze_896 = None
        unsqueeze_898: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 3);  unsqueeze_897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_908: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_390, 0);  squeeze_390 = None
        unsqueeze_909: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 2);  unsqueeze_908 = None
        unsqueeze_910: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 3);  unsqueeze_909 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_920: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_387, 0);  squeeze_387 = None
        unsqueeze_921: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 2);  unsqueeze_920 = None
        unsqueeze_922: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 3);  unsqueeze_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_932: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_384, 0);  squeeze_384 = None
        unsqueeze_933: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 2);  unsqueeze_932 = None
        unsqueeze_934: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 3);  unsqueeze_933 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_944: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_381, 0);  squeeze_381 = None
        unsqueeze_945: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 2);  unsqueeze_944 = None
        unsqueeze_946: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 3);  unsqueeze_945 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_956: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_378, 0);  squeeze_378 = None
        unsqueeze_957: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 2);  unsqueeze_956 = None
        unsqueeze_958: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 3);  unsqueeze_957 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_968: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_375, 0);  squeeze_375 = None
        unsqueeze_969: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 2);  unsqueeze_968 = None
        unsqueeze_970: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 3);  unsqueeze_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_980: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_372, 0);  squeeze_372 = None
        unsqueeze_981: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 2);  unsqueeze_980 = None
        unsqueeze_982: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 3);  unsqueeze_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_992: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_369, 0);  squeeze_369 = None
        unsqueeze_993: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 2);  unsqueeze_992 = None
        unsqueeze_994: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_993, 3);  unsqueeze_993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1004: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_366, 0);  squeeze_366 = None
        unsqueeze_1005: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 2);  unsqueeze_1004 = None
        unsqueeze_1006: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 3);  unsqueeze_1005 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1016: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_363, 0);  squeeze_363 = None
        unsqueeze_1017: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 2);  unsqueeze_1016 = None
        unsqueeze_1018: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 3);  unsqueeze_1017 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1028: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_360, 0);  squeeze_360 = None
        unsqueeze_1029: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 2);  unsqueeze_1028 = None
        unsqueeze_1030: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 3);  unsqueeze_1029 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1040: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_357, 0);  squeeze_357 = None
        unsqueeze_1041: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, 2);  unsqueeze_1040 = None
        unsqueeze_1042: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1041, 3);  unsqueeze_1041 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1052: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_354, 0);  squeeze_354 = None
        unsqueeze_1053: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, 2);  unsqueeze_1052 = None
        unsqueeze_1054: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1053, 3);  unsqueeze_1053 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1064: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_351, 0);  squeeze_351 = None
        unsqueeze_1065: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, 2);  unsqueeze_1064 = None
        unsqueeze_1066: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1065, 3);  unsqueeze_1065 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1076: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_348, 0);  squeeze_348 = None
        unsqueeze_1077: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, 2);  unsqueeze_1076 = None
        unsqueeze_1078: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1077, 3);  unsqueeze_1077 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1088: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_345, 0);  squeeze_345 = None
        unsqueeze_1089: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, 2);  unsqueeze_1088 = None
        unsqueeze_1090: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1089, 3);  unsqueeze_1089 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1100: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_342, 0);  squeeze_342 = None
        unsqueeze_1101: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, 2);  unsqueeze_1100 = None
        unsqueeze_1102: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1101, 3);  unsqueeze_1101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1112: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_339, 0);  squeeze_339 = None
        unsqueeze_1113: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, 2);  unsqueeze_1112 = None
        unsqueeze_1114: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1113, 3);  unsqueeze_1113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1124: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_336, 0);  squeeze_336 = None
        unsqueeze_1125: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, 2);  unsqueeze_1124 = None
        unsqueeze_1126: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1125, 3);  unsqueeze_1125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1136: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_333, 0);  squeeze_333 = None
        unsqueeze_1137: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, 2);  unsqueeze_1136 = None
        unsqueeze_1138: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1137, 3);  unsqueeze_1137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1148: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_330, 0);  squeeze_330 = None
        unsqueeze_1149: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, 2);  unsqueeze_1148 = None
        unsqueeze_1150: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1149, 3);  unsqueeze_1149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1160: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_327, 0);  squeeze_327 = None
        unsqueeze_1161: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, 2);  unsqueeze_1160 = None
        unsqueeze_1162: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1161, 3);  unsqueeze_1161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1172: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_324, 0);  squeeze_324 = None
        unsqueeze_1173: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, 2);  unsqueeze_1172 = None
        unsqueeze_1174: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1173, 3);  unsqueeze_1173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1184: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_321, 0);  squeeze_321 = None
        unsqueeze_1185: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, 2);  unsqueeze_1184 = None
        unsqueeze_1186: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1185, 3);  unsqueeze_1185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1196: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_318, 0);  squeeze_318 = None
        unsqueeze_1197: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, 2);  unsqueeze_1196 = None
        unsqueeze_1198: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1197, 3);  unsqueeze_1197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1208: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_315, 0);  squeeze_315 = None
        unsqueeze_1209: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, 2);  unsqueeze_1208 = None
        unsqueeze_1210: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1209, 3);  unsqueeze_1209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1220: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_312, 0);  squeeze_312 = None
        unsqueeze_1221: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, 2);  unsqueeze_1220 = None
        unsqueeze_1222: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1221, 3);  unsqueeze_1221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1232: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_309, 0);  squeeze_309 = None
        unsqueeze_1233: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, 2);  unsqueeze_1232 = None
        unsqueeze_1234: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1233, 3);  unsqueeze_1233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1244: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_306, 0);  squeeze_306 = None
        unsqueeze_1245: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, 2);  unsqueeze_1244 = None
        unsqueeze_1246: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1245, 3);  unsqueeze_1245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1256: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_303, 0);  squeeze_303 = None
        unsqueeze_1257: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, 2);  unsqueeze_1256 = None
        unsqueeze_1258: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1257, 3);  unsqueeze_1257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1268: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_300, 0);  squeeze_300 = None
        unsqueeze_1269: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, 2);  unsqueeze_1268 = None
        unsqueeze_1270: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1269, 3);  unsqueeze_1269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1280: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_297, 0);  squeeze_297 = None
        unsqueeze_1281: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1280, 2);  unsqueeze_1280 = None
        unsqueeze_1282: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1281, 3);  unsqueeze_1281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1292: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_294, 0);  squeeze_294 = None
        unsqueeze_1293: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1292, 2);  unsqueeze_1292 = None
        unsqueeze_1294: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1293, 3);  unsqueeze_1293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1304: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_291, 0);  squeeze_291 = None
        unsqueeze_1305: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1304, 2);  unsqueeze_1304 = None
        unsqueeze_1306: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1305, 3);  unsqueeze_1305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1316: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_288, 0);  squeeze_288 = None
        unsqueeze_1317: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1316, 2);  unsqueeze_1316 = None
        unsqueeze_1318: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1317, 3);  unsqueeze_1317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1328: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_285, 0);  squeeze_285 = None
        unsqueeze_1329: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1328, 2);  unsqueeze_1328 = None
        unsqueeze_1330: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1329, 3);  unsqueeze_1329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1340: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_282, 0);  squeeze_282 = None
        unsqueeze_1341: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1340, 2);  unsqueeze_1340 = None
        unsqueeze_1342: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1341, 3);  unsqueeze_1341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1352: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_279, 0);  squeeze_279 = None
        unsqueeze_1353: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1352, 2);  unsqueeze_1352 = None
        unsqueeze_1354: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1353, 3);  unsqueeze_1353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1364: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_276, 0);  squeeze_276 = None
        unsqueeze_1365: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1364, 2);  unsqueeze_1364 = None
        unsqueeze_1366: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1365, 3);  unsqueeze_1365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1376: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_273, 0);  squeeze_273 = None
        unsqueeze_1377: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1376, 2);  unsqueeze_1376 = None
        unsqueeze_1378: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1377, 3);  unsqueeze_1377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1388: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_270, 0);  squeeze_270 = None
        unsqueeze_1389: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1388, 2);  unsqueeze_1388 = None
        unsqueeze_1390: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1389, 3);  unsqueeze_1389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1400: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_267, 0);  squeeze_267 = None
        unsqueeze_1401: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1400, 2);  unsqueeze_1400 = None
        unsqueeze_1402: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1401, 3);  unsqueeze_1401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1412: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_264, 0);  squeeze_264 = None
        unsqueeze_1413: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1412, 2);  unsqueeze_1412 = None
        unsqueeze_1414: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1413, 3);  unsqueeze_1413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1424: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_261, 0);  squeeze_261 = None
        unsqueeze_1425: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1424, 2);  unsqueeze_1424 = None
        unsqueeze_1426: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1425, 3);  unsqueeze_1425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1436: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_258, 0);  squeeze_258 = None
        unsqueeze_1437: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1436, 2);  unsqueeze_1436 = None
        unsqueeze_1438: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1437, 3);  unsqueeze_1437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1448: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_255, 0);  squeeze_255 = None
        unsqueeze_1449: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1448, 2);  unsqueeze_1448 = None
        unsqueeze_1450: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1449, 3);  unsqueeze_1449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1460: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_252, 0);  squeeze_252 = None
        unsqueeze_1461: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1460, 2);  unsqueeze_1460 = None
        unsqueeze_1462: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1461, 3);  unsqueeze_1461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1472: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_249, 0);  squeeze_249 = None
        unsqueeze_1473: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1472, 2);  unsqueeze_1472 = None
        unsqueeze_1474: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1473, 3);  unsqueeze_1473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1484: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_246, 0);  squeeze_246 = None
        unsqueeze_1485: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1484, 2);  unsqueeze_1484 = None
        unsqueeze_1486: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1485, 3);  unsqueeze_1485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1496: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_243, 0);  squeeze_243 = None
        unsqueeze_1497: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1496, 2);  unsqueeze_1496 = None
        unsqueeze_1498: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1497, 3);  unsqueeze_1497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1508: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_240, 0);  squeeze_240 = None
        unsqueeze_1509: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1508, 2);  unsqueeze_1508 = None
        unsqueeze_1510: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1509, 3);  unsqueeze_1509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1520: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_237, 0);  squeeze_237 = None
        unsqueeze_1521: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1520, 2);  unsqueeze_1520 = None
        unsqueeze_1522: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1521, 3);  unsqueeze_1521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1532: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_234, 0);  squeeze_234 = None
        unsqueeze_1533: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1532, 2);  unsqueeze_1532 = None
        unsqueeze_1534: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1533, 3);  unsqueeze_1533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1544: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_231, 0);  squeeze_231 = None
        unsqueeze_1545: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1544, 2);  unsqueeze_1544 = None
        unsqueeze_1546: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1545, 3);  unsqueeze_1545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1556: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_228, 0);  squeeze_228 = None
        unsqueeze_1557: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1556, 2);  unsqueeze_1556 = None
        unsqueeze_1558: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1557, 3);  unsqueeze_1557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1568: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_225, 0);  squeeze_225 = None
        unsqueeze_1569: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1568, 2);  unsqueeze_1568 = None
        unsqueeze_1570: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1569, 3);  unsqueeze_1569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1580: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_222, 0);  squeeze_222 = None
        unsqueeze_1581: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1580, 2);  unsqueeze_1580 = None
        unsqueeze_1582: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1581, 3);  unsqueeze_1581 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1592: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_219, 0);  squeeze_219 = None
        unsqueeze_1593: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1592, 2);  unsqueeze_1592 = None
        unsqueeze_1594: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1593, 3);  unsqueeze_1593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1604: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_216, 0);  squeeze_216 = None
        unsqueeze_1605: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1604, 2);  unsqueeze_1604 = None
        unsqueeze_1606: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1605, 3);  unsqueeze_1605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1616: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_213, 0);  squeeze_213 = None
        unsqueeze_1617: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1616, 2);  unsqueeze_1616 = None
        unsqueeze_1618: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1617, 3);  unsqueeze_1617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1628: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_210, 0);  squeeze_210 = None
        unsqueeze_1629: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1628, 2);  unsqueeze_1628 = None
        unsqueeze_1630: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1629, 3);  unsqueeze_1629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1640: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_207, 0);  squeeze_207 = None
        unsqueeze_1641: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1640, 2);  unsqueeze_1640 = None
        unsqueeze_1642: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1641, 3);  unsqueeze_1641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1652: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_204, 0);  squeeze_204 = None
        unsqueeze_1653: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1652, 2);  unsqueeze_1652 = None
        unsqueeze_1654: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1653, 3);  unsqueeze_1653 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1664: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_201, 0);  squeeze_201 = None
        unsqueeze_1665: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1664, 2);  unsqueeze_1664 = None
        unsqueeze_1666: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1665, 3);  unsqueeze_1665 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1676: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
        unsqueeze_1677: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1676, 2);  unsqueeze_1676 = None
        unsqueeze_1678: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1677, 3);  unsqueeze_1677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1688: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
        unsqueeze_1689: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1688, 2);  unsqueeze_1688 = None
        unsqueeze_1690: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1689, 3);  unsqueeze_1689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1700: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
        unsqueeze_1701: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1700, 2);  unsqueeze_1700 = None
        unsqueeze_1702: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1701, 3);  unsqueeze_1701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1712: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
        unsqueeze_1713: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1712, 2);  unsqueeze_1712 = None
        unsqueeze_1714: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1713, 3);  unsqueeze_1713 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1724: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
        unsqueeze_1725: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1724, 2);  unsqueeze_1724 = None
        unsqueeze_1726: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1725, 3);  unsqueeze_1725 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1736: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
        unsqueeze_1737: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1736, 2);  unsqueeze_1736 = None
        unsqueeze_1738: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1737, 3);  unsqueeze_1737 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1748: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
        unsqueeze_1749: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1748, 2);  unsqueeze_1748 = None
        unsqueeze_1750: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1749, 3);  unsqueeze_1749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1760: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
        unsqueeze_1761: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1760, 2);  unsqueeze_1760 = None
        unsqueeze_1762: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1761, 3);  unsqueeze_1761 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1772: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
        unsqueeze_1773: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1772, 2);  unsqueeze_1772 = None
        unsqueeze_1774: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1773, 3);  unsqueeze_1773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1784: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
        unsqueeze_1785: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1784, 2);  unsqueeze_1784 = None
        unsqueeze_1786: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1785, 3);  unsqueeze_1785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1796: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
        unsqueeze_1797: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1796, 2);  unsqueeze_1796 = None
        unsqueeze_1798: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1797, 3);  unsqueeze_1797 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1808: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
        unsqueeze_1809: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1808, 2);  unsqueeze_1808 = None
        unsqueeze_1810: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1809, 3);  unsqueeze_1809 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1820: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
        unsqueeze_1821: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1820, 2);  unsqueeze_1820 = None
        unsqueeze_1822: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1821, 3);  unsqueeze_1821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1832: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
        unsqueeze_1833: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1832, 2);  unsqueeze_1832 = None
        unsqueeze_1834: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1833, 3);  unsqueeze_1833 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1844: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
        unsqueeze_1845: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1844, 2);  unsqueeze_1844 = None
        unsqueeze_1846: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1845, 3);  unsqueeze_1845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1856: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
        unsqueeze_1857: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1856, 2);  unsqueeze_1856 = None
        unsqueeze_1858: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1857, 3);  unsqueeze_1857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1868: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
        unsqueeze_1869: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1868, 2);  unsqueeze_1868 = None
        unsqueeze_1870: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1869, 3);  unsqueeze_1869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1880: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
        unsqueeze_1881: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1880, 2);  unsqueeze_1880 = None
        unsqueeze_1882: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1881, 3);  unsqueeze_1881 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1892: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
        unsqueeze_1893: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1892, 2);  unsqueeze_1892 = None
        unsqueeze_1894: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1893, 3);  unsqueeze_1893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1904: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
        unsqueeze_1905: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1904, 2);  unsqueeze_1904 = None
        unsqueeze_1906: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1905, 3);  unsqueeze_1905 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1916: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
        unsqueeze_1917: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1916, 2);  unsqueeze_1916 = None
        unsqueeze_1918: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1917, 3);  unsqueeze_1917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1928: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
        unsqueeze_1929: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1928, 2);  unsqueeze_1928 = None
        unsqueeze_1930: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1929, 3);  unsqueeze_1929 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1940: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
        unsqueeze_1941: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1940, 2);  unsqueeze_1940 = None
        unsqueeze_1942: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1941, 3);  unsqueeze_1941 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1952: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
        unsqueeze_1953: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1952, 2);  unsqueeze_1952 = None
        unsqueeze_1954: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1953, 3);  unsqueeze_1953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_1964: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
        unsqueeze_1965: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1964, 2);  unsqueeze_1964 = None
        unsqueeze_1966: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1965, 3);  unsqueeze_1965 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_1976: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
        unsqueeze_1977: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1976, 2);  unsqueeze_1976 = None
        unsqueeze_1978: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1977, 3);  unsqueeze_1977 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_1988: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
        unsqueeze_1989: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1988, 2);  unsqueeze_1988 = None
        unsqueeze_1990: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1989, 3);  unsqueeze_1989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        unsqueeze_2000: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
        unsqueeze_2001: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2000, 2);  unsqueeze_2000 = None
        unsqueeze_2002: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2001, 3);  unsqueeze_2001 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_2012: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
        unsqueeze_2013: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2012, 2);  unsqueeze_2012 = None
        unsqueeze_2014: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2013, 3);  unsqueeze_2013 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_2024: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_2025: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2024, 2);  unsqueeze_2024 = None
        unsqueeze_2026: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2025, 3);  unsqueeze_2025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_2036: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_2037: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2036, 2);  unsqueeze_2036 = None
        unsqueeze_2038: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2037, 3);  unsqueeze_2037 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_2048: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
        unsqueeze_2049: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2048, 2);  unsqueeze_2048 = None
        unsqueeze_2050: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2049, 3);  unsqueeze_2049 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_2060: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_2061: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2060, 2);  unsqueeze_2060 = None
        unsqueeze_2062: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2061, 3);  unsqueeze_2061 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_2072: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_2073: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2072, 2);  unsqueeze_2072 = None
        unsqueeze_2074: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2073, 3);  unsqueeze_2073 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_2084: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
        unsqueeze_2085: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2084, 2);  unsqueeze_2084 = None
        unsqueeze_2086: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2085, 3);  unsqueeze_2085 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_2096: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_2097: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2096, 2);  unsqueeze_2096 = None
        unsqueeze_2098: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2097, 3);  unsqueeze_2097 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_2108: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_2109: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2108, 2);  unsqueeze_2108 = None
        unsqueeze_2110: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2109, 3);  unsqueeze_2109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_2120: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_2121: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2120, 2);  unsqueeze_2120 = None
        unsqueeze_2122: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2121, 3);  unsqueeze_2121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_2132: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_2133: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2132, 2);  unsqueeze_2132 = None
        unsqueeze_2134: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2133, 3);  unsqueeze_2133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_2144: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_2145: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2144, 2);  unsqueeze_2144 = None
        unsqueeze_2146: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2145, 3);  unsqueeze_2145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_2156: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_2157: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2156, 2);  unsqueeze_2156 = None
        unsqueeze_2158: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2157, 3);  unsqueeze_2157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_2168: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_2169: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2168, 2);  unsqueeze_2168 = None
        unsqueeze_2170: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2169, 3);  unsqueeze_2169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_2180: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_2181: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2180, 2);  unsqueeze_2180 = None
        unsqueeze_2182: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2181, 3);  unsqueeze_2181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_2192: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_2193: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2192, 2);  unsqueeze_2192 = None
        unsqueeze_2194: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2193, 3);  unsqueeze_2193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_2204: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_2205: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2204, 2);  unsqueeze_2204 = None
        unsqueeze_2206: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2205, 3);  unsqueeze_2205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_2216: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_2217: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2216, 2);  unsqueeze_2216 = None
        unsqueeze_2218: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2217, 3);  unsqueeze_2217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_2228: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_2229: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2228, 2);  unsqueeze_2228 = None
        unsqueeze_2230: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2229, 3);  unsqueeze_2229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_2240: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_2241: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2240, 2);  unsqueeze_2240 = None
        unsqueeze_2242: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2241, 3);  unsqueeze_2241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_2252: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_2253: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2252, 2);  unsqueeze_2252 = None
        unsqueeze_2254: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2253, 3);  unsqueeze_2253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_2264: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_2265: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2264, 2);  unsqueeze_2264 = None
        unsqueeze_2266: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2265, 3);  unsqueeze_2265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_2276: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_2277: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2276, 2);  unsqueeze_2276 = None
        unsqueeze_2278: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2277, 3);  unsqueeze_2277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_2288: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_2289: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2288, 2);  unsqueeze_2288 = None
        unsqueeze_2290: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2289, 3);  unsqueeze_2289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        unsqueeze_2300: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_2301: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2300, 2);  unsqueeze_2300 = None
        unsqueeze_2302: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2301, 3);  unsqueeze_2301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_2312: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_2313: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2312, 2);  unsqueeze_2312 = None
        unsqueeze_2314: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2313, 3);  unsqueeze_2313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_2324: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_2325: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2324, 2);  unsqueeze_2324 = None
        unsqueeze_2326: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2325, 3);  unsqueeze_2325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_2336: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_2337: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2336, 2);  unsqueeze_2336 = None
        unsqueeze_2338: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2337, 3);  unsqueeze_2337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_2348: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_2349: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2348, 2);  unsqueeze_2348 = None
        unsqueeze_2350: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2349, 3);  unsqueeze_2349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_2360: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_2361: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2360, 2);  unsqueeze_2360 = None
        unsqueeze_2362: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2361, 3);  unsqueeze_2361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_2372: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_2373: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2372, 2);  unsqueeze_2372 = None
        unsqueeze_2374: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2373, 3);  unsqueeze_2373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_2384: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_2385: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2384, 2);  unsqueeze_2384 = None
        unsqueeze_2386: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2385, 3);  unsqueeze_2385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_2396: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_2397: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2396, 2);  unsqueeze_2396 = None
        unsqueeze_2398: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2397, 3);  unsqueeze_2397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_2408: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_2409: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2408, 2);  unsqueeze_2408 = None
        unsqueeze_2410: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2409, 3);  unsqueeze_2409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        unsqueeze_2420: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_2421: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2420, 2);  unsqueeze_2420 = None
        unsqueeze_2422: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2421, 3);  unsqueeze_2421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        unsqueeze_2432: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_2433: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2432, 2);  unsqueeze_2432 = None
        unsqueeze_2434: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2433, 3);  unsqueeze_2433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        unsqueeze_2444: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_2445: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2444, 2);  unsqueeze_2444 = None
        unsqueeze_2446: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2445, 3);  unsqueeze_2445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        unsqueeze_2456: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_2457: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2456, 2);  unsqueeze_2456 = None
        unsqueeze_2458: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2457, 3);  unsqueeze_2457 = None

        # No stacktrace found for following nodes
        copy_: "i64[]" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[64]" = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2: "f32[64]" = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_9, add_5);  primals_9 = add_5 = copy__3 = None
        copy__4: "f32[64]" = torch.ops.aten.copy_.default(primals_10, add_7);  primals_10 = add_7 = copy__4 = None
        copy__5: "f32[64]" = torch.ops.aten.copy_.default(primals_11, add_8);  primals_11 = add_8 = copy__5 = None
        copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_15, add_10);  primals_15 = add_10 = copy__6 = None
        copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_16, add_12);  primals_16 = add_12 = copy__7 = None
        copy__8: "f32[64]" = torch.ops.aten.copy_.default(primals_17, add_13);  primals_17 = add_13 = copy__8 = None
        copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_21, add_15);  primals_21 = add_15 = copy__9 = None
        copy__10: "f32[256]" = torch.ops.aten.copy_.default(primals_22, add_17);  primals_22 = add_17 = copy__10 = None
        copy__11: "f32[256]" = torch.ops.aten.copy_.default(primals_23, add_18);  primals_23 = add_18 = copy__11 = None
        copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_27, add_20);  primals_27 = add_20 = copy__12 = None
        copy__13: "f32[256]" = torch.ops.aten.copy_.default(primals_28, add_22);  primals_28 = add_22 = copy__13 = None
        copy__14: "f32[256]" = torch.ops.aten.copy_.default(primals_29, add_23);  primals_29 = add_23 = copy__14 = None
        copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_33, add_26);  primals_33 = add_26 = copy__15 = None
        copy__16: "f32[64]" = torch.ops.aten.copy_.default(primals_34, add_28);  primals_34 = add_28 = copy__16 = None
        copy__17: "f32[64]" = torch.ops.aten.copy_.default(primals_35, add_29);  primals_35 = add_29 = copy__17 = None
        copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_39, add_31);  primals_39 = add_31 = copy__18 = None
        copy__19: "f32[64]" = torch.ops.aten.copy_.default(primals_40, add_33);  primals_40 = add_33 = copy__19 = None
        copy__20: "f32[64]" = torch.ops.aten.copy_.default(primals_41, add_34);  primals_41 = add_34 = copy__20 = None
        copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_45, add_36);  primals_45 = add_36 = copy__21 = None
        copy__22: "f32[256]" = torch.ops.aten.copy_.default(primals_46, add_38);  primals_46 = add_38 = copy__22 = None
        copy__23: "f32[256]" = torch.ops.aten.copy_.default(primals_47, add_39);  primals_47 = add_39 = copy__23 = None
        copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_51, add_42);  primals_51 = add_42 = copy__24 = None
        copy__25: "f32[64]" = torch.ops.aten.copy_.default(primals_52, add_44);  primals_52 = add_44 = copy__25 = None
        copy__26: "f32[64]" = torch.ops.aten.copy_.default(primals_53, add_45);  primals_53 = add_45 = copy__26 = None
        copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_57, add_47);  primals_57 = add_47 = copy__27 = None
        copy__28: "f32[64]" = torch.ops.aten.copy_.default(primals_58, add_49);  primals_58 = add_49 = copy__28 = None
        copy__29: "f32[64]" = torch.ops.aten.copy_.default(primals_59, add_50);  primals_59 = add_50 = copy__29 = None
        copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_63, add_52);  primals_63 = add_52 = copy__30 = None
        copy__31: "f32[256]" = torch.ops.aten.copy_.default(primals_64, add_54);  primals_64 = add_54 = copy__31 = None
        copy__32: "f32[256]" = torch.ops.aten.copy_.default(primals_65, add_55);  primals_65 = add_55 = copy__32 = None
        copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_69, add_58);  primals_69 = add_58 = copy__33 = None
        copy__34: "f32[128]" = torch.ops.aten.copy_.default(primals_70, add_60);  primals_70 = add_60 = copy__34 = None
        copy__35: "f32[128]" = torch.ops.aten.copy_.default(primals_71, add_61);  primals_71 = add_61 = copy__35 = None
        copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_75, add_63);  primals_75 = add_63 = copy__36 = None
        copy__37: "f32[128]" = torch.ops.aten.copy_.default(primals_76, add_65);  primals_76 = add_65 = copy__37 = None
        copy__38: "f32[128]" = torch.ops.aten.copy_.default(primals_77, add_66);  primals_77 = add_66 = copy__38 = None
        copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_81, add_68);  primals_81 = add_68 = copy__39 = None
        copy__40: "f32[512]" = torch.ops.aten.copy_.default(primals_82, add_70);  primals_82 = add_70 = copy__40 = None
        copy__41: "f32[512]" = torch.ops.aten.copy_.default(primals_83, add_71);  primals_83 = add_71 = copy__41 = None
        copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_87, add_73);  primals_87 = add_73 = copy__42 = None
        copy__43: "f32[512]" = torch.ops.aten.copy_.default(primals_88, add_75);  primals_88 = add_75 = copy__43 = None
        copy__44: "f32[512]" = torch.ops.aten.copy_.default(primals_89, add_76);  primals_89 = add_76 = copy__44 = None
        copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_93, add_79);  primals_93 = add_79 = copy__45 = None
        copy__46: "f32[128]" = torch.ops.aten.copy_.default(primals_94, add_81);  primals_94 = add_81 = copy__46 = None
        copy__47: "f32[128]" = torch.ops.aten.copy_.default(primals_95, add_82);  primals_95 = add_82 = copy__47 = None
        copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_99, add_84);  primals_99 = add_84 = copy__48 = None
        copy__49: "f32[128]" = torch.ops.aten.copy_.default(primals_100, add_86);  primals_100 = add_86 = copy__49 = None
        copy__50: "f32[128]" = torch.ops.aten.copy_.default(primals_101, add_87);  primals_101 = add_87 = copy__50 = None
        copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_105, add_89);  primals_105 = add_89 = copy__51 = None
        copy__52: "f32[512]" = torch.ops.aten.copy_.default(primals_106, add_91);  primals_106 = add_91 = copy__52 = None
        copy__53: "f32[512]" = torch.ops.aten.copy_.default(primals_107, add_92);  primals_107 = add_92 = copy__53 = None
        copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_111, add_95);  primals_111 = add_95 = copy__54 = None
        copy__55: "f32[128]" = torch.ops.aten.copy_.default(primals_112, add_97);  primals_112 = add_97 = copy__55 = None
        copy__56: "f32[128]" = torch.ops.aten.copy_.default(primals_113, add_98);  primals_113 = add_98 = copy__56 = None
        copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_117, add_100);  primals_117 = add_100 = copy__57 = None
        copy__58: "f32[128]" = torch.ops.aten.copy_.default(primals_118, add_102);  primals_118 = add_102 = copy__58 = None
        copy__59: "f32[128]" = torch.ops.aten.copy_.default(primals_119, add_103);  primals_119 = add_103 = copy__59 = None
        copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_123, add_105);  primals_123 = add_105 = copy__60 = None
        copy__61: "f32[512]" = torch.ops.aten.copy_.default(primals_124, add_107);  primals_124 = add_107 = copy__61 = None
        copy__62: "f32[512]" = torch.ops.aten.copy_.default(primals_125, add_108);  primals_125 = add_108 = copy__62 = None
        copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_129, add_111);  primals_129 = add_111 = copy__63 = None
        copy__64: "f32[128]" = torch.ops.aten.copy_.default(primals_130, add_113);  primals_130 = add_113 = copy__64 = None
        copy__65: "f32[128]" = torch.ops.aten.copy_.default(primals_131, add_114);  primals_131 = add_114 = copy__65 = None
        copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_135, add_116);  primals_135 = add_116 = copy__66 = None
        copy__67: "f32[128]" = torch.ops.aten.copy_.default(primals_136, add_118);  primals_136 = add_118 = copy__67 = None
        copy__68: "f32[128]" = torch.ops.aten.copy_.default(primals_137, add_119);  primals_137 = add_119 = copy__68 = None
        copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_141, add_121);  primals_141 = add_121 = copy__69 = None
        copy__70: "f32[512]" = torch.ops.aten.copy_.default(primals_142, add_123);  primals_142 = add_123 = copy__70 = None
        copy__71: "f32[512]" = torch.ops.aten.copy_.default(primals_143, add_124);  primals_143 = add_124 = copy__71 = None
        copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_147, add_127);  primals_147 = add_127 = copy__72 = None
        copy__73: "f32[128]" = torch.ops.aten.copy_.default(primals_148, add_129);  primals_148 = add_129 = copy__73 = None
        copy__74: "f32[128]" = torch.ops.aten.copy_.default(primals_149, add_130);  primals_149 = add_130 = copy__74 = None
        copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_153, add_132);  primals_153 = add_132 = copy__75 = None
        copy__76: "f32[128]" = torch.ops.aten.copy_.default(primals_154, add_134);  primals_154 = add_134 = copy__76 = None
        copy__77: "f32[128]" = torch.ops.aten.copy_.default(primals_155, add_135);  primals_155 = add_135 = copy__77 = None
        copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_159, add_137);  primals_159 = add_137 = copy__78 = None
        copy__79: "f32[512]" = torch.ops.aten.copy_.default(primals_160, add_139);  primals_160 = add_139 = copy__79 = None
        copy__80: "f32[512]" = torch.ops.aten.copy_.default(primals_161, add_140);  primals_161 = add_140 = copy__80 = None
        copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_165, add_143);  primals_165 = add_143 = copy__81 = None
        copy__82: "f32[128]" = torch.ops.aten.copy_.default(primals_166, add_145);  primals_166 = add_145 = copy__82 = None
        copy__83: "f32[128]" = torch.ops.aten.copy_.default(primals_167, add_146);  primals_167 = add_146 = copy__83 = None
        copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_171, add_148);  primals_171 = add_148 = copy__84 = None
        copy__85: "f32[128]" = torch.ops.aten.copy_.default(primals_172, add_150);  primals_172 = add_150 = copy__85 = None
        copy__86: "f32[128]" = torch.ops.aten.copy_.default(primals_173, add_151);  primals_173 = add_151 = copy__86 = None
        copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_177, add_153);  primals_177 = add_153 = copy__87 = None
        copy__88: "f32[512]" = torch.ops.aten.copy_.default(primals_178, add_155);  primals_178 = add_155 = copy__88 = None
        copy__89: "f32[512]" = torch.ops.aten.copy_.default(primals_179, add_156);  primals_179 = add_156 = copy__89 = None
        copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_183, add_159);  primals_183 = add_159 = copy__90 = None
        copy__91: "f32[128]" = torch.ops.aten.copy_.default(primals_184, add_161);  primals_184 = add_161 = copy__91 = None
        copy__92: "f32[128]" = torch.ops.aten.copy_.default(primals_185, add_162);  primals_185 = add_162 = copy__92 = None
        copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_189, add_164);  primals_189 = add_164 = copy__93 = None
        copy__94: "f32[128]" = torch.ops.aten.copy_.default(primals_190, add_166);  primals_190 = add_166 = copy__94 = None
        copy__95: "f32[128]" = torch.ops.aten.copy_.default(primals_191, add_167);  primals_191 = add_167 = copy__95 = None
        copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_195, add_169);  primals_195 = add_169 = copy__96 = None
        copy__97: "f32[512]" = torch.ops.aten.copy_.default(primals_196, add_171);  primals_196 = add_171 = copy__97 = None
        copy__98: "f32[512]" = torch.ops.aten.copy_.default(primals_197, add_172);  primals_197 = add_172 = copy__98 = None
        copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_201, add_175);  primals_201 = add_175 = copy__99 = None
        copy__100: "f32[128]" = torch.ops.aten.copy_.default(primals_202, add_177);  primals_202 = add_177 = copy__100 = None
        copy__101: "f32[128]" = torch.ops.aten.copy_.default(primals_203, add_178);  primals_203 = add_178 = copy__101 = None
        copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_207, add_180);  primals_207 = add_180 = copy__102 = None
        copy__103: "f32[128]" = torch.ops.aten.copy_.default(primals_208, add_182);  primals_208 = add_182 = copy__103 = None
        copy__104: "f32[128]" = torch.ops.aten.copy_.default(primals_209, add_183);  primals_209 = add_183 = copy__104 = None
        copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_213, add_185);  primals_213 = add_185 = copy__105 = None
        copy__106: "f32[512]" = torch.ops.aten.copy_.default(primals_214, add_187);  primals_214 = add_187 = copy__106 = None
        copy__107: "f32[512]" = torch.ops.aten.copy_.default(primals_215, add_188);  primals_215 = add_188 = copy__107 = None
        copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_219, add_191);  primals_219 = add_191 = copy__108 = None
        copy__109: "f32[256]" = torch.ops.aten.copy_.default(primals_220, add_193);  primals_220 = add_193 = copy__109 = None
        copy__110: "f32[256]" = torch.ops.aten.copy_.default(primals_221, add_194);  primals_221 = add_194 = copy__110 = None
        copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_225, add_196);  primals_225 = add_196 = copy__111 = None
        copy__112: "f32[256]" = torch.ops.aten.copy_.default(primals_226, add_198);  primals_226 = add_198 = copy__112 = None
        copy__113: "f32[256]" = torch.ops.aten.copy_.default(primals_227, add_199);  primals_227 = add_199 = copy__113 = None
        copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_231, add_201);  primals_231 = add_201 = copy__114 = None
        copy__115: "f32[1024]" = torch.ops.aten.copy_.default(primals_232, add_203);  primals_232 = add_203 = copy__115 = None
        copy__116: "f32[1024]" = torch.ops.aten.copy_.default(primals_233, add_204);  primals_233 = add_204 = copy__116 = None
        copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_237, add_206);  primals_237 = add_206 = copy__117 = None
        copy__118: "f32[1024]" = torch.ops.aten.copy_.default(primals_238, add_208);  primals_238 = add_208 = copy__118 = None
        copy__119: "f32[1024]" = torch.ops.aten.copy_.default(primals_239, add_209);  primals_239 = add_209 = copy__119 = None
        copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_243, add_212);  primals_243 = add_212 = copy__120 = None
        copy__121: "f32[256]" = torch.ops.aten.copy_.default(primals_244, add_214);  primals_244 = add_214 = copy__121 = None
        copy__122: "f32[256]" = torch.ops.aten.copy_.default(primals_245, add_215);  primals_245 = add_215 = copy__122 = None
        copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_249, add_217);  primals_249 = add_217 = copy__123 = None
        copy__124: "f32[256]" = torch.ops.aten.copy_.default(primals_250, add_219);  primals_250 = add_219 = copy__124 = None
        copy__125: "f32[256]" = torch.ops.aten.copy_.default(primals_251, add_220);  primals_251 = add_220 = copy__125 = None
        copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_255, add_222);  primals_255 = add_222 = copy__126 = None
        copy__127: "f32[1024]" = torch.ops.aten.copy_.default(primals_256, add_224);  primals_256 = add_224 = copy__127 = None
        copy__128: "f32[1024]" = torch.ops.aten.copy_.default(primals_257, add_225);  primals_257 = add_225 = copy__128 = None
        copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_261, add_228);  primals_261 = add_228 = copy__129 = None
        copy__130: "f32[256]" = torch.ops.aten.copy_.default(primals_262, add_230);  primals_262 = add_230 = copy__130 = None
        copy__131: "f32[256]" = torch.ops.aten.copy_.default(primals_263, add_231);  primals_263 = add_231 = copy__131 = None
        copy__132: "i64[]" = torch.ops.aten.copy_.default(primals_267, add_233);  primals_267 = add_233 = copy__132 = None
        copy__133: "f32[256]" = torch.ops.aten.copy_.default(primals_268, add_235);  primals_268 = add_235 = copy__133 = None
        copy__134: "f32[256]" = torch.ops.aten.copy_.default(primals_269, add_236);  primals_269 = add_236 = copy__134 = None
        copy__135: "i64[]" = torch.ops.aten.copy_.default(primals_273, add_238);  primals_273 = add_238 = copy__135 = None
        copy__136: "f32[1024]" = torch.ops.aten.copy_.default(primals_274, add_240);  primals_274 = add_240 = copy__136 = None
        copy__137: "f32[1024]" = torch.ops.aten.copy_.default(primals_275, add_241);  primals_275 = add_241 = copy__137 = None
        copy__138: "i64[]" = torch.ops.aten.copy_.default(primals_279, add_244);  primals_279 = add_244 = copy__138 = None
        copy__139: "f32[256]" = torch.ops.aten.copy_.default(primals_280, add_246);  primals_280 = add_246 = copy__139 = None
        copy__140: "f32[256]" = torch.ops.aten.copy_.default(primals_281, add_247);  primals_281 = add_247 = copy__140 = None
        copy__141: "i64[]" = torch.ops.aten.copy_.default(primals_285, add_249);  primals_285 = add_249 = copy__141 = None
        copy__142: "f32[256]" = torch.ops.aten.copy_.default(primals_286, add_251);  primals_286 = add_251 = copy__142 = None
        copy__143: "f32[256]" = torch.ops.aten.copy_.default(primals_287, add_252);  primals_287 = add_252 = copy__143 = None
        copy__144: "i64[]" = torch.ops.aten.copy_.default(primals_291, add_254);  primals_291 = add_254 = copy__144 = None
        copy__145: "f32[1024]" = torch.ops.aten.copy_.default(primals_292, add_256);  primals_292 = add_256 = copy__145 = None
        copy__146: "f32[1024]" = torch.ops.aten.copy_.default(primals_293, add_257);  primals_293 = add_257 = copy__146 = None
        copy__147: "i64[]" = torch.ops.aten.copy_.default(primals_297, add_260);  primals_297 = add_260 = copy__147 = None
        copy__148: "f32[256]" = torch.ops.aten.copy_.default(primals_298, add_262);  primals_298 = add_262 = copy__148 = None
        copy__149: "f32[256]" = torch.ops.aten.copy_.default(primals_299, add_263);  primals_299 = add_263 = copy__149 = None
        copy__150: "i64[]" = torch.ops.aten.copy_.default(primals_303, add_265);  primals_303 = add_265 = copy__150 = None
        copy__151: "f32[256]" = torch.ops.aten.copy_.default(primals_304, add_267);  primals_304 = add_267 = copy__151 = None
        copy__152: "f32[256]" = torch.ops.aten.copy_.default(primals_305, add_268);  primals_305 = add_268 = copy__152 = None
        copy__153: "i64[]" = torch.ops.aten.copy_.default(primals_309, add_270);  primals_309 = add_270 = copy__153 = None
        copy__154: "f32[1024]" = torch.ops.aten.copy_.default(primals_310, add_272);  primals_310 = add_272 = copy__154 = None
        copy__155: "f32[1024]" = torch.ops.aten.copy_.default(primals_311, add_273);  primals_311 = add_273 = copy__155 = None
        copy__156: "i64[]" = torch.ops.aten.copy_.default(primals_315, add_276);  primals_315 = add_276 = copy__156 = None
        copy__157: "f32[256]" = torch.ops.aten.copy_.default(primals_316, add_278);  primals_316 = add_278 = copy__157 = None
        copy__158: "f32[256]" = torch.ops.aten.copy_.default(primals_317, add_279);  primals_317 = add_279 = copy__158 = None
        copy__159: "i64[]" = torch.ops.aten.copy_.default(primals_321, add_281);  primals_321 = add_281 = copy__159 = None
        copy__160: "f32[256]" = torch.ops.aten.copy_.default(primals_322, add_283);  primals_322 = add_283 = copy__160 = None
        copy__161: "f32[256]" = torch.ops.aten.copy_.default(primals_323, add_284);  primals_323 = add_284 = copy__161 = None
        copy__162: "i64[]" = torch.ops.aten.copy_.default(primals_327, add_286);  primals_327 = add_286 = copy__162 = None
        copy__163: "f32[1024]" = torch.ops.aten.copy_.default(primals_328, add_288);  primals_328 = add_288 = copy__163 = None
        copy__164: "f32[1024]" = torch.ops.aten.copy_.default(primals_329, add_289);  primals_329 = add_289 = copy__164 = None
        copy__165: "i64[]" = torch.ops.aten.copy_.default(primals_333, add_292);  primals_333 = add_292 = copy__165 = None
        copy__166: "f32[256]" = torch.ops.aten.copy_.default(primals_334, add_294);  primals_334 = add_294 = copy__166 = None
        copy__167: "f32[256]" = torch.ops.aten.copy_.default(primals_335, add_295);  primals_335 = add_295 = copy__167 = None
        copy__168: "i64[]" = torch.ops.aten.copy_.default(primals_339, add_297);  primals_339 = add_297 = copy__168 = None
        copy__169: "f32[256]" = torch.ops.aten.copy_.default(primals_340, add_299);  primals_340 = add_299 = copy__169 = None
        copy__170: "f32[256]" = torch.ops.aten.copy_.default(primals_341, add_300);  primals_341 = add_300 = copy__170 = None
        copy__171: "i64[]" = torch.ops.aten.copy_.default(primals_345, add_302);  primals_345 = add_302 = copy__171 = None
        copy__172: "f32[1024]" = torch.ops.aten.copy_.default(primals_346, add_304);  primals_346 = add_304 = copy__172 = None
        copy__173: "f32[1024]" = torch.ops.aten.copy_.default(primals_347, add_305);  primals_347 = add_305 = copy__173 = None
        copy__174: "i64[]" = torch.ops.aten.copy_.default(primals_351, add_308);  primals_351 = add_308 = copy__174 = None
        copy__175: "f32[256]" = torch.ops.aten.copy_.default(primals_352, add_310);  primals_352 = add_310 = copy__175 = None
        copy__176: "f32[256]" = torch.ops.aten.copy_.default(primals_353, add_311);  primals_353 = add_311 = copy__176 = None
        copy__177: "i64[]" = torch.ops.aten.copy_.default(primals_357, add_313);  primals_357 = add_313 = copy__177 = None
        copy__178: "f32[256]" = torch.ops.aten.copy_.default(primals_358, add_315);  primals_358 = add_315 = copy__178 = None
        copy__179: "f32[256]" = torch.ops.aten.copy_.default(primals_359, add_316);  primals_359 = add_316 = copy__179 = None
        copy__180: "i64[]" = torch.ops.aten.copy_.default(primals_363, add_318);  primals_363 = add_318 = copy__180 = None
        copy__181: "f32[1024]" = torch.ops.aten.copy_.default(primals_364, add_320);  primals_364 = add_320 = copy__181 = None
        copy__182: "f32[1024]" = torch.ops.aten.copy_.default(primals_365, add_321);  primals_365 = add_321 = copy__182 = None
        copy__183: "i64[]" = torch.ops.aten.copy_.default(primals_369, add_324);  primals_369 = add_324 = copy__183 = None
        copy__184: "f32[256]" = torch.ops.aten.copy_.default(primals_370, add_326);  primals_370 = add_326 = copy__184 = None
        copy__185: "f32[256]" = torch.ops.aten.copy_.default(primals_371, add_327);  primals_371 = add_327 = copy__185 = None
        copy__186: "i64[]" = torch.ops.aten.copy_.default(primals_375, add_329);  primals_375 = add_329 = copy__186 = None
        copy__187: "f32[256]" = torch.ops.aten.copy_.default(primals_376, add_331);  primals_376 = add_331 = copy__187 = None
        copy__188: "f32[256]" = torch.ops.aten.copy_.default(primals_377, add_332);  primals_377 = add_332 = copy__188 = None
        copy__189: "i64[]" = torch.ops.aten.copy_.default(primals_381, add_334);  primals_381 = add_334 = copy__189 = None
        copy__190: "f32[1024]" = torch.ops.aten.copy_.default(primals_382, add_336);  primals_382 = add_336 = copy__190 = None
        copy__191: "f32[1024]" = torch.ops.aten.copy_.default(primals_383, add_337);  primals_383 = add_337 = copy__191 = None
        copy__192: "i64[]" = torch.ops.aten.copy_.default(primals_387, add_340);  primals_387 = add_340 = copy__192 = None
        copy__193: "f32[256]" = torch.ops.aten.copy_.default(primals_388, add_342);  primals_388 = add_342 = copy__193 = None
        copy__194: "f32[256]" = torch.ops.aten.copy_.default(primals_389, add_343);  primals_389 = add_343 = copy__194 = None
        copy__195: "i64[]" = torch.ops.aten.copy_.default(primals_393, add_345);  primals_393 = add_345 = copy__195 = None
        copy__196: "f32[256]" = torch.ops.aten.copy_.default(primals_394, add_347);  primals_394 = add_347 = copy__196 = None
        copy__197: "f32[256]" = torch.ops.aten.copy_.default(primals_395, add_348);  primals_395 = add_348 = copy__197 = None
        copy__198: "i64[]" = torch.ops.aten.copy_.default(primals_399, add_350);  primals_399 = add_350 = copy__198 = None
        copy__199: "f32[1024]" = torch.ops.aten.copy_.default(primals_400, add_352);  primals_400 = add_352 = copy__199 = None
        copy__200: "f32[1024]" = torch.ops.aten.copy_.default(primals_401, add_353);  primals_401 = add_353 = copy__200 = None
        copy__201: "i64[]" = torch.ops.aten.copy_.default(primals_405, add_356);  primals_405 = add_356 = copy__201 = None
        copy__202: "f32[256]" = torch.ops.aten.copy_.default(primals_406, add_358);  primals_406 = add_358 = copy__202 = None
        copy__203: "f32[256]" = torch.ops.aten.copy_.default(primals_407, add_359);  primals_407 = add_359 = copy__203 = None
        copy__204: "i64[]" = torch.ops.aten.copy_.default(primals_411, add_361);  primals_411 = add_361 = copy__204 = None
        copy__205: "f32[256]" = torch.ops.aten.copy_.default(primals_412, add_363);  primals_412 = add_363 = copy__205 = None
        copy__206: "f32[256]" = torch.ops.aten.copy_.default(primals_413, add_364);  primals_413 = add_364 = copy__206 = None
        copy__207: "i64[]" = torch.ops.aten.copy_.default(primals_417, add_366);  primals_417 = add_366 = copy__207 = None
        copy__208: "f32[1024]" = torch.ops.aten.copy_.default(primals_418, add_368);  primals_418 = add_368 = copy__208 = None
        copy__209: "f32[1024]" = torch.ops.aten.copy_.default(primals_419, add_369);  primals_419 = add_369 = copy__209 = None
        copy__210: "i64[]" = torch.ops.aten.copy_.default(primals_423, add_372);  primals_423 = add_372 = copy__210 = None
        copy__211: "f32[256]" = torch.ops.aten.copy_.default(primals_424, add_374);  primals_424 = add_374 = copy__211 = None
        copy__212: "f32[256]" = torch.ops.aten.copy_.default(primals_425, add_375);  primals_425 = add_375 = copy__212 = None
        copy__213: "i64[]" = torch.ops.aten.copy_.default(primals_429, add_377);  primals_429 = add_377 = copy__213 = None
        copy__214: "f32[256]" = torch.ops.aten.copy_.default(primals_430, add_379);  primals_430 = add_379 = copy__214 = None
        copy__215: "f32[256]" = torch.ops.aten.copy_.default(primals_431, add_380);  primals_431 = add_380 = copy__215 = None
        copy__216: "i64[]" = torch.ops.aten.copy_.default(primals_435, add_382);  primals_435 = add_382 = copy__216 = None
        copy__217: "f32[1024]" = torch.ops.aten.copy_.default(primals_436, add_384);  primals_436 = add_384 = copy__217 = None
        copy__218: "f32[1024]" = torch.ops.aten.copy_.default(primals_437, add_385);  primals_437 = add_385 = copy__218 = None
        copy__219: "i64[]" = torch.ops.aten.copy_.default(primals_441, add_388);  primals_441 = add_388 = copy__219 = None
        copy__220: "f32[256]" = torch.ops.aten.copy_.default(primals_442, add_390);  primals_442 = add_390 = copy__220 = None
        copy__221: "f32[256]" = torch.ops.aten.copy_.default(primals_443, add_391);  primals_443 = add_391 = copy__221 = None
        copy__222: "i64[]" = torch.ops.aten.copy_.default(primals_447, add_393);  primals_447 = add_393 = copy__222 = None
        copy__223: "f32[256]" = torch.ops.aten.copy_.default(primals_448, add_395);  primals_448 = add_395 = copy__223 = None
        copy__224: "f32[256]" = torch.ops.aten.copy_.default(primals_449, add_396);  primals_449 = add_396 = copy__224 = None
        copy__225: "i64[]" = torch.ops.aten.copy_.default(primals_453, add_398);  primals_453 = add_398 = copy__225 = None
        copy__226: "f32[1024]" = torch.ops.aten.copy_.default(primals_454, add_400);  primals_454 = add_400 = copy__226 = None
        copy__227: "f32[1024]" = torch.ops.aten.copy_.default(primals_455, add_401);  primals_455 = add_401 = copy__227 = None
        copy__228: "i64[]" = torch.ops.aten.copy_.default(primals_459, add_404);  primals_459 = add_404 = copy__228 = None
        copy__229: "f32[256]" = torch.ops.aten.copy_.default(primals_460, add_406);  primals_460 = add_406 = copy__229 = None
        copy__230: "f32[256]" = torch.ops.aten.copy_.default(primals_461, add_407);  primals_461 = add_407 = copy__230 = None
        copy__231: "i64[]" = torch.ops.aten.copy_.default(primals_465, add_409);  primals_465 = add_409 = copy__231 = None
        copy__232: "f32[256]" = torch.ops.aten.copy_.default(primals_466, add_411);  primals_466 = add_411 = copy__232 = None
        copy__233: "f32[256]" = torch.ops.aten.copy_.default(primals_467, add_412);  primals_467 = add_412 = copy__233 = None
        copy__234: "i64[]" = torch.ops.aten.copy_.default(primals_471, add_414);  primals_471 = add_414 = copy__234 = None
        copy__235: "f32[1024]" = torch.ops.aten.copy_.default(primals_472, add_416);  primals_472 = add_416 = copy__235 = None
        copy__236: "f32[1024]" = torch.ops.aten.copy_.default(primals_473, add_417);  primals_473 = add_417 = copy__236 = None
        copy__237: "i64[]" = torch.ops.aten.copy_.default(primals_477, add_420);  primals_477 = add_420 = copy__237 = None
        copy__238: "f32[256]" = torch.ops.aten.copy_.default(primals_478, add_422);  primals_478 = add_422 = copy__238 = None
        copy__239: "f32[256]" = torch.ops.aten.copy_.default(primals_479, add_423);  primals_479 = add_423 = copy__239 = None
        copy__240: "i64[]" = torch.ops.aten.copy_.default(primals_483, add_425);  primals_483 = add_425 = copy__240 = None
        copy__241: "f32[256]" = torch.ops.aten.copy_.default(primals_484, add_427);  primals_484 = add_427 = copy__241 = None
        copy__242: "f32[256]" = torch.ops.aten.copy_.default(primals_485, add_428);  primals_485 = add_428 = copy__242 = None
        copy__243: "i64[]" = torch.ops.aten.copy_.default(primals_489, add_430);  primals_489 = add_430 = copy__243 = None
        copy__244: "f32[1024]" = torch.ops.aten.copy_.default(primals_490, add_432);  primals_490 = add_432 = copy__244 = None
        copy__245: "f32[1024]" = torch.ops.aten.copy_.default(primals_491, add_433);  primals_491 = add_433 = copy__245 = None
        copy__246: "i64[]" = torch.ops.aten.copy_.default(primals_495, add_436);  primals_495 = add_436 = copy__246 = None
        copy__247: "f32[256]" = torch.ops.aten.copy_.default(primals_496, add_438);  primals_496 = add_438 = copy__247 = None
        copy__248: "f32[256]" = torch.ops.aten.copy_.default(primals_497, add_439);  primals_497 = add_439 = copy__248 = None
        copy__249: "i64[]" = torch.ops.aten.copy_.default(primals_501, add_441);  primals_501 = add_441 = copy__249 = None
        copy__250: "f32[256]" = torch.ops.aten.copy_.default(primals_502, add_443);  primals_502 = add_443 = copy__250 = None
        copy__251: "f32[256]" = torch.ops.aten.copy_.default(primals_503, add_444);  primals_503 = add_444 = copy__251 = None
        copy__252: "i64[]" = torch.ops.aten.copy_.default(primals_507, add_446);  primals_507 = add_446 = copy__252 = None
        copy__253: "f32[1024]" = torch.ops.aten.copy_.default(primals_508, add_448);  primals_508 = add_448 = copy__253 = None
        copy__254: "f32[1024]" = torch.ops.aten.copy_.default(primals_509, add_449);  primals_509 = add_449 = copy__254 = None
        copy__255: "i64[]" = torch.ops.aten.copy_.default(primals_513, add_452);  primals_513 = add_452 = copy__255 = None
        copy__256: "f32[256]" = torch.ops.aten.copy_.default(primals_514, add_454);  primals_514 = add_454 = copy__256 = None
        copy__257: "f32[256]" = torch.ops.aten.copy_.default(primals_515, add_455);  primals_515 = add_455 = copy__257 = None
        copy__258: "i64[]" = torch.ops.aten.copy_.default(primals_519, add_457);  primals_519 = add_457 = copy__258 = None
        copy__259: "f32[256]" = torch.ops.aten.copy_.default(primals_520, add_459);  primals_520 = add_459 = copy__259 = None
        copy__260: "f32[256]" = torch.ops.aten.copy_.default(primals_521, add_460);  primals_521 = add_460 = copy__260 = None
        copy__261: "i64[]" = torch.ops.aten.copy_.default(primals_525, add_462);  primals_525 = add_462 = copy__261 = None
        copy__262: "f32[1024]" = torch.ops.aten.copy_.default(primals_526, add_464);  primals_526 = add_464 = copy__262 = None
        copy__263: "f32[1024]" = torch.ops.aten.copy_.default(primals_527, add_465);  primals_527 = add_465 = copy__263 = None
        copy__264: "i64[]" = torch.ops.aten.copy_.default(primals_531, add_468);  primals_531 = add_468 = copy__264 = None
        copy__265: "f32[256]" = torch.ops.aten.copy_.default(primals_532, add_470);  primals_532 = add_470 = copy__265 = None
        copy__266: "f32[256]" = torch.ops.aten.copy_.default(primals_533, add_471);  primals_533 = add_471 = copy__266 = None
        copy__267: "i64[]" = torch.ops.aten.copy_.default(primals_537, add_473);  primals_537 = add_473 = copy__267 = None
        copy__268: "f32[256]" = torch.ops.aten.copy_.default(primals_538, add_475);  primals_538 = add_475 = copy__268 = None
        copy__269: "f32[256]" = torch.ops.aten.copy_.default(primals_539, add_476);  primals_539 = add_476 = copy__269 = None
        copy__270: "i64[]" = torch.ops.aten.copy_.default(primals_543, add_478);  primals_543 = add_478 = copy__270 = None
        copy__271: "f32[1024]" = torch.ops.aten.copy_.default(primals_544, add_480);  primals_544 = add_480 = copy__271 = None
        copy__272: "f32[1024]" = torch.ops.aten.copy_.default(primals_545, add_481);  primals_545 = add_481 = copy__272 = None
        copy__273: "i64[]" = torch.ops.aten.copy_.default(primals_549, add_484);  primals_549 = add_484 = copy__273 = None
        copy__274: "f32[256]" = torch.ops.aten.copy_.default(primals_550, add_486);  primals_550 = add_486 = copy__274 = None
        copy__275: "f32[256]" = torch.ops.aten.copy_.default(primals_551, add_487);  primals_551 = add_487 = copy__275 = None
        copy__276: "i64[]" = torch.ops.aten.copy_.default(primals_555, add_489);  primals_555 = add_489 = copy__276 = None
        copy__277: "f32[256]" = torch.ops.aten.copy_.default(primals_556, add_491);  primals_556 = add_491 = copy__277 = None
        copy__278: "f32[256]" = torch.ops.aten.copy_.default(primals_557, add_492);  primals_557 = add_492 = copy__278 = None
        copy__279: "i64[]" = torch.ops.aten.copy_.default(primals_561, add_494);  primals_561 = add_494 = copy__279 = None
        copy__280: "f32[1024]" = torch.ops.aten.copy_.default(primals_562, add_496);  primals_562 = add_496 = copy__280 = None
        copy__281: "f32[1024]" = torch.ops.aten.copy_.default(primals_563, add_497);  primals_563 = add_497 = copy__281 = None
        copy__282: "i64[]" = torch.ops.aten.copy_.default(primals_567, add_500);  primals_567 = add_500 = copy__282 = None
        copy__283: "f32[256]" = torch.ops.aten.copy_.default(primals_568, add_502);  primals_568 = add_502 = copy__283 = None
        copy__284: "f32[256]" = torch.ops.aten.copy_.default(primals_569, add_503);  primals_569 = add_503 = copy__284 = None
        copy__285: "i64[]" = torch.ops.aten.copy_.default(primals_573, add_505);  primals_573 = add_505 = copy__285 = None
        copy__286: "f32[256]" = torch.ops.aten.copy_.default(primals_574, add_507);  primals_574 = add_507 = copy__286 = None
        copy__287: "f32[256]" = torch.ops.aten.copy_.default(primals_575, add_508);  primals_575 = add_508 = copy__287 = None
        copy__288: "i64[]" = torch.ops.aten.copy_.default(primals_579, add_510);  primals_579 = add_510 = copy__288 = None
        copy__289: "f32[1024]" = torch.ops.aten.copy_.default(primals_580, add_512);  primals_580 = add_512 = copy__289 = None
        copy__290: "f32[1024]" = torch.ops.aten.copy_.default(primals_581, add_513);  primals_581 = add_513 = copy__290 = None
        copy__291: "i64[]" = torch.ops.aten.copy_.default(primals_585, add_516);  primals_585 = add_516 = copy__291 = None
        copy__292: "f32[256]" = torch.ops.aten.copy_.default(primals_586, add_518);  primals_586 = add_518 = copy__292 = None
        copy__293: "f32[256]" = torch.ops.aten.copy_.default(primals_587, add_519);  primals_587 = add_519 = copy__293 = None
        copy__294: "i64[]" = torch.ops.aten.copy_.default(primals_591, add_521);  primals_591 = add_521 = copy__294 = None
        copy__295: "f32[256]" = torch.ops.aten.copy_.default(primals_592, add_523);  primals_592 = add_523 = copy__295 = None
        copy__296: "f32[256]" = torch.ops.aten.copy_.default(primals_593, add_524);  primals_593 = add_524 = copy__296 = None
        copy__297: "i64[]" = torch.ops.aten.copy_.default(primals_597, add_526);  primals_597 = add_526 = copy__297 = None
        copy__298: "f32[1024]" = torch.ops.aten.copy_.default(primals_598, add_528);  primals_598 = add_528 = copy__298 = None
        copy__299: "f32[1024]" = torch.ops.aten.copy_.default(primals_599, add_529);  primals_599 = add_529 = copy__299 = None
        copy__300: "i64[]" = torch.ops.aten.copy_.default(primals_603, add_532);  primals_603 = add_532 = copy__300 = None
        copy__301: "f32[256]" = torch.ops.aten.copy_.default(primals_604, add_534);  primals_604 = add_534 = copy__301 = None
        copy__302: "f32[256]" = torch.ops.aten.copy_.default(primals_605, add_535);  primals_605 = add_535 = copy__302 = None
        copy__303: "i64[]" = torch.ops.aten.copy_.default(primals_609, add_537);  primals_609 = add_537 = copy__303 = None
        copy__304: "f32[256]" = torch.ops.aten.copy_.default(primals_610, add_539);  primals_610 = add_539 = copy__304 = None
        copy__305: "f32[256]" = torch.ops.aten.copy_.default(primals_611, add_540);  primals_611 = add_540 = copy__305 = None
        copy__306: "i64[]" = torch.ops.aten.copy_.default(primals_615, add_542);  primals_615 = add_542 = copy__306 = None
        copy__307: "f32[1024]" = torch.ops.aten.copy_.default(primals_616, add_544);  primals_616 = add_544 = copy__307 = None
        copy__308: "f32[1024]" = torch.ops.aten.copy_.default(primals_617, add_545);  primals_617 = add_545 = copy__308 = None
        copy__309: "i64[]" = torch.ops.aten.copy_.default(primals_621, add_548);  primals_621 = add_548 = copy__309 = None
        copy__310: "f32[256]" = torch.ops.aten.copy_.default(primals_622, add_550);  primals_622 = add_550 = copy__310 = None
        copy__311: "f32[256]" = torch.ops.aten.copy_.default(primals_623, add_551);  primals_623 = add_551 = copy__311 = None
        copy__312: "i64[]" = torch.ops.aten.copy_.default(primals_627, add_553);  primals_627 = add_553 = copy__312 = None
        copy__313: "f32[256]" = torch.ops.aten.copy_.default(primals_628, add_555);  primals_628 = add_555 = copy__313 = None
        copy__314: "f32[256]" = torch.ops.aten.copy_.default(primals_629, add_556);  primals_629 = add_556 = copy__314 = None
        copy__315: "i64[]" = torch.ops.aten.copy_.default(primals_633, add_558);  primals_633 = add_558 = copy__315 = None
        copy__316: "f32[1024]" = torch.ops.aten.copy_.default(primals_634, add_560);  primals_634 = add_560 = copy__316 = None
        copy__317: "f32[1024]" = torch.ops.aten.copy_.default(primals_635, add_561);  primals_635 = add_561 = copy__317 = None
        copy__318: "i64[]" = torch.ops.aten.copy_.default(primals_639, add_564);  primals_639 = add_564 = copy__318 = None
        copy__319: "f32[256]" = torch.ops.aten.copy_.default(primals_640, add_566);  primals_640 = add_566 = copy__319 = None
        copy__320: "f32[256]" = torch.ops.aten.copy_.default(primals_641, add_567);  primals_641 = add_567 = copy__320 = None
        copy__321: "i64[]" = torch.ops.aten.copy_.default(primals_645, add_569);  primals_645 = add_569 = copy__321 = None
        copy__322: "f32[256]" = torch.ops.aten.copy_.default(primals_646, add_571);  primals_646 = add_571 = copy__322 = None
        copy__323: "f32[256]" = torch.ops.aten.copy_.default(primals_647, add_572);  primals_647 = add_572 = copy__323 = None
        copy__324: "i64[]" = torch.ops.aten.copy_.default(primals_651, add_574);  primals_651 = add_574 = copy__324 = None
        copy__325: "f32[1024]" = torch.ops.aten.copy_.default(primals_652, add_576);  primals_652 = add_576 = copy__325 = None
        copy__326: "f32[1024]" = torch.ops.aten.copy_.default(primals_653, add_577);  primals_653 = add_577 = copy__326 = None
        copy__327: "i64[]" = torch.ops.aten.copy_.default(primals_657, add_580);  primals_657 = add_580 = copy__327 = None
        copy__328: "f32[256]" = torch.ops.aten.copy_.default(primals_658, add_582);  primals_658 = add_582 = copy__328 = None
        copy__329: "f32[256]" = torch.ops.aten.copy_.default(primals_659, add_583);  primals_659 = add_583 = copy__329 = None
        copy__330: "i64[]" = torch.ops.aten.copy_.default(primals_663, add_585);  primals_663 = add_585 = copy__330 = None
        copy__331: "f32[256]" = torch.ops.aten.copy_.default(primals_664, add_587);  primals_664 = add_587 = copy__331 = None
        copy__332: "f32[256]" = torch.ops.aten.copy_.default(primals_665, add_588);  primals_665 = add_588 = copy__332 = None
        copy__333: "i64[]" = torch.ops.aten.copy_.default(primals_669, add_590);  primals_669 = add_590 = copy__333 = None
        copy__334: "f32[1024]" = torch.ops.aten.copy_.default(primals_670, add_592);  primals_670 = add_592 = copy__334 = None
        copy__335: "f32[1024]" = torch.ops.aten.copy_.default(primals_671, add_593);  primals_671 = add_593 = copy__335 = None
        copy__336: "i64[]" = torch.ops.aten.copy_.default(primals_675, add_596);  primals_675 = add_596 = copy__336 = None
        copy__337: "f32[256]" = torch.ops.aten.copy_.default(primals_676, add_598);  primals_676 = add_598 = copy__337 = None
        copy__338: "f32[256]" = torch.ops.aten.copy_.default(primals_677, add_599);  primals_677 = add_599 = copy__338 = None
        copy__339: "i64[]" = torch.ops.aten.copy_.default(primals_681, add_601);  primals_681 = add_601 = copy__339 = None
        copy__340: "f32[256]" = torch.ops.aten.copy_.default(primals_682, add_603);  primals_682 = add_603 = copy__340 = None
        copy__341: "f32[256]" = torch.ops.aten.copy_.default(primals_683, add_604);  primals_683 = add_604 = copy__341 = None
        copy__342: "i64[]" = torch.ops.aten.copy_.default(primals_687, add_606);  primals_687 = add_606 = copy__342 = None
        copy__343: "f32[1024]" = torch.ops.aten.copy_.default(primals_688, add_608);  primals_688 = add_608 = copy__343 = None
        copy__344: "f32[1024]" = torch.ops.aten.copy_.default(primals_689, add_609);  primals_689 = add_609 = copy__344 = None
        copy__345: "i64[]" = torch.ops.aten.copy_.default(primals_693, add_612);  primals_693 = add_612 = copy__345 = None
        copy__346: "f32[256]" = torch.ops.aten.copy_.default(primals_694, add_614);  primals_694 = add_614 = copy__346 = None
        copy__347: "f32[256]" = torch.ops.aten.copy_.default(primals_695, add_615);  primals_695 = add_615 = copy__347 = None
        copy__348: "i64[]" = torch.ops.aten.copy_.default(primals_699, add_617);  primals_699 = add_617 = copy__348 = None
        copy__349: "f32[256]" = torch.ops.aten.copy_.default(primals_700, add_619);  primals_700 = add_619 = copy__349 = None
        copy__350: "f32[256]" = torch.ops.aten.copy_.default(primals_701, add_620);  primals_701 = add_620 = copy__350 = None
        copy__351: "i64[]" = torch.ops.aten.copy_.default(primals_705, add_622);  primals_705 = add_622 = copy__351 = None
        copy__352: "f32[1024]" = torch.ops.aten.copy_.default(primals_706, add_624);  primals_706 = add_624 = copy__352 = None
        copy__353: "f32[1024]" = torch.ops.aten.copy_.default(primals_707, add_625);  primals_707 = add_625 = copy__353 = None
        copy__354: "i64[]" = torch.ops.aten.copy_.default(primals_711, add_628);  primals_711 = add_628 = copy__354 = None
        copy__355: "f32[256]" = torch.ops.aten.copy_.default(primals_712, add_630);  primals_712 = add_630 = copy__355 = None
        copy__356: "f32[256]" = torch.ops.aten.copy_.default(primals_713, add_631);  primals_713 = add_631 = copy__356 = None
        copy__357: "i64[]" = torch.ops.aten.copy_.default(primals_717, add_633);  primals_717 = add_633 = copy__357 = None
        copy__358: "f32[256]" = torch.ops.aten.copy_.default(primals_718, add_635);  primals_718 = add_635 = copy__358 = None
        copy__359: "f32[256]" = torch.ops.aten.copy_.default(primals_719, add_636);  primals_719 = add_636 = copy__359 = None
        copy__360: "i64[]" = torch.ops.aten.copy_.default(primals_723, add_638);  primals_723 = add_638 = copy__360 = None
        copy__361: "f32[1024]" = torch.ops.aten.copy_.default(primals_724, add_640);  primals_724 = add_640 = copy__361 = None
        copy__362: "f32[1024]" = torch.ops.aten.copy_.default(primals_725, add_641);  primals_725 = add_641 = copy__362 = None
        copy__363: "i64[]" = torch.ops.aten.copy_.default(primals_729, add_644);  primals_729 = add_644 = copy__363 = None
        copy__364: "f32[256]" = torch.ops.aten.copy_.default(primals_730, add_646);  primals_730 = add_646 = copy__364 = None
        copy__365: "f32[256]" = torch.ops.aten.copy_.default(primals_731, add_647);  primals_731 = add_647 = copy__365 = None
        copy__366: "i64[]" = torch.ops.aten.copy_.default(primals_735, add_649);  primals_735 = add_649 = copy__366 = None
        copy__367: "f32[256]" = torch.ops.aten.copy_.default(primals_736, add_651);  primals_736 = add_651 = copy__367 = None
        copy__368: "f32[256]" = torch.ops.aten.copy_.default(primals_737, add_652);  primals_737 = add_652 = copy__368 = None
        copy__369: "i64[]" = torch.ops.aten.copy_.default(primals_741, add_654);  primals_741 = add_654 = copy__369 = None
        copy__370: "f32[1024]" = torch.ops.aten.copy_.default(primals_742, add_656);  primals_742 = add_656 = copy__370 = None
        copy__371: "f32[1024]" = torch.ops.aten.copy_.default(primals_743, add_657);  primals_743 = add_657 = copy__371 = None
        copy__372: "i64[]" = torch.ops.aten.copy_.default(primals_747, add_660);  primals_747 = add_660 = copy__372 = None
        copy__373: "f32[256]" = torch.ops.aten.copy_.default(primals_748, add_662);  primals_748 = add_662 = copy__373 = None
        copy__374: "f32[256]" = torch.ops.aten.copy_.default(primals_749, add_663);  primals_749 = add_663 = copy__374 = None
        copy__375: "i64[]" = torch.ops.aten.copy_.default(primals_753, add_665);  primals_753 = add_665 = copy__375 = None
        copy__376: "f32[256]" = torch.ops.aten.copy_.default(primals_754, add_667);  primals_754 = add_667 = copy__376 = None
        copy__377: "f32[256]" = torch.ops.aten.copy_.default(primals_755, add_668);  primals_755 = add_668 = copy__377 = None
        copy__378: "i64[]" = torch.ops.aten.copy_.default(primals_759, add_670);  primals_759 = add_670 = copy__378 = None
        copy__379: "f32[1024]" = torch.ops.aten.copy_.default(primals_760, add_672);  primals_760 = add_672 = copy__379 = None
        copy__380: "f32[1024]" = torch.ops.aten.copy_.default(primals_761, add_673);  primals_761 = add_673 = copy__380 = None
        copy__381: "i64[]" = torch.ops.aten.copy_.default(primals_765, add_676);  primals_765 = add_676 = copy__381 = None
        copy__382: "f32[256]" = torch.ops.aten.copy_.default(primals_766, add_678);  primals_766 = add_678 = copy__382 = None
        copy__383: "f32[256]" = torch.ops.aten.copy_.default(primals_767, add_679);  primals_767 = add_679 = copy__383 = None
        copy__384: "i64[]" = torch.ops.aten.copy_.default(primals_771, add_681);  primals_771 = add_681 = copy__384 = None
        copy__385: "f32[256]" = torch.ops.aten.copy_.default(primals_772, add_683);  primals_772 = add_683 = copy__385 = None
        copy__386: "f32[256]" = torch.ops.aten.copy_.default(primals_773, add_684);  primals_773 = add_684 = copy__386 = None
        copy__387: "i64[]" = torch.ops.aten.copy_.default(primals_777, add_686);  primals_777 = add_686 = copy__387 = None
        copy__388: "f32[1024]" = torch.ops.aten.copy_.default(primals_778, add_688);  primals_778 = add_688 = copy__388 = None
        copy__389: "f32[1024]" = torch.ops.aten.copy_.default(primals_779, add_689);  primals_779 = add_689 = copy__389 = None
        copy__390: "i64[]" = torch.ops.aten.copy_.default(primals_783, add_692);  primals_783 = add_692 = copy__390 = None
        copy__391: "f32[256]" = torch.ops.aten.copy_.default(primals_784, add_694);  primals_784 = add_694 = copy__391 = None
        copy__392: "f32[256]" = torch.ops.aten.copy_.default(primals_785, add_695);  primals_785 = add_695 = copy__392 = None
        copy__393: "i64[]" = torch.ops.aten.copy_.default(primals_789, add_697);  primals_789 = add_697 = copy__393 = None
        copy__394: "f32[256]" = torch.ops.aten.copy_.default(primals_790, add_699);  primals_790 = add_699 = copy__394 = None
        copy__395: "f32[256]" = torch.ops.aten.copy_.default(primals_791, add_700);  primals_791 = add_700 = copy__395 = None
        copy__396: "i64[]" = torch.ops.aten.copy_.default(primals_795, add_702);  primals_795 = add_702 = copy__396 = None
        copy__397: "f32[1024]" = torch.ops.aten.copy_.default(primals_796, add_704);  primals_796 = add_704 = copy__397 = None
        copy__398: "f32[1024]" = torch.ops.aten.copy_.default(primals_797, add_705);  primals_797 = add_705 = copy__398 = None
        copy__399: "i64[]" = torch.ops.aten.copy_.default(primals_801, add_708);  primals_801 = add_708 = copy__399 = None
        copy__400: "f32[256]" = torch.ops.aten.copy_.default(primals_802, add_710);  primals_802 = add_710 = copy__400 = None
        copy__401: "f32[256]" = torch.ops.aten.copy_.default(primals_803, add_711);  primals_803 = add_711 = copy__401 = None
        copy__402: "i64[]" = torch.ops.aten.copy_.default(primals_807, add_713);  primals_807 = add_713 = copy__402 = None
        copy__403: "f32[256]" = torch.ops.aten.copy_.default(primals_808, add_715);  primals_808 = add_715 = copy__403 = None
        copy__404: "f32[256]" = torch.ops.aten.copy_.default(primals_809, add_716);  primals_809 = add_716 = copy__404 = None
        copy__405: "i64[]" = torch.ops.aten.copy_.default(primals_813, add_718);  primals_813 = add_718 = copy__405 = None
        copy__406: "f32[1024]" = torch.ops.aten.copy_.default(primals_814, add_720);  primals_814 = add_720 = copy__406 = None
        copy__407: "f32[1024]" = torch.ops.aten.copy_.default(primals_815, add_721);  primals_815 = add_721 = copy__407 = None
        copy__408: "i64[]" = torch.ops.aten.copy_.default(primals_819, add_724);  primals_819 = add_724 = copy__408 = None
        copy__409: "f32[256]" = torch.ops.aten.copy_.default(primals_820, add_726);  primals_820 = add_726 = copy__409 = None
        copy__410: "f32[256]" = torch.ops.aten.copy_.default(primals_821, add_727);  primals_821 = add_727 = copy__410 = None
        copy__411: "i64[]" = torch.ops.aten.copy_.default(primals_825, add_729);  primals_825 = add_729 = copy__411 = None
        copy__412: "f32[256]" = torch.ops.aten.copy_.default(primals_826, add_731);  primals_826 = add_731 = copy__412 = None
        copy__413: "f32[256]" = torch.ops.aten.copy_.default(primals_827, add_732);  primals_827 = add_732 = copy__413 = None
        copy__414: "i64[]" = torch.ops.aten.copy_.default(primals_831, add_734);  primals_831 = add_734 = copy__414 = None
        copy__415: "f32[1024]" = torch.ops.aten.copy_.default(primals_832, add_736);  primals_832 = add_736 = copy__415 = None
        copy__416: "f32[1024]" = torch.ops.aten.copy_.default(primals_833, add_737);  primals_833 = add_737 = copy__416 = None
        copy__417: "i64[]" = torch.ops.aten.copy_.default(primals_837, add_740);  primals_837 = add_740 = copy__417 = None
        copy__418: "f32[256]" = torch.ops.aten.copy_.default(primals_838, add_742);  primals_838 = add_742 = copy__418 = None
        copy__419: "f32[256]" = torch.ops.aten.copy_.default(primals_839, add_743);  primals_839 = add_743 = copy__419 = None
        copy__420: "i64[]" = torch.ops.aten.copy_.default(primals_843, add_745);  primals_843 = add_745 = copy__420 = None
        copy__421: "f32[256]" = torch.ops.aten.copy_.default(primals_844, add_747);  primals_844 = add_747 = copy__421 = None
        copy__422: "f32[256]" = torch.ops.aten.copy_.default(primals_845, add_748);  primals_845 = add_748 = copy__422 = None
        copy__423: "i64[]" = torch.ops.aten.copy_.default(primals_849, add_750);  primals_849 = add_750 = copy__423 = None
        copy__424: "f32[1024]" = torch.ops.aten.copy_.default(primals_850, add_752);  primals_850 = add_752 = copy__424 = None
        copy__425: "f32[1024]" = torch.ops.aten.copy_.default(primals_851, add_753);  primals_851 = add_753 = copy__425 = None
        copy__426: "i64[]" = torch.ops.aten.copy_.default(primals_855, add_756);  primals_855 = add_756 = copy__426 = None
        copy__427: "f32[256]" = torch.ops.aten.copy_.default(primals_856, add_758);  primals_856 = add_758 = copy__427 = None
        copy__428: "f32[256]" = torch.ops.aten.copy_.default(primals_857, add_759);  primals_857 = add_759 = copy__428 = None
        copy__429: "i64[]" = torch.ops.aten.copy_.default(primals_861, add_761);  primals_861 = add_761 = copy__429 = None
        copy__430: "f32[256]" = torch.ops.aten.copy_.default(primals_862, add_763);  primals_862 = add_763 = copy__430 = None
        copy__431: "f32[256]" = torch.ops.aten.copy_.default(primals_863, add_764);  primals_863 = add_764 = copy__431 = None
        copy__432: "i64[]" = torch.ops.aten.copy_.default(primals_867, add_766);  primals_867 = add_766 = copy__432 = None
        copy__433: "f32[1024]" = torch.ops.aten.copy_.default(primals_868, add_768);  primals_868 = add_768 = copy__433 = None
        copy__434: "f32[1024]" = torch.ops.aten.copy_.default(primals_869, add_769);  primals_869 = add_769 = copy__434 = None
        copy__435: "i64[]" = torch.ops.aten.copy_.default(primals_873, add_772);  primals_873 = add_772 = copy__435 = None
        copy__436: "f32[512]" = torch.ops.aten.copy_.default(primals_874, add_774);  primals_874 = add_774 = copy__436 = None
        copy__437: "f32[512]" = torch.ops.aten.copy_.default(primals_875, add_775);  primals_875 = add_775 = copy__437 = None
        copy__438: "i64[]" = torch.ops.aten.copy_.default(primals_879, add_777);  primals_879 = add_777 = copy__438 = None
        copy__439: "f32[512]" = torch.ops.aten.copy_.default(primals_880, add_779);  primals_880 = add_779 = copy__439 = None
        copy__440: "f32[512]" = torch.ops.aten.copy_.default(primals_881, add_780);  primals_881 = add_780 = copy__440 = None
        copy__441: "i64[]" = torch.ops.aten.copy_.default(primals_885, add_782);  primals_885 = add_782 = copy__441 = None
        copy__442: "f32[2048]" = torch.ops.aten.copy_.default(primals_886, add_784);  primals_886 = add_784 = copy__442 = None
        copy__443: "f32[2048]" = torch.ops.aten.copy_.default(primals_887, add_785);  primals_887 = add_785 = copy__443 = None
        copy__444: "i64[]" = torch.ops.aten.copy_.default(primals_891, add_787);  primals_891 = add_787 = copy__444 = None
        copy__445: "f32[2048]" = torch.ops.aten.copy_.default(primals_892, add_789);  primals_892 = add_789 = copy__445 = None
        copy__446: "f32[2048]" = torch.ops.aten.copy_.default(primals_893, add_790);  primals_893 = add_790 = copy__446 = None
        copy__447: "i64[]" = torch.ops.aten.copy_.default(primals_897, add_793);  primals_897 = add_793 = copy__447 = None
        copy__448: "f32[512]" = torch.ops.aten.copy_.default(primals_898, add_795);  primals_898 = add_795 = copy__448 = None
        copy__449: "f32[512]" = torch.ops.aten.copy_.default(primals_899, add_796);  primals_899 = add_796 = copy__449 = None
        copy__450: "i64[]" = torch.ops.aten.copy_.default(primals_903, add_798);  primals_903 = add_798 = copy__450 = None
        copy__451: "f32[512]" = torch.ops.aten.copy_.default(primals_904, add_800);  primals_904 = add_800 = copy__451 = None
        copy__452: "f32[512]" = torch.ops.aten.copy_.default(primals_905, add_801);  primals_905 = add_801 = copy__452 = None
        copy__453: "i64[]" = torch.ops.aten.copy_.default(primals_909, add_803);  primals_909 = add_803 = copy__453 = None
        copy__454: "f32[2048]" = torch.ops.aten.copy_.default(primals_910, add_805);  primals_910 = add_805 = copy__454 = None
        copy__455: "f32[2048]" = torch.ops.aten.copy_.default(primals_911, add_806);  primals_911 = add_806 = copy__455 = None
        copy__456: "i64[]" = torch.ops.aten.copy_.default(primals_915, add_809);  primals_915 = add_809 = copy__456 = None
        copy__457: "f32[512]" = torch.ops.aten.copy_.default(primals_916, add_811);  primals_916 = add_811 = copy__457 = None
        copy__458: "f32[512]" = torch.ops.aten.copy_.default(primals_917, add_812);  primals_917 = add_812 = copy__458 = None
        copy__459: "i64[]" = torch.ops.aten.copy_.default(primals_921, add_814);  primals_921 = add_814 = copy__459 = None
        copy__460: "f32[512]" = torch.ops.aten.copy_.default(primals_922, add_816);  primals_922 = add_816 = copy__460 = None
        copy__461: "f32[512]" = torch.ops.aten.copy_.default(primals_923, add_817);  primals_923 = add_817 = copy__461 = None
        copy__462: "i64[]" = torch.ops.aten.copy_.default(primals_927, add_819);  primals_927 = add_819 = copy__462 = None
        copy__463: "f32[2048]" = torch.ops.aten.copy_.default(primals_928, add_821);  primals_928 = add_821 = copy__463 = None
        copy__464: "f32[2048]" = torch.ops.aten.copy_.default(primals_929, add_822);  primals_929 = add_822 = copy__464 = None
        return (addmm, primals_1, primals_2, primals_6, primals_7, primals_8, primals_12, primals_14, primals_18, primals_20, primals_24, primals_26, primals_30, primals_32, primals_36, primals_38, primals_42, primals_44, primals_48, primals_50, primals_54, primals_56, primals_60, primals_62, primals_66, primals_68, primals_72, primals_74, primals_78, primals_80, primals_84, primals_86, primals_90, primals_92, primals_96, primals_98, primals_102, primals_104, primals_108, primals_110, primals_114, primals_116, primals_120, primals_122, primals_126, primals_128, primals_132, primals_134, primals_138, primals_140, primals_144, primals_146, primals_150, primals_152, primals_156, primals_158, primals_162, primals_164, primals_168, primals_170, primals_174, primals_176, primals_180, primals_182, primals_186, primals_188, primals_192, primals_194, primals_198, primals_200, primals_204, primals_206, primals_210, primals_212, primals_216, primals_218, primals_222, primals_224, primals_228, primals_230, primals_234, primals_236, primals_240, primals_242, primals_246, primals_248, primals_252, primals_254, primals_258, primals_260, primals_264, primals_266, primals_270, primals_272, primals_276, primals_278, primals_282, primals_284, primals_288, primals_290, primals_294, primals_296, primals_300, primals_302, primals_306, primals_308, primals_312, primals_314, primals_318, primals_320, primals_324, primals_326, primals_330, primals_332, primals_336, primals_338, primals_342, primals_344, primals_348, primals_350, primals_354, primals_356, primals_360, primals_362, primals_366, primals_368, primals_372, primals_374, primals_378, primals_380, primals_384, primals_386, primals_390, primals_392, primals_396, primals_398, primals_402, primals_404, primals_408, primals_410, primals_414, primals_416, primals_420, primals_422, primals_426, primals_428, primals_432, primals_434, primals_438, primals_440, primals_444, primals_446, primals_450, primals_452, primals_456, primals_458, primals_462, primals_464, primals_468, primals_470, primals_474, primals_476, primals_480, primals_482, primals_486, primals_488, primals_492, primals_494, primals_498, primals_500, primals_504, primals_506, primals_510, primals_512, primals_516, primals_518, primals_522, primals_524, primals_528, primals_530, primals_534, primals_536, primals_540, primals_542, primals_546, primals_548, primals_552, primals_554, primals_558, primals_560, primals_564, primals_566, primals_570, primals_572, primals_576, primals_578, primals_582, primals_584, primals_588, primals_590, primals_594, primals_596, primals_600, primals_602, primals_606, primals_608, primals_612, primals_614, primals_618, primals_620, primals_624, primals_626, primals_630, primals_632, primals_636, primals_638, primals_642, primals_644, primals_648, primals_650, primals_654, primals_656, primals_660, primals_662, primals_666, primals_668, primals_672, primals_674, primals_678, primals_680, primals_684, primals_686, primals_690, primals_692, primals_696, primals_698, primals_702, primals_704, primals_708, primals_710, primals_714, primals_716, primals_720, primals_722, primals_726, primals_728, primals_732, primals_734, primals_738, primals_740, primals_744, primals_746, primals_750, primals_752, primals_756, primals_758, primals_762, primals_764, primals_768, primals_770, primals_774, primals_776, primals_780, primals_782, primals_786, primals_788, primals_792, primals_794, primals_798, primals_800, primals_804, primals_806, primals_810, primals_812, primals_816, primals_818, primals_822, primals_824, primals_828, primals_830, primals_834, primals_836, primals_840, primals_842, primals_846, primals_848, primals_852, primals_854, primals_858, primals_860, primals_864, primals_866, primals_870, primals_872, primals_876, primals_878, primals_882, primals_884, primals_888, primals_890, primals_894, primals_896, primals_900, primals_902, primals_906, primals_908, primals_912, primals_914, primals_918, primals_920, primals_924, primals_926, primals_930, primals_932, convolution, getitem_1, rsqrt, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, relu_4, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, squeeze_31, relu_9, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, relu_11, convolution_13, squeeze_40, convolution_14, squeeze_43, relu_12, convolution_15, squeeze_46, relu_13, convolution_16, squeeze_49, relu_14, convolution_17, squeeze_52, relu_15, convolution_18, squeeze_55, relu_16, convolution_19, squeeze_58, relu_17, convolution_20, squeeze_61, relu_18, convolution_21, squeeze_64, relu_19, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, relu_21, convolution_24, squeeze_73, relu_22, convolution_25, squeeze_76, relu_23, convolution_26, squeeze_79, relu_24, convolution_27, squeeze_82, relu_25, convolution_28, squeeze_85, relu_26, convolution_29, squeeze_88, relu_27, convolution_30, squeeze_91, relu_28, convolution_31, squeeze_94, relu_29, convolution_32, squeeze_97, relu_30, convolution_33, squeeze_100, relu_31, convolution_34, squeeze_103, relu_32, convolution_35, squeeze_106, relu_33, convolution_36, squeeze_109, relu_34, convolution_37, squeeze_112, relu_35, convolution_38, squeeze_115, convolution_39, squeeze_118, relu_36, convolution_40, squeeze_121, relu_37, convolution_41, squeeze_124, relu_38, convolution_42, squeeze_127, relu_39, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, relu_41, convolution_45, squeeze_136, relu_42, convolution_46, squeeze_139, relu_43, convolution_47, squeeze_142, relu_44, convolution_48, squeeze_145, relu_45, convolution_49, squeeze_148, relu_46, convolution_50, squeeze_151, relu_47, convolution_51, squeeze_154, relu_48, convolution_52, squeeze_157, relu_49, convolution_53, squeeze_160, relu_50, convolution_54, squeeze_163, relu_51, convolution_55, squeeze_166, relu_52, convolution_56, squeeze_169, relu_53, convolution_57, squeeze_172, relu_54, convolution_58, squeeze_175, relu_55, convolution_59, squeeze_178, relu_56, convolution_60, squeeze_181, relu_57, convolution_61, squeeze_184, relu_58, convolution_62, squeeze_187, relu_59, convolution_63, squeeze_190, relu_60, convolution_64, squeeze_193, relu_61, convolution_65, squeeze_196, relu_62, convolution_66, squeeze_199, relu_63, convolution_67, squeeze_202, relu_64, convolution_68, squeeze_205, relu_65, convolution_69, squeeze_208, relu_66, convolution_70, squeeze_211, relu_67, convolution_71, squeeze_214, relu_68, convolution_72, squeeze_217, relu_69, convolution_73, squeeze_220, relu_70, convolution_74, squeeze_223, relu_71, convolution_75, squeeze_226, relu_72, convolution_76, squeeze_229, relu_73, convolution_77, squeeze_232, relu_74, convolution_78, squeeze_235, relu_75, convolution_79, squeeze_238, relu_76, convolution_80, squeeze_241, relu_77, convolution_81, squeeze_244, relu_78, convolution_82, squeeze_247, relu_79, convolution_83, squeeze_250, relu_80, convolution_84, squeeze_253, relu_81, convolution_85, squeeze_256, relu_82, convolution_86, squeeze_259, relu_83, convolution_87, squeeze_262, relu_84, convolution_88, squeeze_265, relu_85, convolution_89, squeeze_268, relu_86, convolution_90, squeeze_271, relu_87, convolution_91, squeeze_274, relu_88, convolution_92, squeeze_277, relu_89, convolution_93, squeeze_280, relu_90, convolution_94, squeeze_283, relu_91, convolution_95, squeeze_286, relu_92, convolution_96, squeeze_289, relu_93, convolution_97, squeeze_292, relu_94, convolution_98, squeeze_295, relu_95, convolution_99, squeeze_298, relu_96, convolution_100, squeeze_301, relu_97, convolution_101, squeeze_304, relu_98, convolution_102, squeeze_307, relu_99, convolution_103, squeeze_310, relu_100, convolution_104, squeeze_313, relu_101, convolution_105, squeeze_316, relu_102, convolution_106, squeeze_319, relu_103, convolution_107, squeeze_322, relu_104, convolution_108, squeeze_325, relu_105, convolution_109, squeeze_328, relu_106, convolution_110, squeeze_331, relu_107, convolution_111, squeeze_334, relu_108, convolution_112, squeeze_337, relu_109, convolution_113, squeeze_340, relu_110, convolution_114, squeeze_343, relu_111, convolution_115, squeeze_346, relu_112, convolution_116, squeeze_349, relu_113, convolution_117, squeeze_352, relu_114, convolution_118, squeeze_355, relu_115, convolution_119, squeeze_358, relu_116, convolution_120, squeeze_361, relu_117, convolution_121, squeeze_364, relu_118, convolution_122, squeeze_367, relu_119, convolution_123, squeeze_370, relu_120, convolution_124, squeeze_373, relu_121, convolution_125, squeeze_376, relu_122, convolution_126, squeeze_379, relu_123, convolution_127, squeeze_382, relu_124, convolution_128, squeeze_385, relu_125, convolution_129, squeeze_388, relu_126, convolution_130, squeeze_391, relu_127, convolution_131, squeeze_394, relu_128, convolution_132, squeeze_397, relu_129, convolution_133, squeeze_400, relu_130, convolution_134, squeeze_403, relu_131, convolution_135, squeeze_406, relu_132, convolution_136, squeeze_409, relu_133, convolution_137, squeeze_412, relu_134, convolution_138, squeeze_415, relu_135, convolution_139, squeeze_418, relu_136, convolution_140, squeeze_421, relu_137, convolution_141, squeeze_424, relu_138, convolution_142, squeeze_427, relu_139, convolution_143, squeeze_430, relu_140, convolution_144, squeeze_433, relu_141, convolution_145, squeeze_436, relu_142, convolution_146, squeeze_439, relu_143, convolution_147, squeeze_442, convolution_148, squeeze_445, relu_144, convolution_149, squeeze_448, relu_145, convolution_150, squeeze_451, relu_146, convolution_151, squeeze_454, relu_147, convolution_152, squeeze_457, relu_148, convolution_153, squeeze_460, relu_149, convolution_154, squeeze_463, view, le, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, unsqueeze_850, unsqueeze_862, unsqueeze_874, unsqueeze_886, unsqueeze_898, unsqueeze_910, unsqueeze_922, unsqueeze_934, unsqueeze_946, unsqueeze_958, unsqueeze_970, unsqueeze_982, unsqueeze_994, unsqueeze_1006, unsqueeze_1018, unsqueeze_1030, unsqueeze_1042, unsqueeze_1054, unsqueeze_1066, unsqueeze_1078, unsqueeze_1090, unsqueeze_1102, unsqueeze_1114, unsqueeze_1126, unsqueeze_1138, unsqueeze_1150, unsqueeze_1162, unsqueeze_1174, unsqueeze_1186, unsqueeze_1198, unsqueeze_1210, unsqueeze_1222, unsqueeze_1234, unsqueeze_1246, unsqueeze_1258, unsqueeze_1270, unsqueeze_1282, unsqueeze_1294, unsqueeze_1306, unsqueeze_1318, unsqueeze_1330, unsqueeze_1342, unsqueeze_1354, unsqueeze_1366, unsqueeze_1378, unsqueeze_1390, unsqueeze_1402, unsqueeze_1414, unsqueeze_1426, unsqueeze_1438, unsqueeze_1450, unsqueeze_1462, unsqueeze_1474, unsqueeze_1486, unsqueeze_1498, unsqueeze_1510, unsqueeze_1522, unsqueeze_1534, unsqueeze_1546, unsqueeze_1558, unsqueeze_1570, unsqueeze_1582, unsqueeze_1594, unsqueeze_1606, unsqueeze_1618, unsqueeze_1630, unsqueeze_1642, unsqueeze_1654, unsqueeze_1666, unsqueeze_1678, unsqueeze_1690, unsqueeze_1702, unsqueeze_1714, unsqueeze_1726, unsqueeze_1738, unsqueeze_1750, unsqueeze_1762, unsqueeze_1774, unsqueeze_1786, unsqueeze_1798, unsqueeze_1810, unsqueeze_1822, unsqueeze_1834, unsqueeze_1846, unsqueeze_1858, unsqueeze_1870, unsqueeze_1882, unsqueeze_1894, unsqueeze_1906, unsqueeze_1918, unsqueeze_1930, unsqueeze_1942, unsqueeze_1954, unsqueeze_1966, unsqueeze_1978, unsqueeze_1990, unsqueeze_2002, unsqueeze_2014, unsqueeze_2026, unsqueeze_2038, unsqueeze_2050, unsqueeze_2062, unsqueeze_2074, unsqueeze_2086, unsqueeze_2098, unsqueeze_2110, unsqueeze_2122, unsqueeze_2134, unsqueeze_2146, unsqueeze_2158, unsqueeze_2170, unsqueeze_2182, unsqueeze_2194, unsqueeze_2206, unsqueeze_2218, unsqueeze_2230, unsqueeze_2242, unsqueeze_2254, unsqueeze_2266, unsqueeze_2278, unsqueeze_2290, unsqueeze_2302, unsqueeze_2314, unsqueeze_2326, unsqueeze_2338, unsqueeze_2350, unsqueeze_2362, unsqueeze_2374, unsqueeze_2386, unsqueeze_2398, unsqueeze_2410, unsqueeze_2422, unsqueeze_2434, unsqueeze_2446, unsqueeze_2458)
