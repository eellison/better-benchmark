import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[256, 128]", primals_2: "i64[1, 512]", primals_3: "f32[30522, 128]", primals_4: "f32[512, 384]", primals_5: "f32[512]", primals_6: "f32[512, 512]", primals_7: "f32[2, 512]", primals_8: "f32[512]", primals_9: "f32[512]", primals_10: "f32[128, 512]", primals_11: "f32[128]", primals_12: "f32[128]", primals_13: "f32[128]", primals_14: "f32[128, 512]", primals_15: "f32[128]", primals_16: "f32[128]", primals_17: "f32[128]", primals_18: "f32[128, 128]", primals_19: "f32[128]", primals_20: "f32[128, 128]", primals_21: "f32[128]", primals_22: "f32[128, 512]", primals_23: "f32[128]", primals_24: "f32[128, 128]", primals_25: "f32[128]", primals_26: "f32[128]", primals_27: "f32[128]", primals_28: "f32[512, 128]", primals_29: "f32[512]", primals_30: "f32[128, 512]", primals_31: "f32[128]", primals_32: "f32[128]", primals_33: "f32[128]", primals_34: "f32[512, 128]", primals_35: "f32[512]", primals_36: "f32[128, 512]", primals_37: "f32[128]", primals_38: "f32[128]", primals_39: "f32[128]", primals_40: "f32[512, 128]", primals_41: "f32[512]", primals_42: "f32[128, 512]", primals_43: "f32[128]", primals_44: "f32[128]", primals_45: "f32[128]", primals_46: "f32[512, 128]", primals_47: "f32[512]", primals_48: "f32[128, 512]", primals_49: "f32[128]", primals_50: "f32[128]", primals_51: "f32[128]", primals_52: "f32[512, 128]", primals_53: "f32[512]", primals_54: "f32[512]", primals_55: "f32[512]", primals_56: "f32[128, 512]", primals_57: "f32[128]", primals_58: "f32[128]", primals_59: "f32[128]", primals_60: "f32[128, 512]", primals_61: "f32[128]", primals_62: "f32[128]", primals_63: "f32[128]", primals_64: "f32[128, 128]", primals_65: "f32[128]", primals_66: "f32[128, 128]", primals_67: "f32[128]", primals_68: "f32[128, 512]", primals_69: "f32[128]", primals_70: "f32[128, 128]", primals_71: "f32[128]", primals_72: "f32[128]", primals_73: "f32[128]", primals_74: "f32[512, 128]", primals_75: "f32[512]", primals_76: "f32[128, 512]", primals_77: "f32[128]", primals_78: "f32[128]", primals_79: "f32[128]", primals_80: "f32[512, 128]", primals_81: "f32[512]", primals_82: "f32[128, 512]", primals_83: "f32[128]", primals_84: "f32[128]", primals_85: "f32[128]", primals_86: "f32[512, 128]", primals_87: "f32[512]", primals_88: "f32[128, 512]", primals_89: "f32[128]", primals_90: "f32[128]", primals_91: "f32[128]", primals_92: "f32[512, 128]", primals_93: "f32[512]", primals_94: "f32[128, 512]", primals_95: "f32[128]", primals_96: "f32[128]", primals_97: "f32[128]", primals_98: "f32[512, 128]", primals_99: "f32[512]", primals_100: "f32[512]", primals_101: "f32[512]", primals_102: "f32[128, 512]", primals_103: "f32[128]", primals_104: "f32[128]", primals_105: "f32[128]", primals_106: "f32[128, 512]", primals_107: "f32[128]", primals_108: "f32[128]", primals_109: "f32[128]", primals_110: "f32[128, 128]", primals_111: "f32[128]", primals_112: "f32[128, 128]", primals_113: "f32[128]", primals_114: "f32[128, 512]", primals_115: "f32[128]", primals_116: "f32[128, 128]", primals_117: "f32[128]", primals_118: "f32[128]", primals_119: "f32[128]", primals_120: "f32[512, 128]", primals_121: "f32[512]", primals_122: "f32[128, 512]", primals_123: "f32[128]", primals_124: "f32[128]", primals_125: "f32[128]", primals_126: "f32[512, 128]", primals_127: "f32[512]", primals_128: "f32[128, 512]", primals_129: "f32[128]", primals_130: "f32[128]", primals_131: "f32[128]", primals_132: "f32[512, 128]", primals_133: "f32[512]", primals_134: "f32[128, 512]", primals_135: "f32[128]", primals_136: "f32[128]", primals_137: "f32[128]", primals_138: "f32[512, 128]", primals_139: "f32[512]", primals_140: "f32[128, 512]", primals_141: "f32[128]", primals_142: "f32[128]", primals_143: "f32[128]", primals_144: "f32[512, 128]", primals_145: "f32[512]", primals_146: "f32[512]", primals_147: "f32[512]", primals_148: "f32[128, 512]", primals_149: "f32[128]", primals_150: "f32[128]", primals_151: "f32[128]", primals_152: "f32[128, 512]", primals_153: "f32[128]", primals_154: "f32[128]", primals_155: "f32[128]", primals_156: "f32[128, 128]", primals_157: "f32[128]", primals_158: "f32[128, 128]", primals_159: "f32[128]", primals_160: "f32[128, 512]", primals_161: "f32[128]", primals_162: "f32[128, 128]", primals_163: "f32[128]", primals_164: "f32[128]", primals_165: "f32[128]", primals_166: "f32[512, 128]", primals_167: "f32[512]", primals_168: "f32[128, 512]", primals_169: "f32[128]", primals_170: "f32[128]", primals_171: "f32[128]", primals_172: "f32[512, 128]", primals_173: "f32[512]", primals_174: "f32[128, 512]", primals_175: "f32[128]", primals_176: "f32[128]", primals_177: "f32[128]", primals_178: "f32[512, 128]", primals_179: "f32[512]", primals_180: "f32[128, 512]", primals_181: "f32[128]", primals_182: "f32[128]", primals_183: "f32[128]", primals_184: "f32[512, 128]", primals_185: "f32[512]", primals_186: "f32[128, 512]", primals_187: "f32[128]", primals_188: "f32[128]", primals_189: "f32[128]", primals_190: "f32[512, 128]", primals_191: "f32[512]", primals_192: "f32[512]", primals_193: "f32[512]", primals_194: "f32[128, 512]", primals_195: "f32[128]", primals_196: "f32[128]", primals_197: "f32[128]", primals_198: "f32[128, 512]", primals_199: "f32[128]", primals_200: "f32[128]", primals_201: "f32[128]", primals_202: "f32[128, 128]", primals_203: "f32[128]", primals_204: "f32[128, 128]", primals_205: "f32[128]", primals_206: "f32[128, 512]", primals_207: "f32[128]", primals_208: "f32[128, 128]", primals_209: "f32[128]", primals_210: "f32[128]", primals_211: "f32[128]", primals_212: "f32[512, 128]", primals_213: "f32[512]", primals_214: "f32[128, 512]", primals_215: "f32[128]", primals_216: "f32[128]", primals_217: "f32[128]", primals_218: "f32[512, 128]", primals_219: "f32[512]", primals_220: "f32[128, 512]", primals_221: "f32[128]", primals_222: "f32[128]", primals_223: "f32[128]", primals_224: "f32[512, 128]", primals_225: "f32[512]", primals_226: "f32[128, 512]", primals_227: "f32[128]", primals_228: "f32[128]", primals_229: "f32[128]", primals_230: "f32[512, 128]", primals_231: "f32[512]", primals_232: "f32[128, 512]", primals_233: "f32[128]", primals_234: "f32[128]", primals_235: "f32[128]", primals_236: "f32[512, 128]", primals_237: "f32[512]", primals_238: "f32[512]", primals_239: "f32[512]", primals_240: "f32[128, 512]", primals_241: "f32[128]", primals_242: "f32[128]", primals_243: "f32[128]", primals_244: "f32[128, 512]", primals_245: "f32[128]", primals_246: "f32[128]", primals_247: "f32[128]", primals_248: "f32[128, 128]", primals_249: "f32[128]", primals_250: "f32[128, 128]", primals_251: "f32[128]", primals_252: "f32[128, 512]", primals_253: "f32[128]", primals_254: "f32[128, 128]", primals_255: "f32[128]", primals_256: "f32[128]", primals_257: "f32[128]", primals_258: "f32[512, 128]", primals_259: "f32[512]", primals_260: "f32[128, 512]", primals_261: "f32[128]", primals_262: "f32[128]", primals_263: "f32[128]", primals_264: "f32[512, 128]", primals_265: "f32[512]", primals_266: "f32[128, 512]", primals_267: "f32[128]", primals_268: "f32[128]", primals_269: "f32[128]", primals_270: "f32[512, 128]", primals_271: "f32[512]", primals_272: "f32[128, 512]", primals_273: "f32[128]", primals_274: "f32[128]", primals_275: "f32[128]", primals_276: "f32[512, 128]", primals_277: "f32[512]", primals_278: "f32[128, 512]", primals_279: "f32[128]", primals_280: "f32[128]", primals_281: "f32[128]", primals_282: "f32[512, 128]", primals_283: "f32[512]", primals_284: "f32[512]", primals_285: "f32[512]", primals_286: "f32[128, 512]", primals_287: "f32[128]", primals_288: "f32[128]", primals_289: "f32[128]", primals_290: "f32[128, 512]", primals_291: "f32[128]", primals_292: "f32[128]", primals_293: "f32[128]", primals_294: "f32[128, 128]", primals_295: "f32[128]", primals_296: "f32[128, 128]", primals_297: "f32[128]", primals_298: "f32[128, 512]", primals_299: "f32[128]", primals_300: "f32[128, 128]", primals_301: "f32[128]", primals_302: "f32[128]", primals_303: "f32[128]", primals_304: "f32[512, 128]", primals_305: "f32[512]", primals_306: "f32[128, 512]", primals_307: "f32[128]", primals_308: "f32[128]", primals_309: "f32[128]", primals_310: "f32[512, 128]", primals_311: "f32[512]", primals_312: "f32[128, 512]", primals_313: "f32[128]", primals_314: "f32[128]", primals_315: "f32[128]", primals_316: "f32[512, 128]", primals_317: "f32[512]", primals_318: "f32[128, 512]", primals_319: "f32[128]", primals_320: "f32[128]", primals_321: "f32[128]", primals_322: "f32[512, 128]", primals_323: "f32[512]", primals_324: "f32[128, 512]", primals_325: "f32[128]", primals_326: "f32[128]", primals_327: "f32[128]", primals_328: "f32[512, 128]", primals_329: "f32[512]", primals_330: "f32[512]", primals_331: "f32[512]", primals_332: "f32[128, 512]", primals_333: "f32[128]", primals_334: "f32[128]", primals_335: "f32[128]", primals_336: "f32[128, 512]", primals_337: "f32[128]", primals_338: "f32[128]", primals_339: "f32[128]", primals_340: "f32[128, 128]", primals_341: "f32[128]", primals_342: "f32[128, 128]", primals_343: "f32[128]", primals_344: "f32[128, 512]", primals_345: "f32[128]", primals_346: "f32[128, 128]", primals_347: "f32[128]", primals_348: "f32[128]", primals_349: "f32[128]", primals_350: "f32[512, 128]", primals_351: "f32[512]", primals_352: "f32[128, 512]", primals_353: "f32[128]", primals_354: "f32[128]", primals_355: "f32[128]", primals_356: "f32[512, 128]", primals_357: "f32[512]", primals_358: "f32[128, 512]", primals_359: "f32[128]", primals_360: "f32[128]", primals_361: "f32[128]", primals_362: "f32[512, 128]", primals_363: "f32[512]", primals_364: "f32[128, 512]", primals_365: "f32[128]", primals_366: "f32[128]", primals_367: "f32[128]", primals_368: "f32[512, 128]", primals_369: "f32[512]", primals_370: "f32[128, 512]", primals_371: "f32[128]", primals_372: "f32[128]", primals_373: "f32[128]", primals_374: "f32[512, 128]", primals_375: "f32[512]", primals_376: "f32[512]", primals_377: "f32[512]", primals_378: "f32[128, 512]", primals_379: "f32[128]", primals_380: "f32[128]", primals_381: "f32[128]", primals_382: "f32[128, 512]", primals_383: "f32[128]", primals_384: "f32[128]", primals_385: "f32[128]", primals_386: "f32[128, 128]", primals_387: "f32[128]", primals_388: "f32[128, 128]", primals_389: "f32[128]", primals_390: "f32[128, 512]", primals_391: "f32[128]", primals_392: "f32[128, 128]", primals_393: "f32[128]", primals_394: "f32[128]", primals_395: "f32[128]", primals_396: "f32[512, 128]", primals_397: "f32[512]", primals_398: "f32[128, 512]", primals_399: "f32[128]", primals_400: "f32[128]", primals_401: "f32[128]", primals_402: "f32[512, 128]", primals_403: "f32[512]", primals_404: "f32[128, 512]", primals_405: "f32[128]", primals_406: "f32[128]", primals_407: "f32[128]", primals_408: "f32[512, 128]", primals_409: "f32[512]", primals_410: "f32[128, 512]", primals_411: "f32[128]", primals_412: "f32[128]", primals_413: "f32[128]", primals_414: "f32[512, 128]", primals_415: "f32[512]", primals_416: "f32[128, 512]", primals_417: "f32[128]", primals_418: "f32[128]", primals_419: "f32[128]", primals_420: "f32[512, 128]", primals_421: "f32[512]", primals_422: "f32[512]", primals_423: "f32[512]", primals_424: "f32[128, 512]", primals_425: "f32[128]", primals_426: "f32[128]", primals_427: "f32[128]", primals_428: "f32[128, 512]", primals_429: "f32[128]", primals_430: "f32[128]", primals_431: "f32[128]", primals_432: "f32[128, 128]", primals_433: "f32[128]", primals_434: "f32[128, 128]", primals_435: "f32[128]", primals_436: "f32[128, 512]", primals_437: "f32[128]", primals_438: "f32[128, 128]", primals_439: "f32[128]", primals_440: "f32[128]", primals_441: "f32[128]", primals_442: "f32[512, 128]", primals_443: "f32[512]", primals_444: "f32[128, 512]", primals_445: "f32[128]", primals_446: "f32[128]", primals_447: "f32[128]", primals_448: "f32[512, 128]", primals_449: "f32[512]", primals_450: "f32[128, 512]", primals_451: "f32[128]", primals_452: "f32[128]", primals_453: "f32[128]", primals_454: "f32[512, 128]", primals_455: "f32[512]", primals_456: "f32[128, 512]", primals_457: "f32[128]", primals_458: "f32[128]", primals_459: "f32[128]", primals_460: "f32[512, 128]", primals_461: "f32[512]", primals_462: "f32[128, 512]", primals_463: "f32[128]", primals_464: "f32[128]", primals_465: "f32[128]", primals_466: "f32[512, 128]", primals_467: "f32[512]", primals_468: "f32[512]", primals_469: "f32[512]", primals_470: "f32[128, 512]", primals_471: "f32[128]", primals_472: "f32[128]", primals_473: "f32[128]", primals_474: "f32[128, 512]", primals_475: "f32[128]", primals_476: "f32[128]", primals_477: "f32[128]", primals_478: "f32[128, 128]", primals_479: "f32[128]", primals_480: "f32[128, 128]", primals_481: "f32[128]", primals_482: "f32[128, 512]", primals_483: "f32[128]", primals_484: "f32[128, 128]", primals_485: "f32[128]", primals_486: "f32[128]", primals_487: "f32[128]", primals_488: "f32[512, 128]", primals_489: "f32[512]", primals_490: "f32[128, 512]", primals_491: "f32[128]", primals_492: "f32[128]", primals_493: "f32[128]", primals_494: "f32[512, 128]", primals_495: "f32[512]", primals_496: "f32[128, 512]", primals_497: "f32[128]", primals_498: "f32[128]", primals_499: "f32[128]", primals_500: "f32[512, 128]", primals_501: "f32[512]", primals_502: "f32[128, 512]", primals_503: "f32[128]", primals_504: "f32[128]", primals_505: "f32[128]", primals_506: "f32[512, 128]", primals_507: "f32[512]", primals_508: "f32[128, 512]", primals_509: "f32[128]", primals_510: "f32[128]", primals_511: "f32[128]", primals_512: "f32[512, 128]", primals_513: "f32[512]", primals_514: "f32[512]", primals_515: "f32[512]", primals_516: "f32[128, 512]", primals_517: "f32[128]", primals_518: "f32[128]", primals_519: "f32[128]", primals_520: "f32[128, 512]", primals_521: "f32[128]", primals_522: "f32[128]", primals_523: "f32[128]", primals_524: "f32[128, 128]", primals_525: "f32[128]", primals_526: "f32[128, 128]", primals_527: "f32[128]", primals_528: "f32[128, 512]", primals_529: "f32[128]", primals_530: "f32[128, 128]", primals_531: "f32[128]", primals_532: "f32[128]", primals_533: "f32[128]", primals_534: "f32[512, 128]", primals_535: "f32[512]", primals_536: "f32[128, 512]", primals_537: "f32[128]", primals_538: "f32[128]", primals_539: "f32[128]", primals_540: "f32[512, 128]", primals_541: "f32[512]", primals_542: "f32[128, 512]", primals_543: "f32[128]", primals_544: "f32[128]", primals_545: "f32[128]", primals_546: "f32[512, 128]", primals_547: "f32[512]", primals_548: "f32[128, 512]", primals_549: "f32[128]", primals_550: "f32[128]", primals_551: "f32[128]", primals_552: "f32[512, 128]", primals_553: "f32[512]", primals_554: "f32[128, 512]", primals_555: "f32[128]", primals_556: "f32[128]", primals_557: "f32[128]", primals_558: "f32[512, 128]", primals_559: "f32[512]", primals_560: "f32[512]", primals_561: "f32[512]", primals_562: "f32[128, 512]", primals_563: "f32[128]", primals_564: "f32[128]", primals_565: "f32[128]", primals_566: "f32[128, 512]", primals_567: "f32[128]", primals_568: "f32[128]", primals_569: "f32[128]", primals_570: "f32[128, 128]", primals_571: "f32[128]", primals_572: "f32[128, 128]", primals_573: "f32[128]", primals_574: "f32[128, 512]", primals_575: "f32[128]", primals_576: "f32[128, 128]", primals_577: "f32[128]", primals_578: "f32[128]", primals_579: "f32[128]", primals_580: "f32[512, 128]", primals_581: "f32[512]", primals_582: "f32[128, 512]", primals_583: "f32[128]", primals_584: "f32[128]", primals_585: "f32[128]", primals_586: "f32[512, 128]", primals_587: "f32[512]", primals_588: "f32[128, 512]", primals_589: "f32[128]", primals_590: "f32[128]", primals_591: "f32[128]", primals_592: "f32[512, 128]", primals_593: "f32[512]", primals_594: "f32[128, 512]", primals_595: "f32[128]", primals_596: "f32[128]", primals_597: "f32[128]", primals_598: "f32[512, 128]", primals_599: "f32[512]", primals_600: "f32[128, 512]", primals_601: "f32[128]", primals_602: "f32[128]", primals_603: "f32[128]", primals_604: "f32[512, 128]", primals_605: "f32[512]", primals_606: "f32[512]", primals_607: "f32[512]", primals_608: "f32[128, 512]", primals_609: "f32[128]", primals_610: "f32[128]", primals_611: "f32[128]", primals_612: "f32[128, 512]", primals_613: "f32[128]", primals_614: "f32[128]", primals_615: "f32[128]", primals_616: "f32[128, 128]", primals_617: "f32[128]", primals_618: "f32[128, 128]", primals_619: "f32[128]", primals_620: "f32[128, 512]", primals_621: "f32[128]", primals_622: "f32[128, 128]", primals_623: "f32[128]", primals_624: "f32[128]", primals_625: "f32[128]", primals_626: "f32[512, 128]", primals_627: "f32[512]", primals_628: "f32[128, 512]", primals_629: "f32[128]", primals_630: "f32[128]", primals_631: "f32[128]", primals_632: "f32[512, 128]", primals_633: "f32[512]", primals_634: "f32[128, 512]", primals_635: "f32[128]", primals_636: "f32[128]", primals_637: "f32[128]", primals_638: "f32[512, 128]", primals_639: "f32[512]", primals_640: "f32[128, 512]", primals_641: "f32[128]", primals_642: "f32[128]", primals_643: "f32[128]", primals_644: "f32[512, 128]", primals_645: "f32[512]", primals_646: "f32[128, 512]", primals_647: "f32[128]", primals_648: "f32[128]", primals_649: "f32[128]", primals_650: "f32[512, 128]", primals_651: "f32[512]", primals_652: "f32[512]", primals_653: "f32[512]", primals_654: "f32[128, 512]", primals_655: "f32[128]", primals_656: "f32[128]", primals_657: "f32[128]", primals_658: "f32[128, 512]", primals_659: "f32[128]", primals_660: "f32[128]", primals_661: "f32[128]", primals_662: "f32[128, 128]", primals_663: "f32[128]", primals_664: "f32[128, 128]", primals_665: "f32[128]", primals_666: "f32[128, 512]", primals_667: "f32[128]", primals_668: "f32[128, 128]", primals_669: "f32[128]", primals_670: "f32[128]", primals_671: "f32[128]", primals_672: "f32[512, 128]", primals_673: "f32[512]", primals_674: "f32[128, 512]", primals_675: "f32[128]", primals_676: "f32[128]", primals_677: "f32[128]", primals_678: "f32[512, 128]", primals_679: "f32[512]", primals_680: "f32[128, 512]", primals_681: "f32[128]", primals_682: "f32[128]", primals_683: "f32[128]", primals_684: "f32[512, 128]", primals_685: "f32[512]", primals_686: "f32[128, 512]", primals_687: "f32[128]", primals_688: "f32[128]", primals_689: "f32[128]", primals_690: "f32[512, 128]", primals_691: "f32[512]", primals_692: "f32[128, 512]", primals_693: "f32[128]", primals_694: "f32[128]", primals_695: "f32[128]", primals_696: "f32[512, 128]", primals_697: "f32[512]", primals_698: "f32[512]", primals_699: "f32[512]", primals_700: "f32[128, 512]", primals_701: "f32[128]", primals_702: "f32[128]", primals_703: "f32[128]", primals_704: "f32[128, 512]", primals_705: "f32[128]", primals_706: "f32[128]", primals_707: "f32[128]", primals_708: "f32[128, 128]", primals_709: "f32[128]", primals_710: "f32[128, 128]", primals_711: "f32[128]", primals_712: "f32[128, 512]", primals_713: "f32[128]", primals_714: "f32[128, 128]", primals_715: "f32[128]", primals_716: "f32[128]", primals_717: "f32[128]", primals_718: "f32[512, 128]", primals_719: "f32[512]", primals_720: "f32[128, 512]", primals_721: "f32[128]", primals_722: "f32[128]", primals_723: "f32[128]", primals_724: "f32[512, 128]", primals_725: "f32[512]", primals_726: "f32[128, 512]", primals_727: "f32[128]", primals_728: "f32[128]", primals_729: "f32[128]", primals_730: "f32[512, 128]", primals_731: "f32[512]", primals_732: "f32[128, 512]", primals_733: "f32[128]", primals_734: "f32[128]", primals_735: "f32[128]", primals_736: "f32[512, 128]", primals_737: "f32[512]", primals_738: "f32[128, 512]", primals_739: "f32[128]", primals_740: "f32[128]", primals_741: "f32[128]", primals_742: "f32[512, 128]", primals_743: "f32[512]", primals_744: "f32[512]", primals_745: "f32[512]", primals_746: "f32[128, 512]", primals_747: "f32[128]", primals_748: "f32[128]", primals_749: "f32[128]", primals_750: "f32[128, 512]", primals_751: "f32[128]", primals_752: "f32[128]", primals_753: "f32[128]", primals_754: "f32[128, 128]", primals_755: "f32[128]", primals_756: "f32[128, 128]", primals_757: "f32[128]", primals_758: "f32[128, 512]", primals_759: "f32[128]", primals_760: "f32[128, 128]", primals_761: "f32[128]", primals_762: "f32[128]", primals_763: "f32[128]", primals_764: "f32[512, 128]", primals_765: "f32[512]", primals_766: "f32[128, 512]", primals_767: "f32[128]", primals_768: "f32[128]", primals_769: "f32[128]", primals_770: "f32[512, 128]", primals_771: "f32[512]", primals_772: "f32[128, 512]", primals_773: "f32[128]", primals_774: "f32[128]", primals_775: "f32[128]", primals_776: "f32[512, 128]", primals_777: "f32[512]", primals_778: "f32[128, 512]", primals_779: "f32[128]", primals_780: "f32[128]", primals_781: "f32[128]", primals_782: "f32[512, 128]", primals_783: "f32[512]", primals_784: "f32[128, 512]", primals_785: "f32[128]", primals_786: "f32[128]", primals_787: "f32[128]", primals_788: "f32[512, 128]", primals_789: "f32[512]", primals_790: "f32[512]", primals_791: "f32[512]", primals_792: "f32[128, 512]", primals_793: "f32[128]", primals_794: "f32[128]", primals_795: "f32[128]", primals_796: "f32[128, 512]", primals_797: "f32[128]", primals_798: "f32[128]", primals_799: "f32[128]", primals_800: "f32[128, 128]", primals_801: "f32[128]", primals_802: "f32[128, 128]", primals_803: "f32[128]", primals_804: "f32[128, 512]", primals_805: "f32[128]", primals_806: "f32[128, 128]", primals_807: "f32[128]", primals_808: "f32[128]", primals_809: "f32[128]", primals_810: "f32[512, 128]", primals_811: "f32[512]", primals_812: "f32[128, 512]", primals_813: "f32[128]", primals_814: "f32[128]", primals_815: "f32[128]", primals_816: "f32[512, 128]", primals_817: "f32[512]", primals_818: "f32[128, 512]", primals_819: "f32[128]", primals_820: "f32[128]", primals_821: "f32[128]", primals_822: "f32[512, 128]", primals_823: "f32[512]", primals_824: "f32[128, 512]", primals_825: "f32[128]", primals_826: "f32[128]", primals_827: "f32[128]", primals_828: "f32[512, 128]", primals_829: "f32[512]", primals_830: "f32[128, 512]", primals_831: "f32[128]", primals_832: "f32[128]", primals_833: "f32[128]", primals_834: "f32[512, 128]", primals_835: "f32[512]", primals_836: "f32[512]", primals_837: "f32[512]", primals_838: "f32[128, 512]", primals_839: "f32[128]", primals_840: "f32[128]", primals_841: "f32[128]", primals_842: "f32[128, 512]", primals_843: "f32[128]", primals_844: "f32[128]", primals_845: "f32[128]", primals_846: "f32[128, 128]", primals_847: "f32[128]", primals_848: "f32[128, 128]", primals_849: "f32[128]", primals_850: "f32[128, 512]", primals_851: "f32[128]", primals_852: "f32[128, 128]", primals_853: "f32[128]", primals_854: "f32[128]", primals_855: "f32[128]", primals_856: "f32[512, 128]", primals_857: "f32[512]", primals_858: "f32[128, 512]", primals_859: "f32[128]", primals_860: "f32[128]", primals_861: "f32[128]", primals_862: "f32[512, 128]", primals_863: "f32[512]", primals_864: "f32[128, 512]", primals_865: "f32[128]", primals_866: "f32[128]", primals_867: "f32[128]", primals_868: "f32[512, 128]", primals_869: "f32[512]", primals_870: "f32[128, 512]", primals_871: "f32[128]", primals_872: "f32[128]", primals_873: "f32[128]", primals_874: "f32[512, 128]", primals_875: "f32[512]", primals_876: "f32[128, 512]", primals_877: "f32[128]", primals_878: "f32[128]", primals_879: "f32[128]", primals_880: "f32[512, 128]", primals_881: "f32[512]", primals_882: "f32[512]", primals_883: "f32[512]", primals_884: "f32[128, 512]", primals_885: "f32[128]", primals_886: "f32[128]", primals_887: "f32[128]", primals_888: "f32[128, 512]", primals_889: "f32[128]", primals_890: "f32[128]", primals_891: "f32[128]", primals_892: "f32[128, 128]", primals_893: "f32[128]", primals_894: "f32[128, 128]", primals_895: "f32[128]", primals_896: "f32[128, 512]", primals_897: "f32[128]", primals_898: "f32[128, 128]", primals_899: "f32[128]", primals_900: "f32[128]", primals_901: "f32[128]", primals_902: "f32[512, 128]", primals_903: "f32[512]", primals_904: "f32[128, 512]", primals_905: "f32[128]", primals_906: "f32[128]", primals_907: "f32[128]", primals_908: "f32[512, 128]", primals_909: "f32[512]", primals_910: "f32[128, 512]", primals_911: "f32[128]", primals_912: "f32[128]", primals_913: "f32[128]", primals_914: "f32[512, 128]", primals_915: "f32[512]", primals_916: "f32[128, 512]", primals_917: "f32[128]", primals_918: "f32[128]", primals_919: "f32[128]", primals_920: "f32[512, 128]", primals_921: "f32[512]", primals_922: "f32[128, 512]", primals_923: "f32[128]", primals_924: "f32[128]", primals_925: "f32[128]", primals_926: "f32[512, 128]", primals_927: "f32[512]", primals_928: "f32[512]", primals_929: "f32[512]", primals_930: "f32[128, 512]", primals_931: "f32[128]", primals_932: "f32[128]", primals_933: "f32[128]", primals_934: "f32[128, 512]", primals_935: "f32[128]", primals_936: "f32[128]", primals_937: "f32[128]", primals_938: "f32[128, 128]", primals_939: "f32[128]", primals_940: "f32[128, 128]", primals_941: "f32[128]", primals_942: "f32[128, 512]", primals_943: "f32[128]", primals_944: "f32[128, 128]", primals_945: "f32[128]", primals_946: "f32[128]", primals_947: "f32[128]", primals_948: "f32[512, 128]", primals_949: "f32[512]", primals_950: "f32[128, 512]", primals_951: "f32[128]", primals_952: "f32[128]", primals_953: "f32[128]", primals_954: "f32[512, 128]", primals_955: "f32[512]", primals_956: "f32[128, 512]", primals_957: "f32[128]", primals_958: "f32[128]", primals_959: "f32[128]", primals_960: "f32[512, 128]", primals_961: "f32[512]", primals_962: "f32[128, 512]", primals_963: "f32[128]", primals_964: "f32[128]", primals_965: "f32[128]", primals_966: "f32[512, 128]", primals_967: "f32[512]", primals_968: "f32[128, 512]", primals_969: "f32[128]", primals_970: "f32[128]", primals_971: "f32[128]", primals_972: "f32[512, 128]", primals_973: "f32[512]", primals_974: "f32[512]", primals_975: "f32[512]", primals_976: "f32[128, 512]", primals_977: "f32[128]", primals_978: "f32[128]", primals_979: "f32[128]", primals_980: "f32[128, 512]", primals_981: "f32[128]", primals_982: "f32[128]", primals_983: "f32[128]", primals_984: "f32[128, 128]", primals_985: "f32[128]", primals_986: "f32[128, 128]", primals_987: "f32[128]", primals_988: "f32[128, 512]", primals_989: "f32[128]", primals_990: "f32[128, 128]", primals_991: "f32[128]", primals_992: "f32[128]", primals_993: "f32[128]", primals_994: "f32[512, 128]", primals_995: "f32[512]", primals_996: "f32[128, 512]", primals_997: "f32[128]", primals_998: "f32[128]", primals_999: "f32[128]", primals_1000: "f32[512, 128]", primals_1001: "f32[512]", primals_1002: "f32[128, 512]", primals_1003: "f32[128]", primals_1004: "f32[128]", primals_1005: "f32[128]", primals_1006: "f32[512, 128]", primals_1007: "f32[512]", primals_1008: "f32[128, 512]", primals_1009: "f32[128]", primals_1010: "f32[128]", primals_1011: "f32[128]", primals_1012: "f32[512, 128]", primals_1013: "f32[512]", primals_1014: "f32[128, 512]", primals_1015: "f32[128]", primals_1016: "f32[128]", primals_1017: "f32[128]", primals_1018: "f32[512, 128]", primals_1019: "f32[512]", primals_1020: "f32[512]", primals_1021: "f32[512]", primals_1022: "f32[128, 512]", primals_1023: "f32[128]", primals_1024: "f32[128]", primals_1025: "f32[128]", primals_1026: "f32[128, 512]", primals_1027: "f32[128]", primals_1028: "f32[128]", primals_1029: "f32[128]", primals_1030: "f32[128, 128]", primals_1031: "f32[128]", primals_1032: "f32[128, 128]", primals_1033: "f32[128]", primals_1034: "f32[128, 512]", primals_1035: "f32[128]", primals_1036: "f32[128, 128]", primals_1037: "f32[128]", primals_1038: "f32[128]", primals_1039: "f32[128]", primals_1040: "f32[512, 128]", primals_1041: "f32[512]", primals_1042: "f32[128, 512]", primals_1043: "f32[128]", primals_1044: "f32[128]", primals_1045: "f32[128]", primals_1046: "f32[512, 128]", primals_1047: "f32[512]", primals_1048: "f32[128, 512]", primals_1049: "f32[128]", primals_1050: "f32[128]", primals_1051: "f32[128]", primals_1052: "f32[512, 128]", primals_1053: "f32[512]", primals_1054: "f32[128, 512]", primals_1055: "f32[128]", primals_1056: "f32[128]", primals_1057: "f32[128]", primals_1058: "f32[512, 128]", primals_1059: "f32[512]", primals_1060: "f32[128, 512]", primals_1061: "f32[128]", primals_1062: "f32[128]", primals_1063: "f32[128]", primals_1064: "f32[512, 128]", primals_1065: "f32[512]", primals_1066: "f32[512]", primals_1067: "f32[512]", primals_1068: "f32[128, 512]", primals_1069: "f32[128]", primals_1070: "f32[128]", primals_1071: "f32[128]", primals_1072: "f32[128, 512]", primals_1073: "f32[128]", primals_1074: "f32[128]", primals_1075: "f32[128]", primals_1076: "f32[128, 128]", primals_1077: "f32[128]", primals_1078: "f32[128, 128]", primals_1079: "f32[128]", primals_1080: "f32[128, 512]", primals_1081: "f32[128]", primals_1082: "f32[128, 128]", primals_1083: "f32[128]", primals_1084: "f32[128]", primals_1085: "f32[128]", primals_1086: "f32[512, 128]", primals_1087: "f32[512]", primals_1088: "f32[128, 512]", primals_1089: "f32[128]", primals_1090: "f32[128]", primals_1091: "f32[128]", primals_1092: "f32[512, 128]", primals_1093: "f32[512]", primals_1094: "f32[128, 512]", primals_1095: "f32[128]", primals_1096: "f32[128]", primals_1097: "f32[128]", primals_1098: "f32[512, 128]", primals_1099: "f32[512]", primals_1100: "f32[128, 512]", primals_1101: "f32[128]", primals_1102: "f32[128]", primals_1103: "f32[128]", primals_1104: "f32[512, 128]", primals_1105: "f32[512]", primals_1106: "f32[128, 512]", primals_1107: "f32[128]", primals_1108: "f32[128]", primals_1109: "f32[128]", primals_1110: "f32[512, 128]", primals_1111: "f32[512]", primals_1112: "f32[512]", primals_1113: "f32[512]", primals_1114: "f32[512, 512]", primals_1115: "f32[512]", primals_1116: "f32[512]", primals_1117: "f32[512]", primals_1118: "f32[384, 30522]", primals_1119: "f32[30522]", primals_1120: "i64[256, 128]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:108 in forward, code: position_ids = self.position_ids[:, :seq_length]
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(primals_2, 1, 0, 128)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:111 in forward, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        full_default: "i64[256, 128]" = torch.ops.aten.full.default([256, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:113 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[256, 128, 128]" = torch.ops.aten.embedding.default(primals_3, primals_1, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:125 in forward, code: nn.functional.pad(inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0], value=0.0),
        slice_2: "f32[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding, 1, 1, 9223372036854775807)

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "f32[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_2, [0, 0, 0, 1, 0, 0], 0.0);  slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:127 in forward, code: nn.functional.pad(inputs_embeds[:, :-1], [0, 0, 1, 0, 0, 0], value=0.0),
        slice_3: "f32[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding, 1, 0, -1)

        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_1: "f32[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_3, [0, 0, 1, 0, 0, 0], 0.0);  slice_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:123 in forward, code: inputs_embeds = torch.cat(
        cat: "f32[256, 128, 384]" = torch.ops.aten.cat.default([constant_pad_nd, embedding, constant_pad_nd_1], 2);  constant_pad_nd = embedding = constant_pad_nd_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:132 in forward, code: inputs_embeds = self.embedding_transformation(inputs_embeds)
        view: "f32[32768, 384]" = torch.ops.aten.reshape.default(cat, [32768, 384]);  cat = None
        permute: "f32[384, 512]" = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_5, view, permute);  primals_5 = permute = None
        view_1: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm, [256, 128, 512]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:136 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_1: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(primals_6, slice_1);  primals_6 = slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:137 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_2: "f32[256, 128, 512]" = torch.ops.aten.embedding.default(primals_7, full_default);  primals_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:138 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        add: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_1, embedding_1);  view_1 = embedding_1 = None
        add_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_1, primals_8)
        add_2: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul, primals_9);  mul = primals_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_2: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_3: "i64[128]" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_3, 0);  add_3 = None
        unsqueeze_1: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 128, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[256, 1, 128, 128]" = torch.ops.aten.expand.default(ge, [256, -1, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_2: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_2, [32768, 512])
        permute_1: "f32[512, 128]" = torch.ops.aten.permute.default(primals_10, [1, 0])
        addmm_1: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_11, view_2, permute_1);  primals_11 = permute_1 = None
        view_3: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_1, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_1: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_3, primals_12);  view_3 = None
        add_5: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_1, primals_13);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_2: "f32[512, 128]" = torch.ops.aten.permute.default(primals_14, [1, 0])
        addmm_2: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_15, view_2, permute_2);  primals_15 = permute_2 = None
        view_5: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_2, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_2: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_5, primals_16);  view_5 = None
        add_6: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_2, primals_17);  mul_2 = primals_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_6: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_6, [32768, 128]);  add_6 = None
        permute_3: "f32[128, 128]" = torch.ops.aten.permute.default(primals_18, [1, 0])
        addmm_3: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_19, view_6, permute_3);  primals_19 = permute_3 = None
        view_7: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_3, [256, 128, 128]);  addmm_3 = None
        view_8: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_7, [256, 128, -1, 32]);  view_7 = None
        permute_4: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_5: "f32[128, 128]" = torch.ops.aten.permute.default(primals_20, [1, 0])
        addmm_4: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_21, view_6, permute_5);  primals_21 = permute_5 = None
        view_10: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_4, [256, 128, 128]);  addmm_4 = None
        view_11: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_10, [256, 128, -1, 32]);  view_10 = None
        permute_6: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_7: "f32[512, 128]" = torch.ops.aten.permute.default(primals_22, [1, 0])
        addmm_5: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_23, view_2, permute_7);  primals_23 = permute_7 = None
        view_13: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_5, [256, 128, 128]);  addmm_5 = None
        view_14: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_13, [256, 128, -1, 32]);  view_13 = None
        permute_8: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[256, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_2, full_default_1);  expand = full_default_1 = None
        mul_3: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_4, 0.4204482076268573);  permute_4 = None
        permute_9: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_6, [0, 1, 3, 2]);  permute_6 = None
        mul_4: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_9, 0.4204482076268573);  permute_9 = None
        expand_1: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_3, [256, 4, 128, 32]);  mul_3 = None
        clone_1: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_15: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_1, [1024, 128, 32]);  clone_1 = None
        expand_2: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_4, [256, 4, 32, 128]);  mul_4 = None
        clone_2: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_16: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_2, [1024, 32, 128]);  clone_2 = None
        bmm: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_15, view_16)
        view_17: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm, [256, 4, 128, 128])
        add_7: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_17, where);  view_17 = None
        amax: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_7, [-1], True)
        sub: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_7, amax)
        exp: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        eq: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_7, -inf);  add_7 = None
        logical_not: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_3: "f32[256, 4, 128, 128]" = torch.ops.aten.full.default([256, 4, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full_default_3, div);  div = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[24]" = torch.ops.prims.inductor_seeds.default(24, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_23: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_23, 0.1);  inductor_random_default_23 = None
        mul_5: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt, where_1);  where_1 = None
        mul_6: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_5, 1.1111111111111112);  mul_5 = None
        expand_3: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_6, [256, 4, 128, 128]);  mul_6 = None
        view_18: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_3, [1024, 128, 128]);  expand_3 = None
        expand_4: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_8, [256, 4, 128, 32]);  permute_8 = None
        clone_3: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_19: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_3, [1024, 128, 32]);  clone_3 = None
        bmm_1: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_18, view_19)
        view_20: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_1, [256, 4, 128, 32]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_10: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
        clone_4: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_21: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_4, [256, 128, -1]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_22: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_21, [32768, 128]);  view_21 = None
        permute_11: "f32[128, 128]" = torch.ops.aten.permute.default(primals_24, [1, 0])
        addmm_6: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_25, view_22, permute_11);  primals_25 = permute_11 = None
        view_23: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_6, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_8: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_23, add_5);  view_23 = add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_7: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_8, primals_26);  add_8 = None
        add_9: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_7, primals_27);  mul_7 = primals_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_24: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_9, [32768, 128])
        permute_12: "f32[128, 512]" = torch.ops.aten.permute.default(primals_28, [1, 0])
        addmm_7: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_29, view_24, permute_12);  primals_29 = permute_12 = None
        view_25: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_7, [256, 128, 512]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_25);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_26: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu, [32768, 512])
        permute_13: "f32[512, 128]" = torch.ops.aten.permute.default(primals_30, [1, 0])
        addmm_8: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_31, view_26, permute_13);  primals_31 = permute_13 = None
        view_27: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_8, [256, 128, 128]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_10: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_27, add_9);  view_27 = add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_8: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_10, primals_32)
        add_11: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_8, primals_33);  mul_8 = primals_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_28: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_11, [32768, 128])
        permute_14: "f32[128, 512]" = torch.ops.aten.permute.default(primals_34, [1, 0])
        addmm_9: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_35, view_28, permute_14);  primals_35 = permute_14 = None
        view_29: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_9, [256, 128, 512]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_1: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_29);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_30: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_1, [32768, 512])
        permute_15: "f32[512, 128]" = torch.ops.aten.permute.default(primals_36, [1, 0])
        addmm_10: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_37, view_30, permute_15);  primals_37 = permute_15 = None
        view_31: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_10, [256, 128, 128]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_12: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_31, add_11);  view_31 = add_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_9: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_12, primals_38)
        add_13: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_9, primals_39);  mul_9 = primals_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_32: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_13, [32768, 128])
        permute_16: "f32[128, 512]" = torch.ops.aten.permute.default(primals_40, [1, 0])
        addmm_11: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_41, view_32, permute_16);  primals_41 = permute_16 = None
        view_33: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_11, [256, 128, 512]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_2: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_33);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_34: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_2, [32768, 512])
        permute_17: "f32[512, 128]" = torch.ops.aten.permute.default(primals_42, [1, 0])
        addmm_12: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_43, view_34, permute_17);  primals_43 = permute_17 = None
        view_35: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_12, [256, 128, 128]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_14: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_35, add_13);  view_35 = add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_10: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_14, primals_44)
        add_15: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_10, primals_45);  mul_10 = primals_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_36: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_15, [32768, 128])
        permute_18: "f32[128, 512]" = torch.ops.aten.permute.default(primals_46, [1, 0])
        addmm_13: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_47, view_36, permute_18);  primals_47 = permute_18 = None
        view_37: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_13, [256, 128, 512]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_3: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_37);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_38: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_3, [32768, 512])
        permute_19: "f32[512, 128]" = torch.ops.aten.permute.default(primals_48, [1, 0])
        addmm_14: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_49, view_38, permute_19);  primals_49 = permute_19 = None
        view_39: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_14, [256, 128, 128]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_16: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_39, add_15);  view_39 = add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_11: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_16, primals_50)
        add_17: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_11, primals_51);  mul_11 = primals_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_40: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_17, [32768, 128]);  add_17 = None
        permute_20: "f32[128, 512]" = torch.ops.aten.permute.default(primals_52, [1, 0])
        addmm_15: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_53, view_40, permute_20);  primals_53 = permute_20 = None
        view_41: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_15, [256, 128, 512]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_18: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_41, add_2);  view_41 = add_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_12: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_18, primals_54)
        add_19: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_12, primals_55);  mul_12 = primals_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_42: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_19, [32768, 512])
        permute_21: "f32[512, 128]" = torch.ops.aten.permute.default(primals_56, [1, 0])
        addmm_16: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_57, view_42, permute_21);  primals_57 = permute_21 = None
        view_43: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_16, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_13: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_43, primals_58);  view_43 = None
        add_20: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_13, primals_59);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_22: "f32[512, 128]" = torch.ops.aten.permute.default(primals_60, [1, 0])
        addmm_17: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_61, view_42, permute_22);  primals_61 = permute_22 = None
        view_45: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_17, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_14: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_45, primals_62);  view_45 = None
        add_21: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_14, primals_63);  mul_14 = primals_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_46: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_21, [32768, 128]);  add_21 = None
        permute_23: "f32[128, 128]" = torch.ops.aten.permute.default(primals_64, [1, 0])
        addmm_18: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_65, view_46, permute_23);  primals_65 = permute_23 = None
        view_47: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_18, [256, 128, 128]);  addmm_18 = None
        view_48: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_47, [256, 128, -1, 32]);  view_47 = None
        permute_24: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_25: "f32[128, 128]" = torch.ops.aten.permute.default(primals_66, [1, 0])
        addmm_19: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_67, view_46, permute_25);  primals_67 = permute_25 = None
        view_50: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_19, [256, 128, 128]);  addmm_19 = None
        view_51: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_50, [256, 128, -1, 32]);  view_50 = None
        permute_26: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_27: "f32[512, 128]" = torch.ops.aten.permute.default(primals_68, [1, 0])
        addmm_20: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_69, view_42, permute_27);  primals_69 = permute_27 = None
        view_53: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_20, [256, 128, 128]);  addmm_20 = None
        view_54: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_53, [256, 128, -1, 32]);  view_53 = None
        permute_28: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_15: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_24, 0.4204482076268573);  permute_24 = None
        permute_29: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        mul_16: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_29, 0.4204482076268573);  permute_29 = None
        expand_5: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_15, [256, 4, 128, 32]);  mul_15 = None
        clone_6: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_55: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_6, [1024, 128, 32]);  clone_6 = None
        expand_6: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_16, [256, 4, 32, 128]);  mul_16 = None
        clone_7: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_56: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_7, [1024, 32, 128]);  clone_7 = None
        bmm_2: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_55, view_56)
        view_57: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_2, [256, 4, 128, 128]);  bmm_2 = None
        add_22: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_57, where);  view_57 = None
        amax_1: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_22, [-1], True)
        sub_1: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_22, amax_1);  amax_1 = None
        exp_1: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_2: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        eq_1: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_22, -inf);  add_22 = None
        logical_not_2: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        where_3: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_3, full_default_3, div_1);  logical_not_3 = div_1 = None
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_22: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        gt_1: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_22, 0.1);  inductor_random_default_22 = None
        mul_17: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_1, where_3)
        mul_18: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_17, 1.1111111111111112);  mul_17 = None
        expand_7: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_18, [256, 4, 128, 128]);  mul_18 = None
        view_58: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_7, [1024, 128, 128]);  expand_7 = None
        expand_8: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_28, [256, 4, 128, 32]);  permute_28 = None
        clone_8: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_59: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_8, [1024, 128, 32]);  clone_8 = None
        bmm_3: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_58, view_59)
        view_60: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_3, [256, 4, 128, 32]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_30: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        clone_9: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_61: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_9, [256, 128, -1]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_62: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_61, [32768, 128]);  view_61 = None
        permute_31: "f32[128, 128]" = torch.ops.aten.permute.default(primals_70, [1, 0])
        addmm_21: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_71, view_62, permute_31);  primals_71 = permute_31 = None
        view_63: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_21, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_23: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_63, add_20);  view_63 = add_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_19: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_23, primals_72);  add_23 = None
        add_24: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_19, primals_73);  mul_19 = primals_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_64: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_24, [32768, 128])
        permute_32: "f32[128, 512]" = torch.ops.aten.permute.default(primals_74, [1, 0])
        addmm_22: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_75, view_64, permute_32);  primals_75 = permute_32 = None
        view_65: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_22, [256, 128, 512]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_4: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_65);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_66: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_4, [32768, 512])
        permute_33: "f32[512, 128]" = torch.ops.aten.permute.default(primals_76, [1, 0])
        addmm_23: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_77, view_66, permute_33);  primals_77 = permute_33 = None
        view_67: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_23, [256, 128, 128]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_25: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_67, add_24);  view_67 = add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_20: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_25, primals_78)
        add_26: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_20, primals_79);  mul_20 = primals_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_68: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_26, [32768, 128])
        permute_34: "f32[128, 512]" = torch.ops.aten.permute.default(primals_80, [1, 0])
        addmm_24: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_81, view_68, permute_34);  primals_81 = permute_34 = None
        view_69: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_24, [256, 128, 512]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_5: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_69);  view_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_70: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_5, [32768, 512])
        permute_35: "f32[512, 128]" = torch.ops.aten.permute.default(primals_82, [1, 0])
        addmm_25: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_83, view_70, permute_35);  primals_83 = permute_35 = None
        view_71: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_25, [256, 128, 128]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_27: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_71, add_26);  view_71 = add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_21: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_27, primals_84)
        add_28: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_21, primals_85);  mul_21 = primals_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_72: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_28, [32768, 128])
        permute_36: "f32[128, 512]" = torch.ops.aten.permute.default(primals_86, [1, 0])
        addmm_26: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_87, view_72, permute_36);  primals_87 = permute_36 = None
        view_73: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_26, [256, 128, 512]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_6: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_73);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_74: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_6, [32768, 512])
        permute_37: "f32[512, 128]" = torch.ops.aten.permute.default(primals_88, [1, 0])
        addmm_27: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_89, view_74, permute_37);  primals_89 = permute_37 = None
        view_75: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_27, [256, 128, 128]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_29: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_75, add_28);  view_75 = add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_22: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_29, primals_90)
        add_30: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_22, primals_91);  mul_22 = primals_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_76: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_30, [32768, 128])
        permute_38: "f32[128, 512]" = torch.ops.aten.permute.default(primals_92, [1, 0])
        addmm_28: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_93, view_76, permute_38);  primals_93 = permute_38 = None
        view_77: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_28, [256, 128, 512]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_7: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_77);  view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_78: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_7, [32768, 512])
        permute_39: "f32[512, 128]" = torch.ops.aten.permute.default(primals_94, [1, 0])
        addmm_29: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_95, view_78, permute_39);  primals_95 = permute_39 = None
        view_79: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_29, [256, 128, 128]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_31: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_79, add_30);  view_79 = add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_23: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_31, primals_96)
        add_32: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_23, primals_97);  mul_23 = primals_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_80: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_32, [32768, 128]);  add_32 = None
        permute_40: "f32[128, 512]" = torch.ops.aten.permute.default(primals_98, [1, 0])
        addmm_30: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_99, view_80, permute_40);  primals_99 = permute_40 = None
        view_81: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_30, [256, 128, 512]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_33: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_81, add_19);  view_81 = add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_24: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_33, primals_100)
        add_34: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_24, primals_101);  mul_24 = primals_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_82: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_34, [32768, 512])
        permute_41: "f32[512, 128]" = torch.ops.aten.permute.default(primals_102, [1, 0])
        addmm_31: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_103, view_82, permute_41);  primals_103 = permute_41 = None
        view_83: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_31, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_25: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_83, primals_104);  view_83 = None
        add_35: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_25, primals_105);  mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_42: "f32[512, 128]" = torch.ops.aten.permute.default(primals_106, [1, 0])
        addmm_32: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_107, view_82, permute_42);  primals_107 = permute_42 = None
        view_85: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_32, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_26: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_85, primals_108);  view_85 = None
        add_36: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_26, primals_109);  mul_26 = primals_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_86: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_36, [32768, 128]);  add_36 = None
        permute_43: "f32[128, 128]" = torch.ops.aten.permute.default(primals_110, [1, 0])
        addmm_33: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_111, view_86, permute_43);  primals_111 = permute_43 = None
        view_87: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_33, [256, 128, 128]);  addmm_33 = None
        view_88: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_87, [256, 128, -1, 32]);  view_87 = None
        permute_44: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_45: "f32[128, 128]" = torch.ops.aten.permute.default(primals_112, [1, 0])
        addmm_34: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_113, view_86, permute_45);  primals_113 = permute_45 = None
        view_90: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_34, [256, 128, 128]);  addmm_34 = None
        view_91: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_90, [256, 128, -1, 32]);  view_90 = None
        permute_46: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_91, [0, 2, 1, 3]);  view_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_47: "f32[512, 128]" = torch.ops.aten.permute.default(primals_114, [1, 0])
        addmm_35: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_115, view_82, permute_47);  primals_115 = permute_47 = None
        view_93: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_35, [256, 128, 128]);  addmm_35 = None
        view_94: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_93, [256, 128, -1, 32]);  view_93 = None
        permute_48: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_27: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_44, 0.4204482076268573);  permute_44 = None
        permute_49: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2]);  permute_46 = None
        mul_28: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_49, 0.4204482076268573);  permute_49 = None
        expand_9: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_27, [256, 4, 128, 32]);  mul_27 = None
        clone_11: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_95: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_11, [1024, 128, 32]);  clone_11 = None
        expand_10: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_28, [256, 4, 32, 128]);  mul_28 = None
        clone_12: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_96: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_12, [1024, 32, 128]);  clone_12 = None
        bmm_4: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_95, view_96)
        view_97: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_4, [256, 4, 128, 128]);  bmm_4 = None
        add_37: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_97, where);  view_97 = None
        amax_2: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_37, [-1], True)
        sub_2: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_37, amax_2);  amax_2 = None
        exp_2: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_3: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        eq_2: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_37, -inf);  add_37 = None
        logical_not_4: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        where_5: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_5, full_default_3, div_2);  logical_not_5 = div_2 = None
        inductor_lookup_seed_default_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_21: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_21, 0.1);  inductor_random_default_21 = None
        mul_29: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_2, where_5)
        mul_30: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_29, 1.1111111111111112);  mul_29 = None
        expand_11: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_30, [256, 4, 128, 128]);  mul_30 = None
        view_98: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_11, [1024, 128, 128]);  expand_11 = None
        expand_12: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_48, [256, 4, 128, 32]);  permute_48 = None
        clone_13: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_99: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_13, [1024, 128, 32]);  clone_13 = None
        bmm_5: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_98, view_99)
        view_100: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_5, [256, 4, 128, 32]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_50: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_100, [0, 2, 1, 3]);  view_100 = None
        clone_14: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_101: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_14, [256, 128, -1]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_102: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_101, [32768, 128]);  view_101 = None
        permute_51: "f32[128, 128]" = torch.ops.aten.permute.default(primals_116, [1, 0])
        addmm_36: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_117, view_102, permute_51);  primals_117 = permute_51 = None
        view_103: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_36, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_38: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_103, add_35);  view_103 = add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_31: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_38, primals_118);  add_38 = None
        add_39: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_31, primals_119);  mul_31 = primals_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_104: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_39, [32768, 128])
        permute_52: "f32[128, 512]" = torch.ops.aten.permute.default(primals_120, [1, 0])
        addmm_37: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_121, view_104, permute_52);  primals_121 = permute_52 = None
        view_105: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_37, [256, 128, 512]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_8: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_105);  view_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_106: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_8, [32768, 512])
        permute_53: "f32[512, 128]" = torch.ops.aten.permute.default(primals_122, [1, 0])
        addmm_38: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_123, view_106, permute_53);  primals_123 = permute_53 = None
        view_107: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_38, [256, 128, 128]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_40: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_107, add_39);  view_107 = add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_32: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_40, primals_124)
        add_41: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_32, primals_125);  mul_32 = primals_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_41, [32768, 128])
        permute_54: "f32[128, 512]" = torch.ops.aten.permute.default(primals_126, [1, 0])
        addmm_39: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_127, view_108, permute_54);  primals_127 = permute_54 = None
        view_109: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_39, [256, 128, 512]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_9: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_109);  view_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_110: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_9, [32768, 512])
        permute_55: "f32[512, 128]" = torch.ops.aten.permute.default(primals_128, [1, 0])
        addmm_40: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_129, view_110, permute_55);  primals_129 = permute_55 = None
        view_111: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_40, [256, 128, 128]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_42: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_111, add_41);  view_111 = add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_33: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_42, primals_130)
        add_43: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_33, primals_131);  mul_33 = primals_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_112: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_43, [32768, 128])
        permute_56: "f32[128, 512]" = torch.ops.aten.permute.default(primals_132, [1, 0])
        addmm_41: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_133, view_112, permute_56);  primals_133 = permute_56 = None
        view_113: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_41, [256, 128, 512]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_10: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_113);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_114: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_10, [32768, 512])
        permute_57: "f32[512, 128]" = torch.ops.aten.permute.default(primals_134, [1, 0])
        addmm_42: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_135, view_114, permute_57);  primals_135 = permute_57 = None
        view_115: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_42, [256, 128, 128]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_44: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_115, add_43);  view_115 = add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_34: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_44, primals_136)
        add_45: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_34, primals_137);  mul_34 = primals_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_116: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_45, [32768, 128])
        permute_58: "f32[128, 512]" = torch.ops.aten.permute.default(primals_138, [1, 0])
        addmm_43: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_139, view_116, permute_58);  primals_139 = permute_58 = None
        view_117: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_43, [256, 128, 512]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_11: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_117);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_118: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_11, [32768, 512])
        permute_59: "f32[512, 128]" = torch.ops.aten.permute.default(primals_140, [1, 0])
        addmm_44: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_141, view_118, permute_59);  primals_141 = permute_59 = None
        view_119: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_44, [256, 128, 128]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_46: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_119, add_45);  view_119 = add_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_35: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_46, primals_142)
        add_47: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_35, primals_143);  mul_35 = primals_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_120: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_47, [32768, 128]);  add_47 = None
        permute_60: "f32[128, 512]" = torch.ops.aten.permute.default(primals_144, [1, 0])
        addmm_45: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_145, view_120, permute_60);  primals_145 = permute_60 = None
        view_121: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_45, [256, 128, 512]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_48: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_121, add_34);  view_121 = add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_36: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_48, primals_146)
        add_49: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_36, primals_147);  mul_36 = primals_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_122: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_49, [32768, 512])
        permute_61: "f32[512, 128]" = torch.ops.aten.permute.default(primals_148, [1, 0])
        addmm_46: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_149, view_122, permute_61);  primals_149 = permute_61 = None
        view_123: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_46, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_37: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_123, primals_150);  view_123 = None
        add_50: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_37, primals_151);  mul_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_62: "f32[512, 128]" = torch.ops.aten.permute.default(primals_152, [1, 0])
        addmm_47: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_153, view_122, permute_62);  primals_153 = permute_62 = None
        view_125: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_47, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_38: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_125, primals_154);  view_125 = None
        add_51: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_38, primals_155);  mul_38 = primals_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_126: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_51, [32768, 128]);  add_51 = None
        permute_63: "f32[128, 128]" = torch.ops.aten.permute.default(primals_156, [1, 0])
        addmm_48: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_157, view_126, permute_63);  primals_157 = permute_63 = None
        view_127: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_48, [256, 128, 128]);  addmm_48 = None
        view_128: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_127, [256, 128, -1, 32]);  view_127 = None
        permute_64: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_65: "f32[128, 128]" = torch.ops.aten.permute.default(primals_158, [1, 0])
        addmm_49: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_159, view_126, permute_65);  primals_159 = permute_65 = None
        view_130: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_49, [256, 128, 128]);  addmm_49 = None
        view_131: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_130, [256, 128, -1, 32]);  view_130 = None
        permute_66: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_67: "f32[512, 128]" = torch.ops.aten.permute.default(primals_160, [1, 0])
        addmm_50: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_161, view_122, permute_67);  primals_161 = permute_67 = None
        view_133: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_50, [256, 128, 128]);  addmm_50 = None
        view_134: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_133, [256, 128, -1, 32]);  view_133 = None
        permute_68: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_39: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_64, 0.4204482076268573);  permute_64 = None
        permute_69: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_66, [0, 1, 3, 2]);  permute_66 = None
        mul_40: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_69, 0.4204482076268573);  permute_69 = None
        expand_13: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_39, [256, 4, 128, 32]);  mul_39 = None
        clone_16: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_135: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_16, [1024, 128, 32]);  clone_16 = None
        expand_14: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_40, [256, 4, 32, 128]);  mul_40 = None
        clone_17: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_136: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_17, [1024, 32, 128]);  clone_17 = None
        bmm_6: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_135, view_136)
        view_137: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_6, [256, 4, 128, 128]);  bmm_6 = None
        add_52: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_137, where);  view_137 = None
        amax_3: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_52, [-1], True)
        sub_3: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_52, amax_3);  amax_3 = None
        exp_3: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_4: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        eq_3: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_52, -inf);  add_52 = None
        logical_not_6: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        where_7: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_7, full_default_3, div_3);  logical_not_7 = div_3 = None
        inductor_lookup_seed_default_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_20: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        gt_3: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_41: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_3, where_7)
        mul_42: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_41, 1.1111111111111112);  mul_41 = None
        expand_15: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_42, [256, 4, 128, 128]);  mul_42 = None
        view_138: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_15, [1024, 128, 128]);  expand_15 = None
        expand_16: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_68, [256, 4, 128, 32]);  permute_68 = None
        clone_18: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_139: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_18, [1024, 128, 32]);  clone_18 = None
        bmm_7: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_138, view_139)
        view_140: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_7, [256, 4, 128, 32]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_70: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        clone_19: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_141: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_19, [256, 128, -1]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_142: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_141, [32768, 128]);  view_141 = None
        permute_71: "f32[128, 128]" = torch.ops.aten.permute.default(primals_162, [1, 0])
        addmm_51: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_163, view_142, permute_71);  primals_163 = permute_71 = None
        view_143: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_51, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_53: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_143, add_50);  view_143 = add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_43: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_53, primals_164);  add_53 = None
        add_54: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_43, primals_165);  mul_43 = primals_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_144: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_54, [32768, 128])
        permute_72: "f32[128, 512]" = torch.ops.aten.permute.default(primals_166, [1, 0])
        addmm_52: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_167, view_144, permute_72);  primals_167 = permute_72 = None
        view_145: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_52, [256, 128, 512]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_12: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_145);  view_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_146: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_12, [32768, 512])
        permute_73: "f32[512, 128]" = torch.ops.aten.permute.default(primals_168, [1, 0])
        addmm_53: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_169, view_146, permute_73);  primals_169 = permute_73 = None
        view_147: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_53, [256, 128, 128]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_55: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_147, add_54);  view_147 = add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_44: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_55, primals_170)
        add_56: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_44, primals_171);  mul_44 = primals_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_148: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_56, [32768, 128])
        permute_74: "f32[128, 512]" = torch.ops.aten.permute.default(primals_172, [1, 0])
        addmm_54: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_173, view_148, permute_74);  primals_173 = permute_74 = None
        view_149: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_54, [256, 128, 512]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_13: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_149);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_150: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_13, [32768, 512])
        permute_75: "f32[512, 128]" = torch.ops.aten.permute.default(primals_174, [1, 0])
        addmm_55: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_175, view_150, permute_75);  primals_175 = permute_75 = None
        view_151: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_55, [256, 128, 128]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_57: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_151, add_56);  view_151 = add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_45: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_57, primals_176)
        add_58: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_45, primals_177);  mul_45 = primals_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_152: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_58, [32768, 128])
        permute_76: "f32[128, 512]" = torch.ops.aten.permute.default(primals_178, [1, 0])
        addmm_56: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_179, view_152, permute_76);  primals_179 = permute_76 = None
        view_153: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_56, [256, 128, 512]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_14: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_153);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_154: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_14, [32768, 512])
        permute_77: "f32[512, 128]" = torch.ops.aten.permute.default(primals_180, [1, 0])
        addmm_57: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_181, view_154, permute_77);  primals_181 = permute_77 = None
        view_155: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_57, [256, 128, 128]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_59: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_155, add_58);  view_155 = add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_46: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_59, primals_182)
        add_60: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_46, primals_183);  mul_46 = primals_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_156: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_60, [32768, 128])
        permute_78: "f32[128, 512]" = torch.ops.aten.permute.default(primals_184, [1, 0])
        addmm_58: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_185, view_156, permute_78);  primals_185 = permute_78 = None
        view_157: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_58, [256, 128, 512]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_15: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_157);  view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_158: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_15, [32768, 512])
        permute_79: "f32[512, 128]" = torch.ops.aten.permute.default(primals_186, [1, 0])
        addmm_59: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_187, view_158, permute_79);  primals_187 = permute_79 = None
        view_159: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_59, [256, 128, 128]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_61: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_159, add_60);  view_159 = add_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_47: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_61, primals_188)
        add_62: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_47, primals_189);  mul_47 = primals_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_160: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_62, [32768, 128]);  add_62 = None
        permute_80: "f32[128, 512]" = torch.ops.aten.permute.default(primals_190, [1, 0])
        addmm_60: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_191, view_160, permute_80);  primals_191 = permute_80 = None
        view_161: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_60, [256, 128, 512]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_63: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_161, add_49);  view_161 = add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_48: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_63, primals_192)
        add_64: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_48, primals_193);  mul_48 = primals_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_162: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_64, [32768, 512])
        permute_81: "f32[512, 128]" = torch.ops.aten.permute.default(primals_194, [1, 0])
        addmm_61: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_195, view_162, permute_81);  primals_195 = permute_81 = None
        view_163: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_61, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_49: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_163, primals_196);  view_163 = None
        add_65: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_49, primals_197);  mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_82: "f32[512, 128]" = torch.ops.aten.permute.default(primals_198, [1, 0])
        addmm_62: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_199, view_162, permute_82);  primals_199 = permute_82 = None
        view_165: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_62, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_50: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_165, primals_200);  view_165 = None
        add_66: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_50, primals_201);  mul_50 = primals_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_166: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_66, [32768, 128]);  add_66 = None
        permute_83: "f32[128, 128]" = torch.ops.aten.permute.default(primals_202, [1, 0])
        addmm_63: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_203, view_166, permute_83);  primals_203 = permute_83 = None
        view_167: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_63, [256, 128, 128]);  addmm_63 = None
        view_168: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_167, [256, 128, -1, 32]);  view_167 = None
        permute_84: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_85: "f32[128, 128]" = torch.ops.aten.permute.default(primals_204, [1, 0])
        addmm_64: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_205, view_166, permute_85);  primals_205 = permute_85 = None
        view_170: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_64, [256, 128, 128]);  addmm_64 = None
        view_171: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_170, [256, 128, -1, 32]);  view_170 = None
        permute_86: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_87: "f32[512, 128]" = torch.ops.aten.permute.default(primals_206, [1, 0])
        addmm_65: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_207, view_162, permute_87);  primals_207 = permute_87 = None
        view_173: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_65, [256, 128, 128]);  addmm_65 = None
        view_174: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_173, [256, 128, -1, 32]);  view_173 = None
        permute_88: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_51: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_84, 0.4204482076268573);  permute_84 = None
        permute_89: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_86, [0, 1, 3, 2]);  permute_86 = None
        mul_52: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_89, 0.4204482076268573);  permute_89 = None
        expand_17: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_51, [256, 4, 128, 32]);  mul_51 = None
        clone_21: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_175: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_21, [1024, 128, 32]);  clone_21 = None
        expand_18: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_52, [256, 4, 32, 128]);  mul_52 = None
        clone_22: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_176: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_22, [1024, 32, 128]);  clone_22 = None
        bmm_8: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_175, view_176)
        view_177: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_8, [256, 4, 128, 128]);  bmm_8 = None
        add_67: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_177, where);  view_177 = None
        amax_4: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_67, [-1], True)
        sub_4: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_67, amax_4);  amax_4 = None
        exp_4: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_5: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        eq_4: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_67, -inf);  add_67 = None
        logical_not_8: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        where_9: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_9, full_default_3, div_4);  logical_not_9 = div_4 = None
        inductor_lookup_seed_default_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_19: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        gt_4: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_19, 0.1);  inductor_random_default_19 = None
        mul_53: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_4, where_9)
        mul_54: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_53, 1.1111111111111112);  mul_53 = None
        expand_19: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_54, [256, 4, 128, 128]);  mul_54 = None
        view_178: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_19, [1024, 128, 128]);  expand_19 = None
        expand_20: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_88, [256, 4, 128, 32]);  permute_88 = None
        clone_23: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_179: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_23, [1024, 128, 32]);  clone_23 = None
        bmm_9: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_178, view_179)
        view_180: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_9, [256, 4, 128, 32]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_90: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
        clone_24: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_181: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_24, [256, 128, -1]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_182: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_181, [32768, 128]);  view_181 = None
        permute_91: "f32[128, 128]" = torch.ops.aten.permute.default(primals_208, [1, 0])
        addmm_66: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_209, view_182, permute_91);  primals_209 = permute_91 = None
        view_183: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_66, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_68: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_183, add_65);  view_183 = add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_55: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_68, primals_210);  add_68 = None
        add_69: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_55, primals_211);  mul_55 = primals_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_184: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_69, [32768, 128])
        permute_92: "f32[128, 512]" = torch.ops.aten.permute.default(primals_212, [1, 0])
        addmm_67: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_213, view_184, permute_92);  primals_213 = permute_92 = None
        view_185: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_67, [256, 128, 512]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_16: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_185);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_186: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_16, [32768, 512])
        permute_93: "f32[512, 128]" = torch.ops.aten.permute.default(primals_214, [1, 0])
        addmm_68: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_215, view_186, permute_93);  primals_215 = permute_93 = None
        view_187: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_68, [256, 128, 128]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_70: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_187, add_69);  view_187 = add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_56: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_70, primals_216)
        add_71: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_56, primals_217);  mul_56 = primals_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_188: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_71, [32768, 128])
        permute_94: "f32[128, 512]" = torch.ops.aten.permute.default(primals_218, [1, 0])
        addmm_69: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_219, view_188, permute_94);  primals_219 = permute_94 = None
        view_189: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_69, [256, 128, 512]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_17: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_189);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_190: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_17, [32768, 512])
        permute_95: "f32[512, 128]" = torch.ops.aten.permute.default(primals_220, [1, 0])
        addmm_70: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_221, view_190, permute_95);  primals_221 = permute_95 = None
        view_191: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_70, [256, 128, 128]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_72: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_191, add_71);  view_191 = add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_57: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_72, primals_222)
        add_73: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_57, primals_223);  mul_57 = primals_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_192: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_73, [32768, 128])
        permute_96: "f32[128, 512]" = torch.ops.aten.permute.default(primals_224, [1, 0])
        addmm_71: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_225, view_192, permute_96);  primals_225 = permute_96 = None
        view_193: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_71, [256, 128, 512]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_18: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_193);  view_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_194: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_18, [32768, 512])
        permute_97: "f32[512, 128]" = torch.ops.aten.permute.default(primals_226, [1, 0])
        addmm_72: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_227, view_194, permute_97);  primals_227 = permute_97 = None
        view_195: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_72, [256, 128, 128]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_74: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_195, add_73);  view_195 = add_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_58: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_74, primals_228)
        add_75: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_58, primals_229);  mul_58 = primals_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_196: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_75, [32768, 128])
        permute_98: "f32[128, 512]" = torch.ops.aten.permute.default(primals_230, [1, 0])
        addmm_73: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_231, view_196, permute_98);  primals_231 = permute_98 = None
        view_197: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_73, [256, 128, 512]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_19: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_197);  view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_198: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_19, [32768, 512])
        permute_99: "f32[512, 128]" = torch.ops.aten.permute.default(primals_232, [1, 0])
        addmm_74: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_233, view_198, permute_99);  primals_233 = permute_99 = None
        view_199: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_74, [256, 128, 128]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_76: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_199, add_75);  view_199 = add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_59: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_76, primals_234)
        add_77: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_59, primals_235);  mul_59 = primals_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_200: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_77, [32768, 128]);  add_77 = None
        permute_100: "f32[128, 512]" = torch.ops.aten.permute.default(primals_236, [1, 0])
        addmm_75: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_237, view_200, permute_100);  primals_237 = permute_100 = None
        view_201: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_75, [256, 128, 512]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_78: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_201, add_64);  view_201 = add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_60: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_78, primals_238)
        add_79: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_60, primals_239);  mul_60 = primals_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_202: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_79, [32768, 512])
        permute_101: "f32[512, 128]" = torch.ops.aten.permute.default(primals_240, [1, 0])
        addmm_76: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_241, view_202, permute_101);  primals_241 = permute_101 = None
        view_203: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_76, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_61: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_203, primals_242);  view_203 = None
        add_80: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_61, primals_243);  mul_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_102: "f32[512, 128]" = torch.ops.aten.permute.default(primals_244, [1, 0])
        addmm_77: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_245, view_202, permute_102);  primals_245 = permute_102 = None
        view_205: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_77, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_62: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_205, primals_246);  view_205 = None
        add_81: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_62, primals_247);  mul_62 = primals_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_206: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_81, [32768, 128]);  add_81 = None
        permute_103: "f32[128, 128]" = torch.ops.aten.permute.default(primals_248, [1, 0])
        addmm_78: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_249, view_206, permute_103);  primals_249 = permute_103 = None
        view_207: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_78, [256, 128, 128]);  addmm_78 = None
        view_208: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_207, [256, 128, -1, 32]);  view_207 = None
        permute_104: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_105: "f32[128, 128]" = torch.ops.aten.permute.default(primals_250, [1, 0])
        addmm_79: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_251, view_206, permute_105);  primals_251 = permute_105 = None
        view_210: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_79, [256, 128, 128]);  addmm_79 = None
        view_211: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_210, [256, 128, -1, 32]);  view_210 = None
        permute_106: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_107: "f32[512, 128]" = torch.ops.aten.permute.default(primals_252, [1, 0])
        addmm_80: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_253, view_202, permute_107);  primals_253 = permute_107 = None
        view_213: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_80, [256, 128, 128]);  addmm_80 = None
        view_214: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_213, [256, 128, -1, 32]);  view_213 = None
        permute_108: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_63: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_104, 0.4204482076268573);  permute_104 = None
        permute_109: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_106, [0, 1, 3, 2]);  permute_106 = None
        mul_64: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_109, 0.4204482076268573);  permute_109 = None
        expand_21: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_63, [256, 4, 128, 32]);  mul_63 = None
        clone_26: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_215: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_26, [1024, 128, 32]);  clone_26 = None
        expand_22: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_64, [256, 4, 32, 128]);  mul_64 = None
        clone_27: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_216: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_27, [1024, 32, 128]);  clone_27 = None
        bmm_10: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_215, view_216)
        view_217: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_10, [256, 4, 128, 128]);  bmm_10 = None
        add_82: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_217, where);  view_217 = None
        amax_5: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_82, [-1], True)
        sub_5: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_82, amax_5);  amax_5 = None
        exp_5: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_6: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        eq_5: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_82, -inf);  add_82 = None
        logical_not_10: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        where_11: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_11, full_default_3, div_5);  logical_not_11 = div_5 = None
        inductor_lookup_seed_default_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_18: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        gt_5: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_18, 0.1);  inductor_random_default_18 = None
        mul_65: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_5, where_11)
        mul_66: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_65, 1.1111111111111112);  mul_65 = None
        expand_23: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_66, [256, 4, 128, 128]);  mul_66 = None
        view_218: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_23, [1024, 128, 128]);  expand_23 = None
        expand_24: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_108, [256, 4, 128, 32]);  permute_108 = None
        clone_28: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_219: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_28, [1024, 128, 32]);  clone_28 = None
        bmm_11: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_218, view_219)
        view_220: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_11, [256, 4, 128, 32]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_110: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None
        clone_29: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_221: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_29, [256, 128, -1]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_222: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_221, [32768, 128]);  view_221 = None
        permute_111: "f32[128, 128]" = torch.ops.aten.permute.default(primals_254, [1, 0])
        addmm_81: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_255, view_222, permute_111);  primals_255 = permute_111 = None
        view_223: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_81, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_83: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_223, add_80);  view_223 = add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_67: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_83, primals_256);  add_83 = None
        add_84: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_67, primals_257);  mul_67 = primals_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_224: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_84, [32768, 128])
        permute_112: "f32[128, 512]" = torch.ops.aten.permute.default(primals_258, [1, 0])
        addmm_82: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_259, view_224, permute_112);  primals_259 = permute_112 = None
        view_225: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_82, [256, 128, 512]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_20: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_225);  view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_226: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_20, [32768, 512])
        permute_113: "f32[512, 128]" = torch.ops.aten.permute.default(primals_260, [1, 0])
        addmm_83: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_261, view_226, permute_113);  primals_261 = permute_113 = None
        view_227: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_83, [256, 128, 128]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_85: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_227, add_84);  view_227 = add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_68: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_85, primals_262)
        add_86: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_68, primals_263);  mul_68 = primals_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_228: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_86, [32768, 128])
        permute_114: "f32[128, 512]" = torch.ops.aten.permute.default(primals_264, [1, 0])
        addmm_84: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_265, view_228, permute_114);  primals_265 = permute_114 = None
        view_229: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_84, [256, 128, 512]);  addmm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_21: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_229);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_230: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_21, [32768, 512])
        permute_115: "f32[512, 128]" = torch.ops.aten.permute.default(primals_266, [1, 0])
        addmm_85: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_267, view_230, permute_115);  primals_267 = permute_115 = None
        view_231: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_85, [256, 128, 128]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_87: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_231, add_86);  view_231 = add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_69: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_87, primals_268)
        add_88: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_69, primals_269);  mul_69 = primals_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_232: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_88, [32768, 128])
        permute_116: "f32[128, 512]" = torch.ops.aten.permute.default(primals_270, [1, 0])
        addmm_86: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_271, view_232, permute_116);  primals_271 = permute_116 = None
        view_233: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_86, [256, 128, 512]);  addmm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_22: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_233);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_234: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_22, [32768, 512])
        permute_117: "f32[512, 128]" = torch.ops.aten.permute.default(primals_272, [1, 0])
        addmm_87: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_273, view_234, permute_117);  primals_273 = permute_117 = None
        view_235: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_87, [256, 128, 128]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_89: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_235, add_88);  view_235 = add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_70: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_89, primals_274)
        add_90: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_70, primals_275);  mul_70 = primals_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_236: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_90, [32768, 128])
        permute_118: "f32[128, 512]" = torch.ops.aten.permute.default(primals_276, [1, 0])
        addmm_88: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_277, view_236, permute_118);  primals_277 = permute_118 = None
        view_237: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_88, [256, 128, 512]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_23: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_237);  view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_238: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_23, [32768, 512])
        permute_119: "f32[512, 128]" = torch.ops.aten.permute.default(primals_278, [1, 0])
        addmm_89: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_279, view_238, permute_119);  primals_279 = permute_119 = None
        view_239: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_89, [256, 128, 128]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_91: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_239, add_90);  view_239 = add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_71: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_91, primals_280)
        add_92: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_71, primals_281);  mul_71 = primals_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_240: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_92, [32768, 128]);  add_92 = None
        permute_120: "f32[128, 512]" = torch.ops.aten.permute.default(primals_282, [1, 0])
        addmm_90: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_283, view_240, permute_120);  primals_283 = permute_120 = None
        view_241: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_90, [256, 128, 512]);  addmm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_93: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_241, add_79);  view_241 = add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_72: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_93, primals_284)
        add_94: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_72, primals_285);  mul_72 = primals_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_242: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_94, [32768, 512])
        permute_121: "f32[512, 128]" = torch.ops.aten.permute.default(primals_286, [1, 0])
        addmm_91: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_287, view_242, permute_121);  primals_287 = permute_121 = None
        view_243: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_91, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_73: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_243, primals_288);  view_243 = None
        add_95: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_73, primals_289);  mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_122: "f32[512, 128]" = torch.ops.aten.permute.default(primals_290, [1, 0])
        addmm_92: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_291, view_242, permute_122);  primals_291 = permute_122 = None
        view_245: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_92, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_74: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_245, primals_292);  view_245 = None
        add_96: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_74, primals_293);  mul_74 = primals_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_246: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_96, [32768, 128]);  add_96 = None
        permute_123: "f32[128, 128]" = torch.ops.aten.permute.default(primals_294, [1, 0])
        addmm_93: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_295, view_246, permute_123);  primals_295 = permute_123 = None
        view_247: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_93, [256, 128, 128]);  addmm_93 = None
        view_248: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_247, [256, 128, -1, 32]);  view_247 = None
        permute_124: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_125: "f32[128, 128]" = torch.ops.aten.permute.default(primals_296, [1, 0])
        addmm_94: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_297, view_246, permute_125);  primals_297 = permute_125 = None
        view_250: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_94, [256, 128, 128]);  addmm_94 = None
        view_251: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_250, [256, 128, -1, 32]);  view_250 = None
        permute_126: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_127: "f32[512, 128]" = torch.ops.aten.permute.default(primals_298, [1, 0])
        addmm_95: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_299, view_242, permute_127);  primals_299 = permute_127 = None
        view_253: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_95, [256, 128, 128]);  addmm_95 = None
        view_254: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_253, [256, 128, -1, 32]);  view_253 = None
        permute_128: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_254, [0, 2, 1, 3]);  view_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_75: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_124, 0.4204482076268573);  permute_124 = None
        permute_129: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_126, [0, 1, 3, 2]);  permute_126 = None
        mul_76: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_129, 0.4204482076268573);  permute_129 = None
        expand_25: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_75, [256, 4, 128, 32]);  mul_75 = None
        clone_31: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_255: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_31, [1024, 128, 32]);  clone_31 = None
        expand_26: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_76, [256, 4, 32, 128]);  mul_76 = None
        clone_32: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_256: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_32, [1024, 32, 128]);  clone_32 = None
        bmm_12: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_255, view_256)
        view_257: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_12, [256, 4, 128, 128]);  bmm_12 = None
        add_97: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_257, where);  view_257 = None
        amax_6: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_97, [-1], True)
        sub_6: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_97, amax_6);  amax_6 = None
        exp_6: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_7: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        eq_6: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_97, -inf);  add_97 = None
        logical_not_12: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        where_13: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_13, full_default_3, div_6);  logical_not_13 = div_6 = None
        inductor_lookup_seed_default_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_17: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_6: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_17, 0.1);  inductor_random_default_17 = None
        mul_77: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_6, where_13)
        mul_78: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_77, 1.1111111111111112);  mul_77 = None
        expand_27: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_78, [256, 4, 128, 128]);  mul_78 = None
        view_258: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_27, [1024, 128, 128]);  expand_27 = None
        expand_28: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_128, [256, 4, 128, 32]);  permute_128 = None
        clone_33: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_259: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_33, [1024, 128, 32]);  clone_33 = None
        bmm_13: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_258, view_259)
        view_260: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_13, [256, 4, 128, 32]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_130: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_260, [0, 2, 1, 3]);  view_260 = None
        clone_34: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_261: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_34, [256, 128, -1]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_262: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_261, [32768, 128]);  view_261 = None
        permute_131: "f32[128, 128]" = torch.ops.aten.permute.default(primals_300, [1, 0])
        addmm_96: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_301, view_262, permute_131);  primals_301 = permute_131 = None
        view_263: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_96, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_98: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_263, add_95);  view_263 = add_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_79: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_98, primals_302);  add_98 = None
        add_99: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_79, primals_303);  mul_79 = primals_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_264: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_99, [32768, 128])
        permute_132: "f32[128, 512]" = torch.ops.aten.permute.default(primals_304, [1, 0])
        addmm_97: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_305, view_264, permute_132);  primals_305 = permute_132 = None
        view_265: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_97, [256, 128, 512]);  addmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_24: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_265);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_266: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_24, [32768, 512])
        permute_133: "f32[512, 128]" = torch.ops.aten.permute.default(primals_306, [1, 0])
        addmm_98: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_307, view_266, permute_133);  primals_307 = permute_133 = None
        view_267: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_98, [256, 128, 128]);  addmm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_100: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_267, add_99);  view_267 = add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_80: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_100, primals_308)
        add_101: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_80, primals_309);  mul_80 = primals_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_268: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_101, [32768, 128])
        permute_134: "f32[128, 512]" = torch.ops.aten.permute.default(primals_310, [1, 0])
        addmm_99: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_311, view_268, permute_134);  primals_311 = permute_134 = None
        view_269: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_99, [256, 128, 512]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_25: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_269);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_270: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_25, [32768, 512])
        permute_135: "f32[512, 128]" = torch.ops.aten.permute.default(primals_312, [1, 0])
        addmm_100: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_313, view_270, permute_135);  primals_313 = permute_135 = None
        view_271: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_100, [256, 128, 128]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_102: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_271, add_101);  view_271 = add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_81: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_102, primals_314)
        add_103: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_81, primals_315);  mul_81 = primals_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_272: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_103, [32768, 128])
        permute_136: "f32[128, 512]" = torch.ops.aten.permute.default(primals_316, [1, 0])
        addmm_101: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_317, view_272, permute_136);  primals_317 = permute_136 = None
        view_273: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_101, [256, 128, 512]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_26: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_273);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_274: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_26, [32768, 512])
        permute_137: "f32[512, 128]" = torch.ops.aten.permute.default(primals_318, [1, 0])
        addmm_102: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_319, view_274, permute_137);  primals_319 = permute_137 = None
        view_275: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_102, [256, 128, 128]);  addmm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_104: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_275, add_103);  view_275 = add_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_82: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_104, primals_320)
        add_105: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_82, primals_321);  mul_82 = primals_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_276: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_105, [32768, 128])
        permute_138: "f32[128, 512]" = torch.ops.aten.permute.default(primals_322, [1, 0])
        addmm_103: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_323, view_276, permute_138);  primals_323 = permute_138 = None
        view_277: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_103, [256, 128, 512]);  addmm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_27: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_277);  view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_278: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_27, [32768, 512])
        permute_139: "f32[512, 128]" = torch.ops.aten.permute.default(primals_324, [1, 0])
        addmm_104: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_325, view_278, permute_139);  primals_325 = permute_139 = None
        view_279: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_104, [256, 128, 128]);  addmm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_106: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_279, add_105);  view_279 = add_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_83: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_106, primals_326)
        add_107: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_83, primals_327);  mul_83 = primals_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_280: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_107, [32768, 128]);  add_107 = None
        permute_140: "f32[128, 512]" = torch.ops.aten.permute.default(primals_328, [1, 0])
        addmm_105: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_329, view_280, permute_140);  primals_329 = permute_140 = None
        view_281: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_105, [256, 128, 512]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_108: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_281, add_94);  view_281 = add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_84: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_108, primals_330)
        add_109: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_84, primals_331);  mul_84 = primals_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_282: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_109, [32768, 512])
        permute_141: "f32[512, 128]" = torch.ops.aten.permute.default(primals_332, [1, 0])
        addmm_106: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_333, view_282, permute_141);  primals_333 = permute_141 = None
        view_283: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_106, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_85: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_283, primals_334);  view_283 = None
        add_110: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_85, primals_335);  mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_142: "f32[512, 128]" = torch.ops.aten.permute.default(primals_336, [1, 0])
        addmm_107: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_337, view_282, permute_142);  primals_337 = permute_142 = None
        view_285: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_107, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_86: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_285, primals_338);  view_285 = None
        add_111: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_86, primals_339);  mul_86 = primals_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_286: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_111, [32768, 128]);  add_111 = None
        permute_143: "f32[128, 128]" = torch.ops.aten.permute.default(primals_340, [1, 0])
        addmm_108: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_341, view_286, permute_143);  primals_341 = permute_143 = None
        view_287: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_108, [256, 128, 128]);  addmm_108 = None
        view_288: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_287, [256, 128, -1, 32]);  view_287 = None
        permute_144: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_145: "f32[128, 128]" = torch.ops.aten.permute.default(primals_342, [1, 0])
        addmm_109: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_343, view_286, permute_145);  primals_343 = permute_145 = None
        view_290: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_109, [256, 128, 128]);  addmm_109 = None
        view_291: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_290, [256, 128, -1, 32]);  view_290 = None
        permute_146: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_147: "f32[512, 128]" = torch.ops.aten.permute.default(primals_344, [1, 0])
        addmm_110: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_345, view_282, permute_147);  primals_345 = permute_147 = None
        view_293: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_110, [256, 128, 128]);  addmm_110 = None
        view_294: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_293, [256, 128, -1, 32]);  view_293 = None
        permute_148: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_87: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_144, 0.4204482076268573);  permute_144 = None
        permute_149: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_146, [0, 1, 3, 2]);  permute_146 = None
        mul_88: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_149, 0.4204482076268573);  permute_149 = None
        expand_29: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_87, [256, 4, 128, 32]);  mul_87 = None
        clone_36: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_295: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_36, [1024, 128, 32]);  clone_36 = None
        expand_30: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_88, [256, 4, 32, 128]);  mul_88 = None
        clone_37: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_296: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_37, [1024, 32, 128]);  clone_37 = None
        bmm_14: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_295, view_296)
        view_297: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_14, [256, 4, 128, 128]);  bmm_14 = None
        add_112: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_297, where);  view_297 = None
        amax_7: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_112, [-1], True)
        sub_7: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_112, amax_7);  amax_7 = None
        exp_7: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_8: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        eq_7: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_112, -inf);  add_112 = None
        logical_not_14: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        where_15: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_15, full_default_3, div_7);  logical_not_15 = div_7 = None
        inductor_lookup_seed_default_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_16: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        gt_7: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_16, 0.1);  inductor_random_default_16 = None
        mul_89: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_7, where_15)
        mul_90: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_89, 1.1111111111111112);  mul_89 = None
        expand_31: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_90, [256, 4, 128, 128]);  mul_90 = None
        view_298: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_31, [1024, 128, 128]);  expand_31 = None
        expand_32: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_148, [256, 4, 128, 32]);  permute_148 = None
        clone_38: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_299: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_38, [1024, 128, 32]);  clone_38 = None
        bmm_15: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_298, view_299)
        view_300: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_15, [256, 4, 128, 32]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_150: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
        clone_39: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_301: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_39, [256, 128, -1]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_302: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_301, [32768, 128]);  view_301 = None
        permute_151: "f32[128, 128]" = torch.ops.aten.permute.default(primals_346, [1, 0])
        addmm_111: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_347, view_302, permute_151);  primals_347 = permute_151 = None
        view_303: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_111, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_113: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_303, add_110);  view_303 = add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_91: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_113, primals_348);  add_113 = None
        add_114: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_91, primals_349);  mul_91 = primals_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_304: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_114, [32768, 128])
        permute_152: "f32[128, 512]" = torch.ops.aten.permute.default(primals_350, [1, 0])
        addmm_112: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_351, view_304, permute_152);  primals_351 = permute_152 = None
        view_305: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_112, [256, 128, 512]);  addmm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_28: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_305);  view_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_306: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_28, [32768, 512])
        permute_153: "f32[512, 128]" = torch.ops.aten.permute.default(primals_352, [1, 0])
        addmm_113: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_353, view_306, permute_153);  primals_353 = permute_153 = None
        view_307: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_113, [256, 128, 128]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_115: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_307, add_114);  view_307 = add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_92: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_115, primals_354)
        add_116: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_92, primals_355);  mul_92 = primals_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_308: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_116, [32768, 128])
        permute_154: "f32[128, 512]" = torch.ops.aten.permute.default(primals_356, [1, 0])
        addmm_114: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_357, view_308, permute_154);  primals_357 = permute_154 = None
        view_309: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_114, [256, 128, 512]);  addmm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_29: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_309);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_310: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_29, [32768, 512])
        permute_155: "f32[512, 128]" = torch.ops.aten.permute.default(primals_358, [1, 0])
        addmm_115: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_359, view_310, permute_155);  primals_359 = permute_155 = None
        view_311: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_115, [256, 128, 128]);  addmm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_117: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_311, add_116);  view_311 = add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_93: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_117, primals_360)
        add_118: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_93, primals_361);  mul_93 = primals_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_312: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_118, [32768, 128])
        permute_156: "f32[128, 512]" = torch.ops.aten.permute.default(primals_362, [1, 0])
        addmm_116: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_363, view_312, permute_156);  primals_363 = permute_156 = None
        view_313: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_116, [256, 128, 512]);  addmm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_30: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_313);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_314: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_30, [32768, 512])
        permute_157: "f32[512, 128]" = torch.ops.aten.permute.default(primals_364, [1, 0])
        addmm_117: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_365, view_314, permute_157);  primals_365 = permute_157 = None
        view_315: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_117, [256, 128, 128]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_119: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_315, add_118);  view_315 = add_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_94: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_119, primals_366)
        add_120: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_94, primals_367);  mul_94 = primals_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_316: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_120, [32768, 128])
        permute_158: "f32[128, 512]" = torch.ops.aten.permute.default(primals_368, [1, 0])
        addmm_118: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_369, view_316, permute_158);  primals_369 = permute_158 = None
        view_317: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_118, [256, 128, 512]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_31: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_317);  view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_318: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_31, [32768, 512])
        permute_159: "f32[512, 128]" = torch.ops.aten.permute.default(primals_370, [1, 0])
        addmm_119: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_371, view_318, permute_159);  primals_371 = permute_159 = None
        view_319: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_119, [256, 128, 128]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_121: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_319, add_120);  view_319 = add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_95: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_121, primals_372)
        add_122: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_95, primals_373);  mul_95 = primals_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_320: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_122, [32768, 128]);  add_122 = None
        permute_160: "f32[128, 512]" = torch.ops.aten.permute.default(primals_374, [1, 0])
        addmm_120: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_375, view_320, permute_160);  primals_375 = permute_160 = None
        view_321: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_120, [256, 128, 512]);  addmm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_123: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_321, add_109);  view_321 = add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_96: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_123, primals_376)
        add_124: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_96, primals_377);  mul_96 = primals_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_322: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_124, [32768, 512])
        permute_161: "f32[512, 128]" = torch.ops.aten.permute.default(primals_378, [1, 0])
        addmm_121: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_379, view_322, permute_161);  primals_379 = permute_161 = None
        view_323: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_121, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_97: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_323, primals_380);  view_323 = None
        add_125: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_97, primals_381);  mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_162: "f32[512, 128]" = torch.ops.aten.permute.default(primals_382, [1, 0])
        addmm_122: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_383, view_322, permute_162);  primals_383 = permute_162 = None
        view_325: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_122, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_98: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_325, primals_384);  view_325 = None
        add_126: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_98, primals_385);  mul_98 = primals_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_326: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_126, [32768, 128]);  add_126 = None
        permute_163: "f32[128, 128]" = torch.ops.aten.permute.default(primals_386, [1, 0])
        addmm_123: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_387, view_326, permute_163);  primals_387 = permute_163 = None
        view_327: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_123, [256, 128, 128]);  addmm_123 = None
        view_328: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_327, [256, 128, -1, 32]);  view_327 = None
        permute_164: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_165: "f32[128, 128]" = torch.ops.aten.permute.default(primals_388, [1, 0])
        addmm_124: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_389, view_326, permute_165);  primals_389 = permute_165 = None
        view_330: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_124, [256, 128, 128]);  addmm_124 = None
        view_331: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_330, [256, 128, -1, 32]);  view_330 = None
        permute_166: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_167: "f32[512, 128]" = torch.ops.aten.permute.default(primals_390, [1, 0])
        addmm_125: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_391, view_322, permute_167);  primals_391 = permute_167 = None
        view_333: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_125, [256, 128, 128]);  addmm_125 = None
        view_334: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_333, [256, 128, -1, 32]);  view_333 = None
        permute_168: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_99: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_164, 0.4204482076268573);  permute_164 = None
        permute_169: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_166, [0, 1, 3, 2]);  permute_166 = None
        mul_100: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_169, 0.4204482076268573);  permute_169 = None
        expand_33: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_99, [256, 4, 128, 32]);  mul_99 = None
        clone_41: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_335: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_41, [1024, 128, 32]);  clone_41 = None
        expand_34: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_100, [256, 4, 32, 128]);  mul_100 = None
        clone_42: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_336: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_42, [1024, 32, 128]);  clone_42 = None
        bmm_16: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_335, view_336)
        view_337: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_16, [256, 4, 128, 128]);  bmm_16 = None
        add_127: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_337, where);  view_337 = None
        amax_8: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_127, [-1], True)
        sub_8: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_127, amax_8);  amax_8 = None
        exp_8: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_9: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        eq_8: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_127, -inf);  add_127 = None
        logical_not_16: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        where_17: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_17, full_default_3, div_8);  logical_not_17 = div_8 = None
        inductor_lookup_seed_default_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_15: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        gt_8: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_15, 0.1);  inductor_random_default_15 = None
        mul_101: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_8, where_17)
        mul_102: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_101, 1.1111111111111112);  mul_101 = None
        expand_35: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_102, [256, 4, 128, 128]);  mul_102 = None
        view_338: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_35, [1024, 128, 128]);  expand_35 = None
        expand_36: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_168, [256, 4, 128, 32]);  permute_168 = None
        clone_43: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_339: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_43, [1024, 128, 32]);  clone_43 = None
        bmm_17: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_338, view_339)
        view_340: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_17, [256, 4, 128, 32]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_170: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
        clone_44: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_341: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_44, [256, 128, -1]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_342: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_341, [32768, 128]);  view_341 = None
        permute_171: "f32[128, 128]" = torch.ops.aten.permute.default(primals_392, [1, 0])
        addmm_126: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_393, view_342, permute_171);  primals_393 = permute_171 = None
        view_343: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_126, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_128: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_343, add_125);  view_343 = add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_103: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_128, primals_394);  add_128 = None
        add_129: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_103, primals_395);  mul_103 = primals_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_344: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_129, [32768, 128])
        permute_172: "f32[128, 512]" = torch.ops.aten.permute.default(primals_396, [1, 0])
        addmm_127: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_397, view_344, permute_172);  primals_397 = permute_172 = None
        view_345: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_127, [256, 128, 512]);  addmm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_32: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_345);  view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_346: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_32, [32768, 512])
        permute_173: "f32[512, 128]" = torch.ops.aten.permute.default(primals_398, [1, 0])
        addmm_128: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_399, view_346, permute_173);  primals_399 = permute_173 = None
        view_347: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_128, [256, 128, 128]);  addmm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_130: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_347, add_129);  view_347 = add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_104: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_130, primals_400)
        add_131: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_104, primals_401);  mul_104 = primals_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_348: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_131, [32768, 128])
        permute_174: "f32[128, 512]" = torch.ops.aten.permute.default(primals_402, [1, 0])
        addmm_129: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_403, view_348, permute_174);  primals_403 = permute_174 = None
        view_349: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_129, [256, 128, 512]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_33: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_349);  view_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_350: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_33, [32768, 512])
        permute_175: "f32[512, 128]" = torch.ops.aten.permute.default(primals_404, [1, 0])
        addmm_130: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_405, view_350, permute_175);  primals_405 = permute_175 = None
        view_351: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_130, [256, 128, 128]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_132: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_351, add_131);  view_351 = add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_105: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_132, primals_406)
        add_133: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_105, primals_407);  mul_105 = primals_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_352: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_133, [32768, 128])
        permute_176: "f32[128, 512]" = torch.ops.aten.permute.default(primals_408, [1, 0])
        addmm_131: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_409, view_352, permute_176);  primals_409 = permute_176 = None
        view_353: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_131, [256, 128, 512]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_34: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_353);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_354: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_34, [32768, 512])
        permute_177: "f32[512, 128]" = torch.ops.aten.permute.default(primals_410, [1, 0])
        addmm_132: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_411, view_354, permute_177);  primals_411 = permute_177 = None
        view_355: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_132, [256, 128, 128]);  addmm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_134: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_355, add_133);  view_355 = add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_106: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_134, primals_412)
        add_135: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_106, primals_413);  mul_106 = primals_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_356: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_135, [32768, 128])
        permute_178: "f32[128, 512]" = torch.ops.aten.permute.default(primals_414, [1, 0])
        addmm_133: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_415, view_356, permute_178);  primals_415 = permute_178 = None
        view_357: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_133, [256, 128, 512]);  addmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_35: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_357);  view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_358: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_35, [32768, 512])
        permute_179: "f32[512, 128]" = torch.ops.aten.permute.default(primals_416, [1, 0])
        addmm_134: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_417, view_358, permute_179);  primals_417 = permute_179 = None
        view_359: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_134, [256, 128, 128]);  addmm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_136: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_359, add_135);  view_359 = add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_107: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_136, primals_418)
        add_137: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_107, primals_419);  mul_107 = primals_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_360: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_137, [32768, 128]);  add_137 = None
        permute_180: "f32[128, 512]" = torch.ops.aten.permute.default(primals_420, [1, 0])
        addmm_135: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_421, view_360, permute_180);  primals_421 = permute_180 = None
        view_361: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_135, [256, 128, 512]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_138: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_361, add_124);  view_361 = add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_108: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_138, primals_422)
        add_139: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_108, primals_423);  mul_108 = primals_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_362: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_139, [32768, 512])
        permute_181: "f32[512, 128]" = torch.ops.aten.permute.default(primals_424, [1, 0])
        addmm_136: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_425, view_362, permute_181);  primals_425 = permute_181 = None
        view_363: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_136, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_109: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_363, primals_426);  view_363 = None
        add_140: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_109, primals_427);  mul_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_182: "f32[512, 128]" = torch.ops.aten.permute.default(primals_428, [1, 0])
        addmm_137: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_429, view_362, permute_182);  primals_429 = permute_182 = None
        view_365: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_137, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_110: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_365, primals_430);  view_365 = None
        add_141: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_110, primals_431);  mul_110 = primals_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_366: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_141, [32768, 128]);  add_141 = None
        permute_183: "f32[128, 128]" = torch.ops.aten.permute.default(primals_432, [1, 0])
        addmm_138: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_433, view_366, permute_183);  primals_433 = permute_183 = None
        view_367: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_138, [256, 128, 128]);  addmm_138 = None
        view_368: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_367, [256, 128, -1, 32]);  view_367 = None
        permute_184: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_185: "f32[128, 128]" = torch.ops.aten.permute.default(primals_434, [1, 0])
        addmm_139: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_435, view_366, permute_185);  primals_435 = permute_185 = None
        view_370: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_139, [256, 128, 128]);  addmm_139 = None
        view_371: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_370, [256, 128, -1, 32]);  view_370 = None
        permute_186: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_187: "f32[512, 128]" = torch.ops.aten.permute.default(primals_436, [1, 0])
        addmm_140: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_437, view_362, permute_187);  primals_437 = permute_187 = None
        view_373: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_140, [256, 128, 128]);  addmm_140 = None
        view_374: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_373, [256, 128, -1, 32]);  view_373 = None
        permute_188: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_111: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_184, 0.4204482076268573);  permute_184 = None
        permute_189: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_186, [0, 1, 3, 2]);  permute_186 = None
        mul_112: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_189, 0.4204482076268573);  permute_189 = None
        expand_37: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_111, [256, 4, 128, 32]);  mul_111 = None
        clone_46: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_375: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_46, [1024, 128, 32]);  clone_46 = None
        expand_38: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_112, [256, 4, 32, 128]);  mul_112 = None
        clone_47: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_376: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_47, [1024, 32, 128]);  clone_47 = None
        bmm_18: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_375, view_376)
        view_377: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_18, [256, 4, 128, 128]);  bmm_18 = None
        add_142: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_377, where);  view_377 = None
        amax_9: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_142, [-1], True)
        sub_9: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_142, amax_9);  amax_9 = None
        exp_9: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_10: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        eq_9: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_142, -inf);  add_142 = None
        logical_not_18: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        where_19: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_19, full_default_3, div_9);  logical_not_19 = div_9 = None
        inductor_lookup_seed_default_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_14: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        gt_9: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_14, 0.1);  inductor_random_default_14 = None
        mul_113: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_9, where_19)
        mul_114: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_113, 1.1111111111111112);  mul_113 = None
        expand_39: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_114, [256, 4, 128, 128]);  mul_114 = None
        view_378: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_39, [1024, 128, 128]);  expand_39 = None
        expand_40: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_188, [256, 4, 128, 32]);  permute_188 = None
        clone_48: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_379: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_48, [1024, 128, 32]);  clone_48 = None
        bmm_19: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_378, view_379)
        view_380: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_19, [256, 4, 128, 32]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_190: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_380, [0, 2, 1, 3]);  view_380 = None
        clone_49: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_381: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_49, [256, 128, -1]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_382: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_381, [32768, 128]);  view_381 = None
        permute_191: "f32[128, 128]" = torch.ops.aten.permute.default(primals_438, [1, 0])
        addmm_141: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_439, view_382, permute_191);  primals_439 = permute_191 = None
        view_383: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_141, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_143: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_383, add_140);  view_383 = add_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_115: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_143, primals_440);  add_143 = None
        add_144: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_115, primals_441);  mul_115 = primals_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_384: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_144, [32768, 128])
        permute_192: "f32[128, 512]" = torch.ops.aten.permute.default(primals_442, [1, 0])
        addmm_142: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_443, view_384, permute_192);  primals_443 = permute_192 = None
        view_385: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_142, [256, 128, 512]);  addmm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_36: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_385);  view_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_386: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_36, [32768, 512])
        permute_193: "f32[512, 128]" = torch.ops.aten.permute.default(primals_444, [1, 0])
        addmm_143: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_445, view_386, permute_193);  primals_445 = permute_193 = None
        view_387: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_143, [256, 128, 128]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_145: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_387, add_144);  view_387 = add_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_116: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_145, primals_446)
        add_146: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_116, primals_447);  mul_116 = primals_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_388: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_146, [32768, 128])
        permute_194: "f32[128, 512]" = torch.ops.aten.permute.default(primals_448, [1, 0])
        addmm_144: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_449, view_388, permute_194);  primals_449 = permute_194 = None
        view_389: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_144, [256, 128, 512]);  addmm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_37: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_389);  view_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_390: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_37, [32768, 512])
        permute_195: "f32[512, 128]" = torch.ops.aten.permute.default(primals_450, [1, 0])
        addmm_145: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_451, view_390, permute_195);  primals_451 = permute_195 = None
        view_391: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_145, [256, 128, 128]);  addmm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_147: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_391, add_146);  view_391 = add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_117: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_147, primals_452)
        add_148: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_117, primals_453);  mul_117 = primals_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_392: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_148, [32768, 128])
        permute_196: "f32[128, 512]" = torch.ops.aten.permute.default(primals_454, [1, 0])
        addmm_146: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_455, view_392, permute_196);  primals_455 = permute_196 = None
        view_393: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_146, [256, 128, 512]);  addmm_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_38: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_393);  view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_394: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_38, [32768, 512])
        permute_197: "f32[512, 128]" = torch.ops.aten.permute.default(primals_456, [1, 0])
        addmm_147: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_457, view_394, permute_197);  primals_457 = permute_197 = None
        view_395: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_147, [256, 128, 128]);  addmm_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_149: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_395, add_148);  view_395 = add_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_118: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_149, primals_458)
        add_150: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_118, primals_459);  mul_118 = primals_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_396: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_150, [32768, 128])
        permute_198: "f32[128, 512]" = torch.ops.aten.permute.default(primals_460, [1, 0])
        addmm_148: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_461, view_396, permute_198);  primals_461 = permute_198 = None
        view_397: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_148, [256, 128, 512]);  addmm_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_39: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_397);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_398: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_39, [32768, 512])
        permute_199: "f32[512, 128]" = torch.ops.aten.permute.default(primals_462, [1, 0])
        addmm_149: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_463, view_398, permute_199);  primals_463 = permute_199 = None
        view_399: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_149, [256, 128, 128]);  addmm_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_151: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_399, add_150);  view_399 = add_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_119: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_151, primals_464)
        add_152: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_119, primals_465);  mul_119 = primals_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_400: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_152, [32768, 128]);  add_152 = None
        permute_200: "f32[128, 512]" = torch.ops.aten.permute.default(primals_466, [1, 0])
        addmm_150: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_467, view_400, permute_200);  primals_467 = permute_200 = None
        view_401: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_150, [256, 128, 512]);  addmm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_153: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_401, add_139);  view_401 = add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_120: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_153, primals_468)
        add_154: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_120, primals_469);  mul_120 = primals_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_402: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_154, [32768, 512])
        permute_201: "f32[512, 128]" = torch.ops.aten.permute.default(primals_470, [1, 0])
        addmm_151: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_471, view_402, permute_201);  primals_471 = permute_201 = None
        view_403: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_151, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_121: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_403, primals_472);  view_403 = None
        add_155: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_121, primals_473);  mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_202: "f32[512, 128]" = torch.ops.aten.permute.default(primals_474, [1, 0])
        addmm_152: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_475, view_402, permute_202);  primals_475 = permute_202 = None
        view_405: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_152, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_122: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_405, primals_476);  view_405 = None
        add_156: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_122, primals_477);  mul_122 = primals_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_406: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_156, [32768, 128]);  add_156 = None
        permute_203: "f32[128, 128]" = torch.ops.aten.permute.default(primals_478, [1, 0])
        addmm_153: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_479, view_406, permute_203);  primals_479 = permute_203 = None
        view_407: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_153, [256, 128, 128]);  addmm_153 = None
        view_408: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_407, [256, 128, -1, 32]);  view_407 = None
        permute_204: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_408, [0, 2, 1, 3]);  view_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_205: "f32[128, 128]" = torch.ops.aten.permute.default(primals_480, [1, 0])
        addmm_154: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_481, view_406, permute_205);  primals_481 = permute_205 = None
        view_410: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_154, [256, 128, 128]);  addmm_154 = None
        view_411: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_410, [256, 128, -1, 32]);  view_410 = None
        permute_206: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_411, [0, 2, 1, 3]);  view_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_207: "f32[512, 128]" = torch.ops.aten.permute.default(primals_482, [1, 0])
        addmm_155: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_483, view_402, permute_207);  primals_483 = permute_207 = None
        view_413: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_155, [256, 128, 128]);  addmm_155 = None
        view_414: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_413, [256, 128, -1, 32]);  view_413 = None
        permute_208: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_123: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_204, 0.4204482076268573);  permute_204 = None
        permute_209: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 2]);  permute_206 = None
        mul_124: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_209, 0.4204482076268573);  permute_209 = None
        expand_41: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_123, [256, 4, 128, 32]);  mul_123 = None
        clone_51: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_415: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_51, [1024, 128, 32]);  clone_51 = None
        expand_42: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_124, [256, 4, 32, 128]);  mul_124 = None
        clone_52: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_416: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_52, [1024, 32, 128]);  clone_52 = None
        bmm_20: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_415, view_416)
        view_417: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_20, [256, 4, 128, 128]);  bmm_20 = None
        add_157: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_417, where);  view_417 = None
        amax_10: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_157, [-1], True)
        sub_10: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_157, amax_10);  amax_10 = None
        exp_10: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_11: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        eq_10: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_157, -inf);  add_157 = None
        logical_not_20: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        where_21: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_21, full_default_3, div_10);  logical_not_21 = div_10 = None
        inductor_lookup_seed_default_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_13: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_10: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_13, 0.1);  inductor_random_default_13 = None
        mul_125: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_10, where_21)
        mul_126: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_125, 1.1111111111111112);  mul_125 = None
        expand_43: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_126, [256, 4, 128, 128]);  mul_126 = None
        view_418: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_43, [1024, 128, 128]);  expand_43 = None
        expand_44: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_208, [256, 4, 128, 32]);  permute_208 = None
        clone_53: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_419: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_53, [1024, 128, 32]);  clone_53 = None
        bmm_21: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_418, view_419)
        view_420: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_21, [256, 4, 128, 32]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_210: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
        clone_54: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_421: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_54, [256, 128, -1]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_422: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_421, [32768, 128]);  view_421 = None
        permute_211: "f32[128, 128]" = torch.ops.aten.permute.default(primals_484, [1, 0])
        addmm_156: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_485, view_422, permute_211);  primals_485 = permute_211 = None
        view_423: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_156, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_158: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_423, add_155);  view_423 = add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_127: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_158, primals_486);  add_158 = None
        add_159: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_127, primals_487);  mul_127 = primals_487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_424: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_159, [32768, 128])
        permute_212: "f32[128, 512]" = torch.ops.aten.permute.default(primals_488, [1, 0])
        addmm_157: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_489, view_424, permute_212);  primals_489 = permute_212 = None
        view_425: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_157, [256, 128, 512]);  addmm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_40: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_425);  view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_426: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_40, [32768, 512])
        permute_213: "f32[512, 128]" = torch.ops.aten.permute.default(primals_490, [1, 0])
        addmm_158: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_491, view_426, permute_213);  primals_491 = permute_213 = None
        view_427: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_158, [256, 128, 128]);  addmm_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_160: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_427, add_159);  view_427 = add_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_128: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_160, primals_492)
        add_161: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_128, primals_493);  mul_128 = primals_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_428: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_161, [32768, 128])
        permute_214: "f32[128, 512]" = torch.ops.aten.permute.default(primals_494, [1, 0])
        addmm_159: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_495, view_428, permute_214);  primals_495 = permute_214 = None
        view_429: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_159, [256, 128, 512]);  addmm_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_41: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_429);  view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_430: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_41, [32768, 512])
        permute_215: "f32[512, 128]" = torch.ops.aten.permute.default(primals_496, [1, 0])
        addmm_160: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_497, view_430, permute_215);  primals_497 = permute_215 = None
        view_431: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_160, [256, 128, 128]);  addmm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_162: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_431, add_161);  view_431 = add_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_129: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_162, primals_498)
        add_163: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_129, primals_499);  mul_129 = primals_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_432: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_163, [32768, 128])
        permute_216: "f32[128, 512]" = torch.ops.aten.permute.default(primals_500, [1, 0])
        addmm_161: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_501, view_432, permute_216);  primals_501 = permute_216 = None
        view_433: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_161, [256, 128, 512]);  addmm_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_42: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_433);  view_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_434: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_42, [32768, 512])
        permute_217: "f32[512, 128]" = torch.ops.aten.permute.default(primals_502, [1, 0])
        addmm_162: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_503, view_434, permute_217);  primals_503 = permute_217 = None
        view_435: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_162, [256, 128, 128]);  addmm_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_164: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_435, add_163);  view_435 = add_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_130: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_164, primals_504)
        add_165: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_130, primals_505);  mul_130 = primals_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_436: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_165, [32768, 128])
        permute_218: "f32[128, 512]" = torch.ops.aten.permute.default(primals_506, [1, 0])
        addmm_163: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_507, view_436, permute_218);  primals_507 = permute_218 = None
        view_437: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_163, [256, 128, 512]);  addmm_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_43: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_437);  view_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_438: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_43, [32768, 512])
        permute_219: "f32[512, 128]" = torch.ops.aten.permute.default(primals_508, [1, 0])
        addmm_164: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_509, view_438, permute_219);  primals_509 = permute_219 = None
        view_439: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_164, [256, 128, 128]);  addmm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_166: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_439, add_165);  view_439 = add_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_131: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_166, primals_510)
        add_167: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_131, primals_511);  mul_131 = primals_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_440: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_167, [32768, 128]);  add_167 = None
        permute_220: "f32[128, 512]" = torch.ops.aten.permute.default(primals_512, [1, 0])
        addmm_165: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_513, view_440, permute_220);  primals_513 = permute_220 = None
        view_441: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_165, [256, 128, 512]);  addmm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_168: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_441, add_154);  view_441 = add_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_132: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_168, primals_514)
        add_169: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_132, primals_515);  mul_132 = primals_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_442: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_169, [32768, 512])
        permute_221: "f32[512, 128]" = torch.ops.aten.permute.default(primals_516, [1, 0])
        addmm_166: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_517, view_442, permute_221);  primals_517 = permute_221 = None
        view_443: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_166, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_133: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_443, primals_518);  view_443 = None
        add_170: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_133, primals_519);  mul_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_222: "f32[512, 128]" = torch.ops.aten.permute.default(primals_520, [1, 0])
        addmm_167: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_521, view_442, permute_222);  primals_521 = permute_222 = None
        view_445: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_167, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_134: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_445, primals_522);  view_445 = None
        add_171: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_134, primals_523);  mul_134 = primals_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_446: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_171, [32768, 128]);  add_171 = None
        permute_223: "f32[128, 128]" = torch.ops.aten.permute.default(primals_524, [1, 0])
        addmm_168: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_525, view_446, permute_223);  primals_525 = permute_223 = None
        view_447: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_168, [256, 128, 128]);  addmm_168 = None
        view_448: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_447, [256, 128, -1, 32]);  view_447 = None
        permute_224: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_225: "f32[128, 128]" = torch.ops.aten.permute.default(primals_526, [1, 0])
        addmm_169: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_527, view_446, permute_225);  primals_527 = permute_225 = None
        view_450: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_169, [256, 128, 128]);  addmm_169 = None
        view_451: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_450, [256, 128, -1, 32]);  view_450 = None
        permute_226: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_451, [0, 2, 1, 3]);  view_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_227: "f32[512, 128]" = torch.ops.aten.permute.default(primals_528, [1, 0])
        addmm_170: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_529, view_442, permute_227);  primals_529 = permute_227 = None
        view_453: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_170, [256, 128, 128]);  addmm_170 = None
        view_454: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_453, [256, 128, -1, 32]);  view_453 = None
        permute_228: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_135: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_224, 0.4204482076268573);  permute_224 = None
        permute_229: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_226, [0, 1, 3, 2]);  permute_226 = None
        mul_136: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_229, 0.4204482076268573);  permute_229 = None
        expand_45: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_135, [256, 4, 128, 32]);  mul_135 = None
        clone_56: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_455: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_56, [1024, 128, 32]);  clone_56 = None
        expand_46: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_136, [256, 4, 32, 128]);  mul_136 = None
        clone_57: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_456: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_57, [1024, 32, 128]);  clone_57 = None
        bmm_22: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_455, view_456)
        view_457: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_22, [256, 4, 128, 128]);  bmm_22 = None
        add_172: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_457, where);  view_457 = None
        amax_11: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_172, [-1], True)
        sub_11: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_172, amax_11);  amax_11 = None
        exp_11: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_12: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        eq_11: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_172, -inf);  add_172 = None
        logical_not_22: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        where_23: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_23, full_default_3, div_11);  logical_not_23 = div_11 = None
        inductor_lookup_seed_default_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_12: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        gt_11: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 0.1);  inductor_random_default_12 = None
        mul_137: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_11, where_23)
        mul_138: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_137, 1.1111111111111112);  mul_137 = None
        expand_47: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_138, [256, 4, 128, 128]);  mul_138 = None
        view_458: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_47, [1024, 128, 128]);  expand_47 = None
        expand_48: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_228, [256, 4, 128, 32]);  permute_228 = None
        clone_58: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_459: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_58, [1024, 128, 32]);  clone_58 = None
        bmm_23: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_458, view_459)
        view_460: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_23, [256, 4, 128, 32]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_230: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None
        clone_59: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_461: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_59, [256, 128, -1]);  clone_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_462: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_461, [32768, 128]);  view_461 = None
        permute_231: "f32[128, 128]" = torch.ops.aten.permute.default(primals_530, [1, 0])
        addmm_171: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_531, view_462, permute_231);  primals_531 = permute_231 = None
        view_463: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_171, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_173: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_463, add_170);  view_463 = add_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_139: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_173, primals_532);  add_173 = None
        add_174: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_139, primals_533);  mul_139 = primals_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_464: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_174, [32768, 128])
        permute_232: "f32[128, 512]" = torch.ops.aten.permute.default(primals_534, [1, 0])
        addmm_172: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_535, view_464, permute_232);  primals_535 = permute_232 = None
        view_465: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_172, [256, 128, 512]);  addmm_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_44: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_465);  view_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_466: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_44, [32768, 512])
        permute_233: "f32[512, 128]" = torch.ops.aten.permute.default(primals_536, [1, 0])
        addmm_173: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_537, view_466, permute_233);  primals_537 = permute_233 = None
        view_467: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_173, [256, 128, 128]);  addmm_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_175: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_467, add_174);  view_467 = add_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_140: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_175, primals_538)
        add_176: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_140, primals_539);  mul_140 = primals_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_468: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_176, [32768, 128])
        permute_234: "f32[128, 512]" = torch.ops.aten.permute.default(primals_540, [1, 0])
        addmm_174: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_541, view_468, permute_234);  primals_541 = permute_234 = None
        view_469: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_174, [256, 128, 512]);  addmm_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_45: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_469);  view_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_470: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_45, [32768, 512])
        permute_235: "f32[512, 128]" = torch.ops.aten.permute.default(primals_542, [1, 0])
        addmm_175: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_543, view_470, permute_235);  primals_543 = permute_235 = None
        view_471: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_175, [256, 128, 128]);  addmm_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_177: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_471, add_176);  view_471 = add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_141: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_177, primals_544)
        add_178: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_141, primals_545);  mul_141 = primals_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_472: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_178, [32768, 128])
        permute_236: "f32[128, 512]" = torch.ops.aten.permute.default(primals_546, [1, 0])
        addmm_176: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_547, view_472, permute_236);  primals_547 = permute_236 = None
        view_473: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_176, [256, 128, 512]);  addmm_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_46: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_473);  view_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_474: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_46, [32768, 512])
        permute_237: "f32[512, 128]" = torch.ops.aten.permute.default(primals_548, [1, 0])
        addmm_177: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_549, view_474, permute_237);  primals_549 = permute_237 = None
        view_475: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_177, [256, 128, 128]);  addmm_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_179: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_475, add_178);  view_475 = add_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_142: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_179, primals_550)
        add_180: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_142, primals_551);  mul_142 = primals_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_476: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_180, [32768, 128])
        permute_238: "f32[128, 512]" = torch.ops.aten.permute.default(primals_552, [1, 0])
        addmm_178: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_553, view_476, permute_238);  primals_553 = permute_238 = None
        view_477: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_178, [256, 128, 512]);  addmm_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_47: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_477);  view_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_478: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_47, [32768, 512])
        permute_239: "f32[512, 128]" = torch.ops.aten.permute.default(primals_554, [1, 0])
        addmm_179: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_555, view_478, permute_239);  primals_555 = permute_239 = None
        view_479: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_179, [256, 128, 128]);  addmm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_181: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_479, add_180);  view_479 = add_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_143: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_181, primals_556)
        add_182: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_143, primals_557);  mul_143 = primals_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_480: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_182, [32768, 128]);  add_182 = None
        permute_240: "f32[128, 512]" = torch.ops.aten.permute.default(primals_558, [1, 0])
        addmm_180: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_559, view_480, permute_240);  primals_559 = permute_240 = None
        view_481: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_180, [256, 128, 512]);  addmm_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_183: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_481, add_169);  view_481 = add_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_144: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_183, primals_560)
        add_184: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_144, primals_561);  mul_144 = primals_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_482: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_184, [32768, 512])
        permute_241: "f32[512, 128]" = torch.ops.aten.permute.default(primals_562, [1, 0])
        addmm_181: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_563, view_482, permute_241);  primals_563 = permute_241 = None
        view_483: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_181, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_145: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_483, primals_564);  view_483 = None
        add_185: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_145, primals_565);  mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_242: "f32[512, 128]" = torch.ops.aten.permute.default(primals_566, [1, 0])
        addmm_182: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_567, view_482, permute_242);  primals_567 = permute_242 = None
        view_485: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_182, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_146: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_485, primals_568);  view_485 = None
        add_186: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_146, primals_569);  mul_146 = primals_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_486: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_186, [32768, 128]);  add_186 = None
        permute_243: "f32[128, 128]" = torch.ops.aten.permute.default(primals_570, [1, 0])
        addmm_183: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_571, view_486, permute_243);  primals_571 = permute_243 = None
        view_487: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_183, [256, 128, 128]);  addmm_183 = None
        view_488: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_487, [256, 128, -1, 32]);  view_487 = None
        permute_244: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_245: "f32[128, 128]" = torch.ops.aten.permute.default(primals_572, [1, 0])
        addmm_184: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_573, view_486, permute_245);  primals_573 = permute_245 = None
        view_490: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_184, [256, 128, 128]);  addmm_184 = None
        view_491: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_490, [256, 128, -1, 32]);  view_490 = None
        permute_246: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_247: "f32[512, 128]" = torch.ops.aten.permute.default(primals_574, [1, 0])
        addmm_185: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_575, view_482, permute_247);  primals_575 = permute_247 = None
        view_493: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_185, [256, 128, 128]);  addmm_185 = None
        view_494: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_493, [256, 128, -1, 32]);  view_493 = None
        permute_248: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_147: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_244, 0.4204482076268573);  permute_244 = None
        permute_249: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_246, [0, 1, 3, 2]);  permute_246 = None
        mul_148: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_249, 0.4204482076268573);  permute_249 = None
        expand_49: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_147, [256, 4, 128, 32]);  mul_147 = None
        clone_61: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_495: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_61, [1024, 128, 32]);  clone_61 = None
        expand_50: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_148, [256, 4, 32, 128]);  mul_148 = None
        clone_62: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_496: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_62, [1024, 32, 128]);  clone_62 = None
        bmm_24: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_495, view_496)
        view_497: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_24, [256, 4, 128, 128]);  bmm_24 = None
        add_187: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_497, where);  view_497 = None
        amax_12: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_187, [-1], True)
        sub_12: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_187, amax_12);  amax_12 = None
        exp_12: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_13: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        eq_12: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_187, -inf);  add_187 = None
        logical_not_24: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_12);  eq_12 = None
        any_13: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_24, -1, True);  logical_not_24 = None
        logical_not_25: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_13);  any_13 = None
        where_25: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_25, full_default_3, div_12);  logical_not_25 = div_12 = None
        inductor_lookup_seed_default_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_11: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        gt_12: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_11, 0.1);  inductor_random_default_11 = None
        mul_149: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_12, where_25)
        mul_150: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_149, 1.1111111111111112);  mul_149 = None
        expand_51: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_150, [256, 4, 128, 128]);  mul_150 = None
        view_498: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_51, [1024, 128, 128]);  expand_51 = None
        expand_52: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_248, [256, 4, 128, 32]);  permute_248 = None
        clone_63: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_499: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_63, [1024, 128, 32]);  clone_63 = None
        bmm_25: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_498, view_499)
        view_500: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_25, [256, 4, 128, 32]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_250: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None
        clone_64: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_501: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_64, [256, 128, -1]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_502: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_501, [32768, 128]);  view_501 = None
        permute_251: "f32[128, 128]" = torch.ops.aten.permute.default(primals_576, [1, 0])
        addmm_186: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_577, view_502, permute_251);  primals_577 = permute_251 = None
        view_503: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_186, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_188: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_503, add_185);  view_503 = add_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_151: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_188, primals_578);  add_188 = None
        add_189: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_151, primals_579);  mul_151 = primals_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_504: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_189, [32768, 128])
        permute_252: "f32[128, 512]" = torch.ops.aten.permute.default(primals_580, [1, 0])
        addmm_187: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_581, view_504, permute_252);  primals_581 = permute_252 = None
        view_505: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_187, [256, 128, 512]);  addmm_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_48: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_505);  view_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_506: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_48, [32768, 512])
        permute_253: "f32[512, 128]" = torch.ops.aten.permute.default(primals_582, [1, 0])
        addmm_188: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_583, view_506, permute_253);  primals_583 = permute_253 = None
        view_507: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_188, [256, 128, 128]);  addmm_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_190: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_507, add_189);  view_507 = add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_152: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_190, primals_584)
        add_191: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_152, primals_585);  mul_152 = primals_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_508: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_191, [32768, 128])
        permute_254: "f32[128, 512]" = torch.ops.aten.permute.default(primals_586, [1, 0])
        addmm_189: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_587, view_508, permute_254);  primals_587 = permute_254 = None
        view_509: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_189, [256, 128, 512]);  addmm_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_49: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_509);  view_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_510: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_49, [32768, 512])
        permute_255: "f32[512, 128]" = torch.ops.aten.permute.default(primals_588, [1, 0])
        addmm_190: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_589, view_510, permute_255);  primals_589 = permute_255 = None
        view_511: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_190, [256, 128, 128]);  addmm_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_192: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_511, add_191);  view_511 = add_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_153: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_192, primals_590)
        add_193: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_153, primals_591);  mul_153 = primals_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_512: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_193, [32768, 128])
        permute_256: "f32[128, 512]" = torch.ops.aten.permute.default(primals_592, [1, 0])
        addmm_191: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_593, view_512, permute_256);  primals_593 = permute_256 = None
        view_513: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_191, [256, 128, 512]);  addmm_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_50: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_513);  view_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_514: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_50, [32768, 512])
        permute_257: "f32[512, 128]" = torch.ops.aten.permute.default(primals_594, [1, 0])
        addmm_192: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_595, view_514, permute_257);  primals_595 = permute_257 = None
        view_515: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_192, [256, 128, 128]);  addmm_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_194: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_515, add_193);  view_515 = add_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_154: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_194, primals_596)
        add_195: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_154, primals_597);  mul_154 = primals_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_516: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_195, [32768, 128])
        permute_258: "f32[128, 512]" = torch.ops.aten.permute.default(primals_598, [1, 0])
        addmm_193: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_599, view_516, permute_258);  primals_599 = permute_258 = None
        view_517: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_193, [256, 128, 512]);  addmm_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_51: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_517);  view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_518: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_51, [32768, 512])
        permute_259: "f32[512, 128]" = torch.ops.aten.permute.default(primals_600, [1, 0])
        addmm_194: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_601, view_518, permute_259);  primals_601 = permute_259 = None
        view_519: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_194, [256, 128, 128]);  addmm_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_196: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_519, add_195);  view_519 = add_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_155: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_196, primals_602)
        add_197: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_155, primals_603);  mul_155 = primals_603 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_520: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_197, [32768, 128]);  add_197 = None
        permute_260: "f32[128, 512]" = torch.ops.aten.permute.default(primals_604, [1, 0])
        addmm_195: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_605, view_520, permute_260);  primals_605 = permute_260 = None
        view_521: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_195, [256, 128, 512]);  addmm_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_198: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_521, add_184);  view_521 = add_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_156: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_198, primals_606)
        add_199: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_156, primals_607);  mul_156 = primals_607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_522: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_199, [32768, 512])
        permute_261: "f32[512, 128]" = torch.ops.aten.permute.default(primals_608, [1, 0])
        addmm_196: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_609, view_522, permute_261);  primals_609 = permute_261 = None
        view_523: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_196, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_157: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_523, primals_610);  view_523 = None
        add_200: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_157, primals_611);  mul_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_262: "f32[512, 128]" = torch.ops.aten.permute.default(primals_612, [1, 0])
        addmm_197: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_613, view_522, permute_262);  primals_613 = permute_262 = None
        view_525: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_197, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_158: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_525, primals_614);  view_525 = None
        add_201: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_158, primals_615);  mul_158 = primals_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_526: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_201, [32768, 128]);  add_201 = None
        permute_263: "f32[128, 128]" = torch.ops.aten.permute.default(primals_616, [1, 0])
        addmm_198: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_617, view_526, permute_263);  primals_617 = permute_263 = None
        view_527: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_198, [256, 128, 128]);  addmm_198 = None
        view_528: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_527, [256, 128, -1, 32]);  view_527 = None
        permute_264: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_528, [0, 2, 1, 3]);  view_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_265: "f32[128, 128]" = torch.ops.aten.permute.default(primals_618, [1, 0])
        addmm_199: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_619, view_526, permute_265);  primals_619 = permute_265 = None
        view_530: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_199, [256, 128, 128]);  addmm_199 = None
        view_531: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_530, [256, 128, -1, 32]);  view_530 = None
        permute_266: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_531, [0, 2, 1, 3]);  view_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_267: "f32[512, 128]" = torch.ops.aten.permute.default(primals_620, [1, 0])
        addmm_200: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_621, view_522, permute_267);  primals_621 = permute_267 = None
        view_533: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_200, [256, 128, 128]);  addmm_200 = None
        view_534: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_533, [256, 128, -1, 32]);  view_533 = None
        permute_268: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_159: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_264, 0.4204482076268573);  permute_264 = None
        permute_269: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_266, [0, 1, 3, 2]);  permute_266 = None
        mul_160: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_269, 0.4204482076268573);  permute_269 = None
        expand_53: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_159, [256, 4, 128, 32]);  mul_159 = None
        clone_66: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_535: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_66, [1024, 128, 32]);  clone_66 = None
        expand_54: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_160, [256, 4, 32, 128]);  mul_160 = None
        clone_67: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_536: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_67, [1024, 32, 128]);  clone_67 = None
        bmm_26: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_535, view_536)
        view_537: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_26, [256, 4, 128, 128]);  bmm_26 = None
        add_202: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_537, where);  view_537 = None
        amax_13: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_202, [-1], True)
        sub_13: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_202, amax_13);  amax_13 = None
        exp_13: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_14: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        eq_13: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_202, -inf);  add_202 = None
        logical_not_26: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_13);  eq_13 = None
        any_14: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_26, -1, True);  logical_not_26 = None
        logical_not_27: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_14);  any_14 = None
        where_27: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_27, full_default_3, div_13);  logical_not_27 = div_13 = None
        inductor_lookup_seed_default_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_10: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        gt_13: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_10, 0.1);  inductor_random_default_10 = None
        mul_161: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_13, where_27)
        mul_162: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_161, 1.1111111111111112);  mul_161 = None
        expand_55: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_162, [256, 4, 128, 128]);  mul_162 = None
        view_538: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_55, [1024, 128, 128]);  expand_55 = None
        expand_56: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_268, [256, 4, 128, 32]);  permute_268 = None
        clone_68: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_539: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_68, [1024, 128, 32]);  clone_68 = None
        bmm_27: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_538, view_539)
        view_540: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_27, [256, 4, 128, 32]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_270: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None
        clone_69: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_541: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_69, [256, 128, -1]);  clone_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_542: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_541, [32768, 128]);  view_541 = None
        permute_271: "f32[128, 128]" = torch.ops.aten.permute.default(primals_622, [1, 0])
        addmm_201: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_623, view_542, permute_271);  primals_623 = permute_271 = None
        view_543: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_201, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_203: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_543, add_200);  view_543 = add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_163: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_203, primals_624);  add_203 = None
        add_204: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_163, primals_625);  mul_163 = primals_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_544: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_204, [32768, 128])
        permute_272: "f32[128, 512]" = torch.ops.aten.permute.default(primals_626, [1, 0])
        addmm_202: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_627, view_544, permute_272);  primals_627 = permute_272 = None
        view_545: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_202, [256, 128, 512]);  addmm_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_52: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_545);  view_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_546: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_52, [32768, 512])
        permute_273: "f32[512, 128]" = torch.ops.aten.permute.default(primals_628, [1, 0])
        addmm_203: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_629, view_546, permute_273);  primals_629 = permute_273 = None
        view_547: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_203, [256, 128, 128]);  addmm_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_205: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_547, add_204);  view_547 = add_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_164: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_205, primals_630)
        add_206: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_164, primals_631);  mul_164 = primals_631 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_548: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_206, [32768, 128])
        permute_274: "f32[128, 512]" = torch.ops.aten.permute.default(primals_632, [1, 0])
        addmm_204: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_633, view_548, permute_274);  primals_633 = permute_274 = None
        view_549: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_204, [256, 128, 512]);  addmm_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_53: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_549);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_550: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_53, [32768, 512])
        permute_275: "f32[512, 128]" = torch.ops.aten.permute.default(primals_634, [1, 0])
        addmm_205: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_635, view_550, permute_275);  primals_635 = permute_275 = None
        view_551: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_205, [256, 128, 128]);  addmm_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_207: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_551, add_206);  view_551 = add_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_165: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_207, primals_636)
        add_208: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_165, primals_637);  mul_165 = primals_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_552: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_208, [32768, 128])
        permute_276: "f32[128, 512]" = torch.ops.aten.permute.default(primals_638, [1, 0])
        addmm_206: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_639, view_552, permute_276);  primals_639 = permute_276 = None
        view_553: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_206, [256, 128, 512]);  addmm_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_54: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_553);  view_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_554: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_54, [32768, 512])
        permute_277: "f32[512, 128]" = torch.ops.aten.permute.default(primals_640, [1, 0])
        addmm_207: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_641, view_554, permute_277);  primals_641 = permute_277 = None
        view_555: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_207, [256, 128, 128]);  addmm_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_209: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_555, add_208);  view_555 = add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_166: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_209, primals_642)
        add_210: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_166, primals_643);  mul_166 = primals_643 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_556: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_210, [32768, 128])
        permute_278: "f32[128, 512]" = torch.ops.aten.permute.default(primals_644, [1, 0])
        addmm_208: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_645, view_556, permute_278);  primals_645 = permute_278 = None
        view_557: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_208, [256, 128, 512]);  addmm_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_55: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_557);  view_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_558: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_55, [32768, 512])
        permute_279: "f32[512, 128]" = torch.ops.aten.permute.default(primals_646, [1, 0])
        addmm_209: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_647, view_558, permute_279);  primals_647 = permute_279 = None
        view_559: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_209, [256, 128, 128]);  addmm_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_211: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_559, add_210);  view_559 = add_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_167: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_211, primals_648)
        add_212: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_167, primals_649);  mul_167 = primals_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_560: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_212, [32768, 128]);  add_212 = None
        permute_280: "f32[128, 512]" = torch.ops.aten.permute.default(primals_650, [1, 0])
        addmm_210: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_651, view_560, permute_280);  primals_651 = permute_280 = None
        view_561: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_210, [256, 128, 512]);  addmm_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_213: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_561, add_199);  view_561 = add_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_168: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_213, primals_652)
        add_214: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_168, primals_653);  mul_168 = primals_653 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_562: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_214, [32768, 512])
        permute_281: "f32[512, 128]" = torch.ops.aten.permute.default(primals_654, [1, 0])
        addmm_211: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_655, view_562, permute_281);  primals_655 = permute_281 = None
        view_563: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_211, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_169: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_563, primals_656);  view_563 = None
        add_215: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_169, primals_657);  mul_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_282: "f32[512, 128]" = torch.ops.aten.permute.default(primals_658, [1, 0])
        addmm_212: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_659, view_562, permute_282);  primals_659 = permute_282 = None
        view_565: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_212, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_170: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_565, primals_660);  view_565 = None
        add_216: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_170, primals_661);  mul_170 = primals_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_566: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_216, [32768, 128]);  add_216 = None
        permute_283: "f32[128, 128]" = torch.ops.aten.permute.default(primals_662, [1, 0])
        addmm_213: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_663, view_566, permute_283);  primals_663 = permute_283 = None
        view_567: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_213, [256, 128, 128]);  addmm_213 = None
        view_568: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_567, [256, 128, -1, 32]);  view_567 = None
        permute_284: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_285: "f32[128, 128]" = torch.ops.aten.permute.default(primals_664, [1, 0])
        addmm_214: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_665, view_566, permute_285);  primals_665 = permute_285 = None
        view_570: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_214, [256, 128, 128]);  addmm_214 = None
        view_571: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_570, [256, 128, -1, 32]);  view_570 = None
        permute_286: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_571, [0, 2, 1, 3]);  view_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_287: "f32[512, 128]" = torch.ops.aten.permute.default(primals_666, [1, 0])
        addmm_215: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_667, view_562, permute_287);  primals_667 = permute_287 = None
        view_573: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_215, [256, 128, 128]);  addmm_215 = None
        view_574: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_573, [256, 128, -1, 32]);  view_573 = None
        permute_288: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_574, [0, 2, 1, 3]);  view_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_171: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_284, 0.4204482076268573);  permute_284 = None
        permute_289: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_286, [0, 1, 3, 2]);  permute_286 = None
        mul_172: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_289, 0.4204482076268573);  permute_289 = None
        expand_57: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_171, [256, 4, 128, 32]);  mul_171 = None
        clone_71: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_575: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_71, [1024, 128, 32]);  clone_71 = None
        expand_58: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_172, [256, 4, 32, 128]);  mul_172 = None
        clone_72: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_576: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_72, [1024, 32, 128]);  clone_72 = None
        bmm_28: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_575, view_576)
        view_577: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_28, [256, 4, 128, 128]);  bmm_28 = None
        add_217: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_577, where);  view_577 = None
        amax_14: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_217, [-1], True)
        sub_14: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_217, amax_14);  amax_14 = None
        exp_14: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_15: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        eq_14: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_217, -inf);  add_217 = None
        logical_not_28: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_14);  eq_14 = None
        any_15: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_28, -1, True);  logical_not_28 = None
        logical_not_29: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_15);  any_15 = None
        where_29: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_29, full_default_3, div_14);  logical_not_29 = div_14 = None
        inductor_lookup_seed_default_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_9: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        gt_14: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_9, 0.1);  inductor_random_default_9 = None
        mul_173: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_14, where_29)
        mul_174: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_173, 1.1111111111111112);  mul_173 = None
        expand_59: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_174, [256, 4, 128, 128]);  mul_174 = None
        view_578: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_59, [1024, 128, 128]);  expand_59 = None
        expand_60: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_288, [256, 4, 128, 32]);  permute_288 = None
        clone_73: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_579: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_73, [1024, 128, 32]);  clone_73 = None
        bmm_29: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_578, view_579)
        view_580: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_29, [256, 4, 128, 32]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_290: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_580, [0, 2, 1, 3]);  view_580 = None
        clone_74: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_290, memory_format = torch.contiguous_format);  permute_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_581: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_74, [256, 128, -1]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_582: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_581, [32768, 128]);  view_581 = None
        permute_291: "f32[128, 128]" = torch.ops.aten.permute.default(primals_668, [1, 0])
        addmm_216: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_669, view_582, permute_291);  primals_669 = permute_291 = None
        view_583: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_216, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_218: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_583, add_215);  view_583 = add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_175: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_218, primals_670);  add_218 = None
        add_219: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_175, primals_671);  mul_175 = primals_671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_584: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_219, [32768, 128])
        permute_292: "f32[128, 512]" = torch.ops.aten.permute.default(primals_672, [1, 0])
        addmm_217: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_673, view_584, permute_292);  primals_673 = permute_292 = None
        view_585: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_217, [256, 128, 512]);  addmm_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_56: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_585);  view_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_586: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_56, [32768, 512])
        permute_293: "f32[512, 128]" = torch.ops.aten.permute.default(primals_674, [1, 0])
        addmm_218: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_675, view_586, permute_293);  primals_675 = permute_293 = None
        view_587: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_218, [256, 128, 128]);  addmm_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_220: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_587, add_219);  view_587 = add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_176: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_220, primals_676)
        add_221: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_176, primals_677);  mul_176 = primals_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_588: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_221, [32768, 128])
        permute_294: "f32[128, 512]" = torch.ops.aten.permute.default(primals_678, [1, 0])
        addmm_219: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_679, view_588, permute_294);  primals_679 = permute_294 = None
        view_589: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_219, [256, 128, 512]);  addmm_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_57: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_589);  view_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_590: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_57, [32768, 512])
        permute_295: "f32[512, 128]" = torch.ops.aten.permute.default(primals_680, [1, 0])
        addmm_220: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_681, view_590, permute_295);  primals_681 = permute_295 = None
        view_591: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_220, [256, 128, 128]);  addmm_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_222: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_591, add_221);  view_591 = add_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_177: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_222, primals_682)
        add_223: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_177, primals_683);  mul_177 = primals_683 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_592: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_223, [32768, 128])
        permute_296: "f32[128, 512]" = torch.ops.aten.permute.default(primals_684, [1, 0])
        addmm_221: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_685, view_592, permute_296);  primals_685 = permute_296 = None
        view_593: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_221, [256, 128, 512]);  addmm_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_58: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_593);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_594: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_58, [32768, 512])
        permute_297: "f32[512, 128]" = torch.ops.aten.permute.default(primals_686, [1, 0])
        addmm_222: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_687, view_594, permute_297);  primals_687 = permute_297 = None
        view_595: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_222, [256, 128, 128]);  addmm_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_224: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_595, add_223);  view_595 = add_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_178: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_224, primals_688)
        add_225: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_178, primals_689);  mul_178 = primals_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_596: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_225, [32768, 128])
        permute_298: "f32[128, 512]" = torch.ops.aten.permute.default(primals_690, [1, 0])
        addmm_223: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_691, view_596, permute_298);  primals_691 = permute_298 = None
        view_597: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_223, [256, 128, 512]);  addmm_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_59: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_597);  view_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_598: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_59, [32768, 512])
        permute_299: "f32[512, 128]" = torch.ops.aten.permute.default(primals_692, [1, 0])
        addmm_224: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_693, view_598, permute_299);  primals_693 = permute_299 = None
        view_599: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_224, [256, 128, 128]);  addmm_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_226: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_599, add_225);  view_599 = add_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_179: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_226, primals_694)
        add_227: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_179, primals_695);  mul_179 = primals_695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_600: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_227, [32768, 128]);  add_227 = None
        permute_300: "f32[128, 512]" = torch.ops.aten.permute.default(primals_696, [1, 0])
        addmm_225: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_697, view_600, permute_300);  primals_697 = permute_300 = None
        view_601: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_225, [256, 128, 512]);  addmm_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_228: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_601, add_214);  view_601 = add_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_180: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_228, primals_698)
        add_229: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_180, primals_699);  mul_180 = primals_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_602: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_229, [32768, 512])
        permute_301: "f32[512, 128]" = torch.ops.aten.permute.default(primals_700, [1, 0])
        addmm_226: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_701, view_602, permute_301);  primals_701 = permute_301 = None
        view_603: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_226, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_181: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_603, primals_702);  view_603 = None
        add_230: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_181, primals_703);  mul_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_302: "f32[512, 128]" = torch.ops.aten.permute.default(primals_704, [1, 0])
        addmm_227: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_705, view_602, permute_302);  primals_705 = permute_302 = None
        view_605: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_227, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_182: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_605, primals_706);  view_605 = None
        add_231: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_182, primals_707);  mul_182 = primals_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_606: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_231, [32768, 128]);  add_231 = None
        permute_303: "f32[128, 128]" = torch.ops.aten.permute.default(primals_708, [1, 0])
        addmm_228: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_709, view_606, permute_303);  primals_709 = permute_303 = None
        view_607: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_228, [256, 128, 128]);  addmm_228 = None
        view_608: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_607, [256, 128, -1, 32]);  view_607 = None
        permute_304: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_608, [0, 2, 1, 3]);  view_608 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_305: "f32[128, 128]" = torch.ops.aten.permute.default(primals_710, [1, 0])
        addmm_229: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_711, view_606, permute_305);  primals_711 = permute_305 = None
        view_610: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_229, [256, 128, 128]);  addmm_229 = None
        view_611: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_610, [256, 128, -1, 32]);  view_610 = None
        permute_306: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_611, [0, 2, 1, 3]);  view_611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_307: "f32[512, 128]" = torch.ops.aten.permute.default(primals_712, [1, 0])
        addmm_230: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_713, view_602, permute_307);  primals_713 = permute_307 = None
        view_613: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_230, [256, 128, 128]);  addmm_230 = None
        view_614: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_613, [256, 128, -1, 32]);  view_613 = None
        permute_308: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_183: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_304, 0.4204482076268573);  permute_304 = None
        permute_309: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_306, [0, 1, 3, 2]);  permute_306 = None
        mul_184: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_309, 0.4204482076268573);  permute_309 = None
        expand_61: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_183, [256, 4, 128, 32]);  mul_183 = None
        clone_76: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_615: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_76, [1024, 128, 32]);  clone_76 = None
        expand_62: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_184, [256, 4, 32, 128]);  mul_184 = None
        clone_77: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_616: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_77, [1024, 32, 128]);  clone_77 = None
        bmm_30: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_615, view_616)
        view_617: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_30, [256, 4, 128, 128]);  bmm_30 = None
        add_232: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_617, where);  view_617 = None
        amax_15: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_232, [-1], True)
        sub_15: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_232, amax_15);  amax_15 = None
        exp_15: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_16: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        eq_15: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_232, -inf);  add_232 = None
        logical_not_30: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_15);  eq_15 = None
        any_16: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_30, -1, True);  logical_not_30 = None
        logical_not_31: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_16);  any_16 = None
        where_31: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_31, full_default_3, div_15);  logical_not_31 = div_15 = None
        inductor_lookup_seed_default_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_8: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        gt_15: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_185: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_15, where_31)
        mul_186: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_185, 1.1111111111111112);  mul_185 = None
        expand_63: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_186, [256, 4, 128, 128]);  mul_186 = None
        view_618: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_63, [1024, 128, 128]);  expand_63 = None
        expand_64: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_308, [256, 4, 128, 32]);  permute_308 = None
        clone_78: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_619: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_78, [1024, 128, 32]);  clone_78 = None
        bmm_31: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_618, view_619)
        view_620: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_31, [256, 4, 128, 32]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_310: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_620, [0, 2, 1, 3]);  view_620 = None
        clone_79: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_621: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_79, [256, 128, -1]);  clone_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_622: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_621, [32768, 128]);  view_621 = None
        permute_311: "f32[128, 128]" = torch.ops.aten.permute.default(primals_714, [1, 0])
        addmm_231: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_715, view_622, permute_311);  primals_715 = permute_311 = None
        view_623: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_231, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_233: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_623, add_230);  view_623 = add_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_187: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_233, primals_716);  add_233 = None
        add_234: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_187, primals_717);  mul_187 = primals_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_624: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_234, [32768, 128])
        permute_312: "f32[128, 512]" = torch.ops.aten.permute.default(primals_718, [1, 0])
        addmm_232: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_719, view_624, permute_312);  primals_719 = permute_312 = None
        view_625: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_232, [256, 128, 512]);  addmm_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_60: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_625);  view_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_626: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_60, [32768, 512])
        permute_313: "f32[512, 128]" = torch.ops.aten.permute.default(primals_720, [1, 0])
        addmm_233: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_721, view_626, permute_313);  primals_721 = permute_313 = None
        view_627: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_233, [256, 128, 128]);  addmm_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_235: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_627, add_234);  view_627 = add_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_188: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_235, primals_722)
        add_236: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_188, primals_723);  mul_188 = primals_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_628: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_236, [32768, 128])
        permute_314: "f32[128, 512]" = torch.ops.aten.permute.default(primals_724, [1, 0])
        addmm_234: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_725, view_628, permute_314);  primals_725 = permute_314 = None
        view_629: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_234, [256, 128, 512]);  addmm_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_61: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_629);  view_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_630: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_61, [32768, 512])
        permute_315: "f32[512, 128]" = torch.ops.aten.permute.default(primals_726, [1, 0])
        addmm_235: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_727, view_630, permute_315);  primals_727 = permute_315 = None
        view_631: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_235, [256, 128, 128]);  addmm_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_237: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_631, add_236);  view_631 = add_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_189: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_237, primals_728)
        add_238: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_189, primals_729);  mul_189 = primals_729 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_632: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_238, [32768, 128])
        permute_316: "f32[128, 512]" = torch.ops.aten.permute.default(primals_730, [1, 0])
        addmm_236: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_731, view_632, permute_316);  primals_731 = permute_316 = None
        view_633: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_236, [256, 128, 512]);  addmm_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_62: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_633);  view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_634: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_62, [32768, 512])
        permute_317: "f32[512, 128]" = torch.ops.aten.permute.default(primals_732, [1, 0])
        addmm_237: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_733, view_634, permute_317);  primals_733 = permute_317 = None
        view_635: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_237, [256, 128, 128]);  addmm_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_239: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_635, add_238);  view_635 = add_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_190: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_239, primals_734)
        add_240: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_190, primals_735);  mul_190 = primals_735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_636: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_240, [32768, 128])
        permute_318: "f32[128, 512]" = torch.ops.aten.permute.default(primals_736, [1, 0])
        addmm_238: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_737, view_636, permute_318);  primals_737 = permute_318 = None
        view_637: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_238, [256, 128, 512]);  addmm_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_63: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_637);  view_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_638: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_63, [32768, 512])
        permute_319: "f32[512, 128]" = torch.ops.aten.permute.default(primals_738, [1, 0])
        addmm_239: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_739, view_638, permute_319);  primals_739 = permute_319 = None
        view_639: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_239, [256, 128, 128]);  addmm_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_241: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_639, add_240);  view_639 = add_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_191: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_241, primals_740)
        add_242: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_191, primals_741);  mul_191 = primals_741 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_640: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_242, [32768, 128]);  add_242 = None
        permute_320: "f32[128, 512]" = torch.ops.aten.permute.default(primals_742, [1, 0])
        addmm_240: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_743, view_640, permute_320);  primals_743 = permute_320 = None
        view_641: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_240, [256, 128, 512]);  addmm_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_243: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_641, add_229);  view_641 = add_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_192: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_243, primals_744)
        add_244: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_192, primals_745);  mul_192 = primals_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_642: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_244, [32768, 512])
        permute_321: "f32[512, 128]" = torch.ops.aten.permute.default(primals_746, [1, 0])
        addmm_241: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_747, view_642, permute_321);  primals_747 = permute_321 = None
        view_643: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_241, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_193: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_643, primals_748);  view_643 = None
        add_245: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_193, primals_749);  mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_322: "f32[512, 128]" = torch.ops.aten.permute.default(primals_750, [1, 0])
        addmm_242: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_751, view_642, permute_322);  primals_751 = permute_322 = None
        view_645: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_242, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_194: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_645, primals_752);  view_645 = None
        add_246: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_194, primals_753);  mul_194 = primals_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_646: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_246, [32768, 128]);  add_246 = None
        permute_323: "f32[128, 128]" = torch.ops.aten.permute.default(primals_754, [1, 0])
        addmm_243: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_755, view_646, permute_323);  primals_755 = permute_323 = None
        view_647: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_243, [256, 128, 128]);  addmm_243 = None
        view_648: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_647, [256, 128, -1, 32]);  view_647 = None
        permute_324: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_325: "f32[128, 128]" = torch.ops.aten.permute.default(primals_756, [1, 0])
        addmm_244: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_757, view_646, permute_325);  primals_757 = permute_325 = None
        view_650: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_244, [256, 128, 128]);  addmm_244 = None
        view_651: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_650, [256, 128, -1, 32]);  view_650 = None
        permute_326: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_651, [0, 2, 1, 3]);  view_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_327: "f32[512, 128]" = torch.ops.aten.permute.default(primals_758, [1, 0])
        addmm_245: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_759, view_642, permute_327);  primals_759 = permute_327 = None
        view_653: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_245, [256, 128, 128]);  addmm_245 = None
        view_654: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_653, [256, 128, -1, 32]);  view_653 = None
        permute_328: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_195: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_324, 0.4204482076268573);  permute_324 = None
        permute_329: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_326, [0, 1, 3, 2]);  permute_326 = None
        mul_196: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_329, 0.4204482076268573);  permute_329 = None
        expand_65: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_195, [256, 4, 128, 32]);  mul_195 = None
        clone_81: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_655: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_81, [1024, 128, 32]);  clone_81 = None
        expand_66: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_196, [256, 4, 32, 128]);  mul_196 = None
        clone_82: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_656: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_82, [1024, 32, 128]);  clone_82 = None
        bmm_32: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_655, view_656)
        view_657: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_32, [256, 4, 128, 128]);  bmm_32 = None
        add_247: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_657, where);  view_657 = None
        amax_16: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_247, [-1], True)
        sub_16: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_247, amax_16);  amax_16 = None
        exp_16: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_17: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        eq_16: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_247, -inf);  add_247 = None
        logical_not_32: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_16);  eq_16 = None
        any_17: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_32, -1, True);  logical_not_32 = None
        logical_not_33: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_17);  any_17 = None
        where_33: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_33, full_default_3, div_16);  logical_not_33 = div_16 = None
        inductor_lookup_seed_default_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_7: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        gt_16: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_7, 0.1);  inductor_random_default_7 = None
        mul_197: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_16, where_33)
        mul_198: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_197, 1.1111111111111112);  mul_197 = None
        expand_67: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_198, [256, 4, 128, 128]);  mul_198 = None
        view_658: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_67, [1024, 128, 128]);  expand_67 = None
        expand_68: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_328, [256, 4, 128, 32]);  permute_328 = None
        clone_83: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_659: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_83, [1024, 128, 32]);  clone_83 = None
        bmm_33: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_658, view_659)
        view_660: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_33, [256, 4, 128, 32]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_330: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_660, [0, 2, 1, 3]);  view_660 = None
        clone_84: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_661: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_84, [256, 128, -1]);  clone_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_662: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_661, [32768, 128]);  view_661 = None
        permute_331: "f32[128, 128]" = torch.ops.aten.permute.default(primals_760, [1, 0])
        addmm_246: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_761, view_662, permute_331);  primals_761 = permute_331 = None
        view_663: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_246, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_248: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_663, add_245);  view_663 = add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_199: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_248, primals_762);  add_248 = None
        add_249: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_199, primals_763);  mul_199 = primals_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_664: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_249, [32768, 128])
        permute_332: "f32[128, 512]" = torch.ops.aten.permute.default(primals_764, [1, 0])
        addmm_247: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_765, view_664, permute_332);  primals_765 = permute_332 = None
        view_665: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_247, [256, 128, 512]);  addmm_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_64: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_665);  view_665 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_666: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_64, [32768, 512])
        permute_333: "f32[512, 128]" = torch.ops.aten.permute.default(primals_766, [1, 0])
        addmm_248: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_767, view_666, permute_333);  primals_767 = permute_333 = None
        view_667: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_248, [256, 128, 128]);  addmm_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_250: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_667, add_249);  view_667 = add_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_200: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_250, primals_768)
        add_251: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_200, primals_769);  mul_200 = primals_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_668: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_251, [32768, 128])
        permute_334: "f32[128, 512]" = torch.ops.aten.permute.default(primals_770, [1, 0])
        addmm_249: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_771, view_668, permute_334);  primals_771 = permute_334 = None
        view_669: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_249, [256, 128, 512]);  addmm_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_65: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_669);  view_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_670: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_65, [32768, 512])
        permute_335: "f32[512, 128]" = torch.ops.aten.permute.default(primals_772, [1, 0])
        addmm_250: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_773, view_670, permute_335);  primals_773 = permute_335 = None
        view_671: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_250, [256, 128, 128]);  addmm_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_252: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_671, add_251);  view_671 = add_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_201: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_252, primals_774)
        add_253: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_201, primals_775);  mul_201 = primals_775 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_672: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_253, [32768, 128])
        permute_336: "f32[128, 512]" = torch.ops.aten.permute.default(primals_776, [1, 0])
        addmm_251: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_777, view_672, permute_336);  primals_777 = permute_336 = None
        view_673: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_251, [256, 128, 512]);  addmm_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_66: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_673);  view_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_674: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_66, [32768, 512])
        permute_337: "f32[512, 128]" = torch.ops.aten.permute.default(primals_778, [1, 0])
        addmm_252: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_779, view_674, permute_337);  primals_779 = permute_337 = None
        view_675: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_252, [256, 128, 128]);  addmm_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_254: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_675, add_253);  view_675 = add_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_202: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_254, primals_780)
        add_255: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_202, primals_781);  mul_202 = primals_781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_676: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_255, [32768, 128])
        permute_338: "f32[128, 512]" = torch.ops.aten.permute.default(primals_782, [1, 0])
        addmm_253: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_783, view_676, permute_338);  primals_783 = permute_338 = None
        view_677: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_253, [256, 128, 512]);  addmm_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_67: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_677);  view_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_678: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_67, [32768, 512])
        permute_339: "f32[512, 128]" = torch.ops.aten.permute.default(primals_784, [1, 0])
        addmm_254: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_785, view_678, permute_339);  primals_785 = permute_339 = None
        view_679: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_254, [256, 128, 128]);  addmm_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_256: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_679, add_255);  view_679 = add_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_203: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_256, primals_786)
        add_257: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_203, primals_787);  mul_203 = primals_787 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_680: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_257, [32768, 128]);  add_257 = None
        permute_340: "f32[128, 512]" = torch.ops.aten.permute.default(primals_788, [1, 0])
        addmm_255: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_789, view_680, permute_340);  primals_789 = permute_340 = None
        view_681: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_255, [256, 128, 512]);  addmm_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_258: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_681, add_244);  view_681 = add_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_204: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_258, primals_790)
        add_259: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_204, primals_791);  mul_204 = primals_791 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_682: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_259, [32768, 512])
        permute_341: "f32[512, 128]" = torch.ops.aten.permute.default(primals_792, [1, 0])
        addmm_256: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_793, view_682, permute_341);  primals_793 = permute_341 = None
        view_683: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_256, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_205: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_683, primals_794);  view_683 = None
        add_260: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_205, primals_795);  mul_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_342: "f32[512, 128]" = torch.ops.aten.permute.default(primals_796, [1, 0])
        addmm_257: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_797, view_682, permute_342);  primals_797 = permute_342 = None
        view_685: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_257, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_206: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_685, primals_798);  view_685 = None
        add_261: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_206, primals_799);  mul_206 = primals_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_686: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_261, [32768, 128]);  add_261 = None
        permute_343: "f32[128, 128]" = torch.ops.aten.permute.default(primals_800, [1, 0])
        addmm_258: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_801, view_686, permute_343);  primals_801 = permute_343 = None
        view_687: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_258, [256, 128, 128]);  addmm_258 = None
        view_688: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_687, [256, 128, -1, 32]);  view_687 = None
        permute_344: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_688, [0, 2, 1, 3]);  view_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_345: "f32[128, 128]" = torch.ops.aten.permute.default(primals_802, [1, 0])
        addmm_259: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_803, view_686, permute_345);  primals_803 = permute_345 = None
        view_690: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_259, [256, 128, 128]);  addmm_259 = None
        view_691: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_690, [256, 128, -1, 32]);  view_690 = None
        permute_346: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_691, [0, 2, 1, 3]);  view_691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_347: "f32[512, 128]" = torch.ops.aten.permute.default(primals_804, [1, 0])
        addmm_260: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_805, view_682, permute_347);  primals_805 = permute_347 = None
        view_693: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_260, [256, 128, 128]);  addmm_260 = None
        view_694: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_693, [256, 128, -1, 32]);  view_693 = None
        permute_348: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_694, [0, 2, 1, 3]);  view_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_207: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_344, 0.4204482076268573);  permute_344 = None
        permute_349: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_346, [0, 1, 3, 2]);  permute_346 = None
        mul_208: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_349, 0.4204482076268573);  permute_349 = None
        expand_69: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_207, [256, 4, 128, 32]);  mul_207 = None
        clone_86: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_695: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_86, [1024, 128, 32]);  clone_86 = None
        expand_70: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_208, [256, 4, 32, 128]);  mul_208 = None
        clone_87: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_696: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_87, [1024, 32, 128]);  clone_87 = None
        bmm_34: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_695, view_696)
        view_697: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_34, [256, 4, 128, 128]);  bmm_34 = None
        add_262: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_697, where);  view_697 = None
        amax_17: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_262, [-1], True)
        sub_17: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_262, amax_17);  amax_17 = None
        exp_17: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_18: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        eq_17: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_262, -inf);  add_262 = None
        logical_not_34: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_17);  eq_17 = None
        any_18: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_34, -1, True);  logical_not_34 = None
        logical_not_35: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_18);  any_18 = None
        where_35: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_35, full_default_3, div_17);  logical_not_35 = div_17 = None
        inductor_lookup_seed_default_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_6: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        gt_17: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_6, 0.1);  inductor_random_default_6 = None
        mul_209: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_17, where_35)
        mul_210: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_209, 1.1111111111111112);  mul_209 = None
        expand_71: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_210, [256, 4, 128, 128]);  mul_210 = None
        view_698: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_71, [1024, 128, 128]);  expand_71 = None
        expand_72: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_348, [256, 4, 128, 32]);  permute_348 = None
        clone_88: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_699: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_88, [1024, 128, 32]);  clone_88 = None
        bmm_35: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_698, view_699)
        view_700: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_35, [256, 4, 128, 32]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_350: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_700, [0, 2, 1, 3]);  view_700 = None
        clone_89: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_350, memory_format = torch.contiguous_format);  permute_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_701: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_89, [256, 128, -1]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_702: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_701, [32768, 128]);  view_701 = None
        permute_351: "f32[128, 128]" = torch.ops.aten.permute.default(primals_806, [1, 0])
        addmm_261: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_807, view_702, permute_351);  primals_807 = permute_351 = None
        view_703: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_261, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_263: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_703, add_260);  view_703 = add_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_211: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_263, primals_808);  add_263 = None
        add_264: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_211, primals_809);  mul_211 = primals_809 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_704: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_264, [32768, 128])
        permute_352: "f32[128, 512]" = torch.ops.aten.permute.default(primals_810, [1, 0])
        addmm_262: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_811, view_704, permute_352);  primals_811 = permute_352 = None
        view_705: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_262, [256, 128, 512]);  addmm_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_68: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_705);  view_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_706: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_68, [32768, 512])
        permute_353: "f32[512, 128]" = torch.ops.aten.permute.default(primals_812, [1, 0])
        addmm_263: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_813, view_706, permute_353);  primals_813 = permute_353 = None
        view_707: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_263, [256, 128, 128]);  addmm_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_265: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_707, add_264);  view_707 = add_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_212: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_265, primals_814)
        add_266: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_212, primals_815);  mul_212 = primals_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_708: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_266, [32768, 128])
        permute_354: "f32[128, 512]" = torch.ops.aten.permute.default(primals_816, [1, 0])
        addmm_264: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_817, view_708, permute_354);  primals_817 = permute_354 = None
        view_709: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_264, [256, 128, 512]);  addmm_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_69: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_709);  view_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_710: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_69, [32768, 512])
        permute_355: "f32[512, 128]" = torch.ops.aten.permute.default(primals_818, [1, 0])
        addmm_265: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_819, view_710, permute_355);  primals_819 = permute_355 = None
        view_711: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_265, [256, 128, 128]);  addmm_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_267: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_711, add_266);  view_711 = add_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_213: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_267, primals_820)
        add_268: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_213, primals_821);  mul_213 = primals_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_712: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_268, [32768, 128])
        permute_356: "f32[128, 512]" = torch.ops.aten.permute.default(primals_822, [1, 0])
        addmm_266: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_823, view_712, permute_356);  primals_823 = permute_356 = None
        view_713: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_266, [256, 128, 512]);  addmm_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_70: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_713);  view_713 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_714: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_70, [32768, 512])
        permute_357: "f32[512, 128]" = torch.ops.aten.permute.default(primals_824, [1, 0])
        addmm_267: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_825, view_714, permute_357);  primals_825 = permute_357 = None
        view_715: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_267, [256, 128, 128]);  addmm_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_269: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_715, add_268);  view_715 = add_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_214: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_269, primals_826)
        add_270: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_214, primals_827);  mul_214 = primals_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_716: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_270, [32768, 128])
        permute_358: "f32[128, 512]" = torch.ops.aten.permute.default(primals_828, [1, 0])
        addmm_268: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_829, view_716, permute_358);  primals_829 = permute_358 = None
        view_717: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_268, [256, 128, 512]);  addmm_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_71: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_717);  view_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_718: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_71, [32768, 512])
        permute_359: "f32[512, 128]" = torch.ops.aten.permute.default(primals_830, [1, 0])
        addmm_269: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_831, view_718, permute_359);  primals_831 = permute_359 = None
        view_719: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_269, [256, 128, 128]);  addmm_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_271: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_719, add_270);  view_719 = add_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_215: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_271, primals_832)
        add_272: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_215, primals_833);  mul_215 = primals_833 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_720: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_272, [32768, 128]);  add_272 = None
        permute_360: "f32[128, 512]" = torch.ops.aten.permute.default(primals_834, [1, 0])
        addmm_270: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_835, view_720, permute_360);  primals_835 = permute_360 = None
        view_721: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_270, [256, 128, 512]);  addmm_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_273: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_721, add_259);  view_721 = add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_216: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_273, primals_836)
        add_274: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_216, primals_837);  mul_216 = primals_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_722: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_274, [32768, 512])
        permute_361: "f32[512, 128]" = torch.ops.aten.permute.default(primals_838, [1, 0])
        addmm_271: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_839, view_722, permute_361);  primals_839 = permute_361 = None
        view_723: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_271, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_217: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_723, primals_840);  view_723 = None
        add_275: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_217, primals_841);  mul_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_362: "f32[512, 128]" = torch.ops.aten.permute.default(primals_842, [1, 0])
        addmm_272: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_843, view_722, permute_362);  primals_843 = permute_362 = None
        view_725: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_272, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_218: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_725, primals_844);  view_725 = None
        add_276: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_218, primals_845);  mul_218 = primals_845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_726: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_276, [32768, 128]);  add_276 = None
        permute_363: "f32[128, 128]" = torch.ops.aten.permute.default(primals_846, [1, 0])
        addmm_273: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_847, view_726, permute_363);  primals_847 = permute_363 = None
        view_727: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_273, [256, 128, 128]);  addmm_273 = None
        view_728: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_727, [256, 128, -1, 32]);  view_727 = None
        permute_364: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_728, [0, 2, 1, 3]);  view_728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_365: "f32[128, 128]" = torch.ops.aten.permute.default(primals_848, [1, 0])
        addmm_274: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_849, view_726, permute_365);  primals_849 = permute_365 = None
        view_730: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_274, [256, 128, 128]);  addmm_274 = None
        view_731: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_730, [256, 128, -1, 32]);  view_730 = None
        permute_366: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_731, [0, 2, 1, 3]);  view_731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_367: "f32[512, 128]" = torch.ops.aten.permute.default(primals_850, [1, 0])
        addmm_275: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_851, view_722, permute_367);  primals_851 = permute_367 = None
        view_733: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_275, [256, 128, 128]);  addmm_275 = None
        view_734: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_733, [256, 128, -1, 32]);  view_733 = None
        permute_368: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_734, [0, 2, 1, 3]);  view_734 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_219: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_364, 0.4204482076268573);  permute_364 = None
        permute_369: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_366, [0, 1, 3, 2]);  permute_366 = None
        mul_220: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_369, 0.4204482076268573);  permute_369 = None
        expand_73: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_219, [256, 4, 128, 32]);  mul_219 = None
        clone_91: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_735: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_91, [1024, 128, 32]);  clone_91 = None
        expand_74: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_220, [256, 4, 32, 128]);  mul_220 = None
        clone_92: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_736: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_92, [1024, 32, 128]);  clone_92 = None
        bmm_36: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_735, view_736)
        view_737: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_36, [256, 4, 128, 128]);  bmm_36 = None
        add_277: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_737, where);  view_737 = None
        amax_18: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_277, [-1], True)
        sub_18: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_277, amax_18);  amax_18 = None
        exp_18: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_19: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        eq_18: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_277, -inf);  add_277 = None
        logical_not_36: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_18);  eq_18 = None
        any_19: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_36, -1, True);  logical_not_36 = None
        logical_not_37: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_19);  any_19 = None
        where_37: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_37, full_default_3, div_18);  logical_not_37 = div_18 = None
        inductor_lookup_seed_default_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_5: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        gt_18: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_5, 0.1);  inductor_random_default_5 = None
        mul_221: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_18, where_37)
        mul_222: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_221, 1.1111111111111112);  mul_221 = None
        expand_75: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_222, [256, 4, 128, 128]);  mul_222 = None
        view_738: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_75, [1024, 128, 128]);  expand_75 = None
        expand_76: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_368, [256, 4, 128, 32]);  permute_368 = None
        clone_93: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_739: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_93, [1024, 128, 32]);  clone_93 = None
        bmm_37: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_738, view_739)
        view_740: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_37, [256, 4, 128, 32]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_370: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_740, [0, 2, 1, 3]);  view_740 = None
        clone_94: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_741: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_94, [256, 128, -1]);  clone_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_742: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_741, [32768, 128]);  view_741 = None
        permute_371: "f32[128, 128]" = torch.ops.aten.permute.default(primals_852, [1, 0])
        addmm_276: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_853, view_742, permute_371);  primals_853 = permute_371 = None
        view_743: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_276, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_278: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_743, add_275);  view_743 = add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_223: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_278, primals_854);  add_278 = None
        add_279: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_223, primals_855);  mul_223 = primals_855 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_744: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_279, [32768, 128])
        permute_372: "f32[128, 512]" = torch.ops.aten.permute.default(primals_856, [1, 0])
        addmm_277: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_857, view_744, permute_372);  primals_857 = permute_372 = None
        view_745: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_277, [256, 128, 512]);  addmm_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_72: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_745);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_746: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_72, [32768, 512])
        permute_373: "f32[512, 128]" = torch.ops.aten.permute.default(primals_858, [1, 0])
        addmm_278: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_859, view_746, permute_373);  primals_859 = permute_373 = None
        view_747: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_278, [256, 128, 128]);  addmm_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_280: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_747, add_279);  view_747 = add_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_224: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_280, primals_860)
        add_281: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_224, primals_861);  mul_224 = primals_861 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_748: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_281, [32768, 128])
        permute_374: "f32[128, 512]" = torch.ops.aten.permute.default(primals_862, [1, 0])
        addmm_279: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_863, view_748, permute_374);  primals_863 = permute_374 = None
        view_749: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_279, [256, 128, 512]);  addmm_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_73: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_749);  view_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_750: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_73, [32768, 512])
        permute_375: "f32[512, 128]" = torch.ops.aten.permute.default(primals_864, [1, 0])
        addmm_280: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_865, view_750, permute_375);  primals_865 = permute_375 = None
        view_751: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_280, [256, 128, 128]);  addmm_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_282: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_751, add_281);  view_751 = add_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_225: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_282, primals_866)
        add_283: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_225, primals_867);  mul_225 = primals_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_752: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_283, [32768, 128])
        permute_376: "f32[128, 512]" = torch.ops.aten.permute.default(primals_868, [1, 0])
        addmm_281: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_869, view_752, permute_376);  primals_869 = permute_376 = None
        view_753: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_281, [256, 128, 512]);  addmm_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_74: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_753);  view_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_754: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_74, [32768, 512])
        permute_377: "f32[512, 128]" = torch.ops.aten.permute.default(primals_870, [1, 0])
        addmm_282: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_871, view_754, permute_377);  primals_871 = permute_377 = None
        view_755: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_282, [256, 128, 128]);  addmm_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_284: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_755, add_283);  view_755 = add_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_226: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_284, primals_872)
        add_285: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_226, primals_873);  mul_226 = primals_873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_756: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_285, [32768, 128])
        permute_378: "f32[128, 512]" = torch.ops.aten.permute.default(primals_874, [1, 0])
        addmm_283: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_875, view_756, permute_378);  primals_875 = permute_378 = None
        view_757: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_283, [256, 128, 512]);  addmm_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_75: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_757);  view_757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_758: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_75, [32768, 512])
        permute_379: "f32[512, 128]" = torch.ops.aten.permute.default(primals_876, [1, 0])
        addmm_284: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_877, view_758, permute_379);  primals_877 = permute_379 = None
        view_759: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_284, [256, 128, 128]);  addmm_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_286: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_759, add_285);  view_759 = add_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_227: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_286, primals_878)
        add_287: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_227, primals_879);  mul_227 = primals_879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_760: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_287, [32768, 128]);  add_287 = None
        permute_380: "f32[128, 512]" = torch.ops.aten.permute.default(primals_880, [1, 0])
        addmm_285: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_881, view_760, permute_380);  primals_881 = permute_380 = None
        view_761: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_285, [256, 128, 512]);  addmm_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_288: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_761, add_274);  view_761 = add_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_228: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_288, primals_882)
        add_289: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_228, primals_883);  mul_228 = primals_883 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_762: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_289, [32768, 512])
        permute_381: "f32[512, 128]" = torch.ops.aten.permute.default(primals_884, [1, 0])
        addmm_286: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_885, view_762, permute_381);  primals_885 = permute_381 = None
        view_763: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_286, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_229: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_763, primals_886);  view_763 = None
        add_290: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_229, primals_887);  mul_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_382: "f32[512, 128]" = torch.ops.aten.permute.default(primals_888, [1, 0])
        addmm_287: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_889, view_762, permute_382);  primals_889 = permute_382 = None
        view_765: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_287, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_230: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_765, primals_890);  view_765 = None
        add_291: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_230, primals_891);  mul_230 = primals_891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_766: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_291, [32768, 128]);  add_291 = None
        permute_383: "f32[128, 128]" = torch.ops.aten.permute.default(primals_892, [1, 0])
        addmm_288: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_893, view_766, permute_383);  primals_893 = permute_383 = None
        view_767: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_288, [256, 128, 128]);  addmm_288 = None
        view_768: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_767, [256, 128, -1, 32]);  view_767 = None
        permute_384: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_768, [0, 2, 1, 3]);  view_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_385: "f32[128, 128]" = torch.ops.aten.permute.default(primals_894, [1, 0])
        addmm_289: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_895, view_766, permute_385);  primals_895 = permute_385 = None
        view_770: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_289, [256, 128, 128]);  addmm_289 = None
        view_771: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_770, [256, 128, -1, 32]);  view_770 = None
        permute_386: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_771, [0, 2, 1, 3]);  view_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_387: "f32[512, 128]" = torch.ops.aten.permute.default(primals_896, [1, 0])
        addmm_290: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_897, view_762, permute_387);  primals_897 = permute_387 = None
        view_773: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_290, [256, 128, 128]);  addmm_290 = None
        view_774: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_773, [256, 128, -1, 32]);  view_773 = None
        permute_388: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_231: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_384, 0.4204482076268573);  permute_384 = None
        permute_389: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_386, [0, 1, 3, 2]);  permute_386 = None
        mul_232: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_389, 0.4204482076268573);  permute_389 = None
        expand_77: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_231, [256, 4, 128, 32]);  mul_231 = None
        clone_96: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_775: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_96, [1024, 128, 32]);  clone_96 = None
        expand_78: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_232, [256, 4, 32, 128]);  mul_232 = None
        clone_97: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
        view_776: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_97, [1024, 32, 128]);  clone_97 = None
        bmm_38: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_775, view_776)
        view_777: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_38, [256, 4, 128, 128]);  bmm_38 = None
        add_292: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_777, where);  view_777 = None
        amax_19: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_292, [-1], True)
        sub_19: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_292, amax_19);  amax_19 = None
        exp_19: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_20: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        eq_19: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_292, -inf);  add_292 = None
        logical_not_38: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_19);  eq_19 = None
        any_20: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_38, -1, True);  logical_not_38 = None
        logical_not_39: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_20);  any_20 = None
        where_39: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_39, full_default_3, div_19);  logical_not_39 = div_19 = None
        inductor_lookup_seed_default_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_4: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        gt_19: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_233: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_19, where_39)
        mul_234: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_233, 1.1111111111111112);  mul_233 = None
        expand_79: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_234, [256, 4, 128, 128]);  mul_234 = None
        view_778: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_79, [1024, 128, 128]);  expand_79 = None
        expand_80: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_388, [256, 4, 128, 32]);  permute_388 = None
        clone_98: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_779: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_98, [1024, 128, 32]);  clone_98 = None
        bmm_39: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_778, view_779)
        view_780: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_39, [256, 4, 128, 32]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_390: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_780, [0, 2, 1, 3]);  view_780 = None
        clone_99: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_390, memory_format = torch.contiguous_format);  permute_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_781: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_99, [256, 128, -1]);  clone_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_782: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_781, [32768, 128]);  view_781 = None
        permute_391: "f32[128, 128]" = torch.ops.aten.permute.default(primals_898, [1, 0])
        addmm_291: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_899, view_782, permute_391);  primals_899 = permute_391 = None
        view_783: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_291, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_293: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_783, add_290);  view_783 = add_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_235: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_293, primals_900);  add_293 = None
        add_294: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_235, primals_901);  mul_235 = primals_901 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_784: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_294, [32768, 128])
        permute_392: "f32[128, 512]" = torch.ops.aten.permute.default(primals_902, [1, 0])
        addmm_292: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_903, view_784, permute_392);  primals_903 = permute_392 = None
        view_785: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_292, [256, 128, 512]);  addmm_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_76: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_785);  view_785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_786: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_76, [32768, 512])
        permute_393: "f32[512, 128]" = torch.ops.aten.permute.default(primals_904, [1, 0])
        addmm_293: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_905, view_786, permute_393);  primals_905 = permute_393 = None
        view_787: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_293, [256, 128, 128]);  addmm_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_295: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_787, add_294);  view_787 = add_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_236: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_295, primals_906)
        add_296: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_236, primals_907);  mul_236 = primals_907 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_788: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_296, [32768, 128])
        permute_394: "f32[128, 512]" = torch.ops.aten.permute.default(primals_908, [1, 0])
        addmm_294: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_909, view_788, permute_394);  primals_909 = permute_394 = None
        view_789: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_294, [256, 128, 512]);  addmm_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_77: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_789);  view_789 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_790: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_77, [32768, 512])
        permute_395: "f32[512, 128]" = torch.ops.aten.permute.default(primals_910, [1, 0])
        addmm_295: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_911, view_790, permute_395);  primals_911 = permute_395 = None
        view_791: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_295, [256, 128, 128]);  addmm_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_297: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_791, add_296);  view_791 = add_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_237: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_297, primals_912)
        add_298: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_237, primals_913);  mul_237 = primals_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_792: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_298, [32768, 128])
        permute_396: "f32[128, 512]" = torch.ops.aten.permute.default(primals_914, [1, 0])
        addmm_296: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_915, view_792, permute_396);  primals_915 = permute_396 = None
        view_793: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_296, [256, 128, 512]);  addmm_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_78: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_793);  view_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_794: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_78, [32768, 512])
        permute_397: "f32[512, 128]" = torch.ops.aten.permute.default(primals_916, [1, 0])
        addmm_297: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_917, view_794, permute_397);  primals_917 = permute_397 = None
        view_795: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_297, [256, 128, 128]);  addmm_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_299: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_795, add_298);  view_795 = add_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_238: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_299, primals_918)
        add_300: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_238, primals_919);  mul_238 = primals_919 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_796: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_300, [32768, 128])
        permute_398: "f32[128, 512]" = torch.ops.aten.permute.default(primals_920, [1, 0])
        addmm_298: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_921, view_796, permute_398);  primals_921 = permute_398 = None
        view_797: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_298, [256, 128, 512]);  addmm_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_79: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_797);  view_797 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_798: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_79, [32768, 512])
        permute_399: "f32[512, 128]" = torch.ops.aten.permute.default(primals_922, [1, 0])
        addmm_299: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_923, view_798, permute_399);  primals_923 = permute_399 = None
        view_799: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_299, [256, 128, 128]);  addmm_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_301: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_799, add_300);  view_799 = add_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_239: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_301, primals_924)
        add_302: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_239, primals_925);  mul_239 = primals_925 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_800: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_302, [32768, 128]);  add_302 = None
        permute_400: "f32[128, 512]" = torch.ops.aten.permute.default(primals_926, [1, 0])
        addmm_300: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_927, view_800, permute_400);  primals_927 = permute_400 = None
        view_801: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_300, [256, 128, 512]);  addmm_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_303: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_801, add_289);  view_801 = add_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_240: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_303, primals_928)
        add_304: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_240, primals_929);  mul_240 = primals_929 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_802: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_304, [32768, 512])
        permute_401: "f32[512, 128]" = torch.ops.aten.permute.default(primals_930, [1, 0])
        addmm_301: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_931, view_802, permute_401);  primals_931 = permute_401 = None
        view_803: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_301, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_241: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_803, primals_932);  view_803 = None
        add_305: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_241, primals_933);  mul_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_402: "f32[512, 128]" = torch.ops.aten.permute.default(primals_934, [1, 0])
        addmm_302: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_935, view_802, permute_402);  primals_935 = permute_402 = None
        view_805: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_302, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_242: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_805, primals_936);  view_805 = None
        add_306: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_242, primals_937);  mul_242 = primals_937 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_806: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_306, [32768, 128]);  add_306 = None
        permute_403: "f32[128, 128]" = torch.ops.aten.permute.default(primals_938, [1, 0])
        addmm_303: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_939, view_806, permute_403);  primals_939 = permute_403 = None
        view_807: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_303, [256, 128, 128]);  addmm_303 = None
        view_808: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_807, [256, 128, -1, 32]);  view_807 = None
        permute_404: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_808, [0, 2, 1, 3]);  view_808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_405: "f32[128, 128]" = torch.ops.aten.permute.default(primals_940, [1, 0])
        addmm_304: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_941, view_806, permute_405);  primals_941 = permute_405 = None
        view_810: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_304, [256, 128, 128]);  addmm_304 = None
        view_811: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_810, [256, 128, -1, 32]);  view_810 = None
        permute_406: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_811, [0, 2, 1, 3]);  view_811 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_407: "f32[512, 128]" = torch.ops.aten.permute.default(primals_942, [1, 0])
        addmm_305: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_943, view_802, permute_407);  primals_943 = permute_407 = None
        view_813: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_305, [256, 128, 128]);  addmm_305 = None
        view_814: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_813, [256, 128, -1, 32]);  view_813 = None
        permute_408: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_814, [0, 2, 1, 3]);  view_814 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_243: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_404, 0.4204482076268573);  permute_404 = None
        permute_409: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_406, [0, 1, 3, 2]);  permute_406 = None
        mul_244: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_409, 0.4204482076268573);  permute_409 = None
        expand_81: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_243, [256, 4, 128, 32]);  mul_243 = None
        clone_101: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_815: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_101, [1024, 128, 32]);  clone_101 = None
        expand_82: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_244, [256, 4, 32, 128]);  mul_244 = None
        clone_102: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_816: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_102, [1024, 32, 128]);  clone_102 = None
        bmm_40: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_815, view_816)
        view_817: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_40, [256, 4, 128, 128]);  bmm_40 = None
        add_307: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_817, where);  view_817 = None
        amax_20: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_307, [-1], True)
        sub_20: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_307, amax_20);  amax_20 = None
        exp_20: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_21: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        eq_20: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_307, -inf);  add_307 = None
        logical_not_40: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_20);  eq_20 = None
        any_21: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_40, -1, True);  logical_not_40 = None
        logical_not_41: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_21);  any_21 = None
        where_41: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_41, full_default_3, div_20);  logical_not_41 = div_20 = None
        inductor_lookup_seed_default_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_3: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        gt_20: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_3, 0.1);  inductor_random_default_3 = None
        mul_245: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_20, where_41)
        mul_246: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_245, 1.1111111111111112);  mul_245 = None
        expand_83: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_246, [256, 4, 128, 128]);  mul_246 = None
        view_818: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_83, [1024, 128, 128]);  expand_83 = None
        expand_84: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_408, [256, 4, 128, 32]);  permute_408 = None
        clone_103: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_819: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_103, [1024, 128, 32]);  clone_103 = None
        bmm_41: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_818, view_819)
        view_820: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_41, [256, 4, 128, 32]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_410: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_820, [0, 2, 1, 3]);  view_820 = None
        clone_104: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_821: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_104, [256, 128, -1]);  clone_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_822: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_821, [32768, 128]);  view_821 = None
        permute_411: "f32[128, 128]" = torch.ops.aten.permute.default(primals_944, [1, 0])
        addmm_306: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_945, view_822, permute_411);  primals_945 = permute_411 = None
        view_823: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_306, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_308: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_823, add_305);  view_823 = add_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_247: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_308, primals_946);  add_308 = None
        add_309: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_247, primals_947);  mul_247 = primals_947 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_824: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_309, [32768, 128])
        permute_412: "f32[128, 512]" = torch.ops.aten.permute.default(primals_948, [1, 0])
        addmm_307: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_949, view_824, permute_412);  primals_949 = permute_412 = None
        view_825: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_307, [256, 128, 512]);  addmm_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_80: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_825);  view_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_826: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_80, [32768, 512])
        permute_413: "f32[512, 128]" = torch.ops.aten.permute.default(primals_950, [1, 0])
        addmm_308: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_951, view_826, permute_413);  primals_951 = permute_413 = None
        view_827: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_308, [256, 128, 128]);  addmm_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_310: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_827, add_309);  view_827 = add_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_248: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_310, primals_952)
        add_311: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_248, primals_953);  mul_248 = primals_953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_828: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_311, [32768, 128])
        permute_414: "f32[128, 512]" = torch.ops.aten.permute.default(primals_954, [1, 0])
        addmm_309: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_955, view_828, permute_414);  primals_955 = permute_414 = None
        view_829: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_309, [256, 128, 512]);  addmm_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_81: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_829);  view_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_830: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_81, [32768, 512])
        permute_415: "f32[512, 128]" = torch.ops.aten.permute.default(primals_956, [1, 0])
        addmm_310: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_957, view_830, permute_415);  primals_957 = permute_415 = None
        view_831: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_310, [256, 128, 128]);  addmm_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_312: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_831, add_311);  view_831 = add_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_249: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_312, primals_958)
        add_313: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_249, primals_959);  mul_249 = primals_959 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_832: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_313, [32768, 128])
        permute_416: "f32[128, 512]" = torch.ops.aten.permute.default(primals_960, [1, 0])
        addmm_311: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_961, view_832, permute_416);  primals_961 = permute_416 = None
        view_833: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_311, [256, 128, 512]);  addmm_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_82: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_833);  view_833 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_834: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_82, [32768, 512])
        permute_417: "f32[512, 128]" = torch.ops.aten.permute.default(primals_962, [1, 0])
        addmm_312: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_963, view_834, permute_417);  primals_963 = permute_417 = None
        view_835: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_312, [256, 128, 128]);  addmm_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_314: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_835, add_313);  view_835 = add_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_250: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_314, primals_964)
        add_315: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_250, primals_965);  mul_250 = primals_965 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_836: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_315, [32768, 128])
        permute_418: "f32[128, 512]" = torch.ops.aten.permute.default(primals_966, [1, 0])
        addmm_313: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_967, view_836, permute_418);  primals_967 = permute_418 = None
        view_837: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_313, [256, 128, 512]);  addmm_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_83: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_837);  view_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_838: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_83, [32768, 512])
        permute_419: "f32[512, 128]" = torch.ops.aten.permute.default(primals_968, [1, 0])
        addmm_314: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_969, view_838, permute_419);  primals_969 = permute_419 = None
        view_839: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_314, [256, 128, 128]);  addmm_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_316: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_839, add_315);  view_839 = add_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_251: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_316, primals_970)
        add_317: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_251, primals_971);  mul_251 = primals_971 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_840: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_317, [32768, 128]);  add_317 = None
        permute_420: "f32[128, 512]" = torch.ops.aten.permute.default(primals_972, [1, 0])
        addmm_315: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_973, view_840, permute_420);  primals_973 = permute_420 = None
        view_841: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_315, [256, 128, 512]);  addmm_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_318: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_841, add_304);  view_841 = add_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_252: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_318, primals_974)
        add_319: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_252, primals_975);  mul_252 = primals_975 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_842: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_319, [32768, 512])
        permute_421: "f32[512, 128]" = torch.ops.aten.permute.default(primals_976, [1, 0])
        addmm_316: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_977, view_842, permute_421);  primals_977 = permute_421 = None
        view_843: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_316, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_253: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_843, primals_978);  view_843 = None
        add_320: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_253, primals_979);  mul_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_422: "f32[512, 128]" = torch.ops.aten.permute.default(primals_980, [1, 0])
        addmm_317: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_981, view_842, permute_422);  primals_981 = permute_422 = None
        view_845: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_317, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_254: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_845, primals_982);  view_845 = None
        add_321: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_254, primals_983);  mul_254 = primals_983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_846: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_321, [32768, 128]);  add_321 = None
        permute_423: "f32[128, 128]" = torch.ops.aten.permute.default(primals_984, [1, 0])
        addmm_318: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_985, view_846, permute_423);  primals_985 = permute_423 = None
        view_847: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_318, [256, 128, 128]);  addmm_318 = None
        view_848: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_847, [256, 128, -1, 32]);  view_847 = None
        permute_424: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_848, [0, 2, 1, 3]);  view_848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_425: "f32[128, 128]" = torch.ops.aten.permute.default(primals_986, [1, 0])
        addmm_319: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_987, view_846, permute_425);  primals_987 = permute_425 = None
        view_850: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_319, [256, 128, 128]);  addmm_319 = None
        view_851: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_850, [256, 128, -1, 32]);  view_850 = None
        permute_426: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_851, [0, 2, 1, 3]);  view_851 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_427: "f32[512, 128]" = torch.ops.aten.permute.default(primals_988, [1, 0])
        addmm_320: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_989, view_842, permute_427);  primals_989 = permute_427 = None
        view_853: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_320, [256, 128, 128]);  addmm_320 = None
        view_854: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_853, [256, 128, -1, 32]);  view_853 = None
        permute_428: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_854, [0, 2, 1, 3]);  view_854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_255: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_424, 0.4204482076268573);  permute_424 = None
        permute_429: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_426, [0, 1, 3, 2]);  permute_426 = None
        mul_256: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_429, 0.4204482076268573);  permute_429 = None
        expand_85: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_255, [256, 4, 128, 32]);  mul_255 = None
        clone_106: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_855: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_106, [1024, 128, 32]);  clone_106 = None
        expand_86: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_256, [256, 4, 32, 128]);  mul_256 = None
        clone_107: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
        view_856: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_107, [1024, 32, 128]);  clone_107 = None
        bmm_42: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_855, view_856)
        view_857: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_42, [256, 4, 128, 128]);  bmm_42 = None
        add_322: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_857, where);  view_857 = None
        amax_21: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_322, [-1], True)
        sub_21: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_322, amax_21);  amax_21 = None
        exp_21: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_22: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        eq_21: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_322, -inf);  add_322 = None
        logical_not_42: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_21);  eq_21 = None
        any_22: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_42, -1, True);  logical_not_42 = None
        logical_not_43: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_22);  any_22 = None
        where_43: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_43, full_default_3, div_21);  logical_not_43 = div_21 = None
        inductor_lookup_seed_default_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_2: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        gt_21: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_2, 0.1);  inductor_random_default_2 = None
        mul_257: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_21, where_43)
        mul_258: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_257, 1.1111111111111112);  mul_257 = None
        expand_87: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_258, [256, 4, 128, 128]);  mul_258 = None
        view_858: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_87, [1024, 128, 128]);  expand_87 = None
        expand_88: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_428, [256, 4, 128, 32]);  permute_428 = None
        clone_108: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_859: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_108, [1024, 128, 32]);  clone_108 = None
        bmm_43: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_858, view_859)
        view_860: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_43, [256, 4, 128, 32]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_430: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_860, [0, 2, 1, 3]);  view_860 = None
        clone_109: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_430, memory_format = torch.contiguous_format);  permute_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_861: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_109, [256, 128, -1]);  clone_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_862: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_861, [32768, 128]);  view_861 = None
        permute_431: "f32[128, 128]" = torch.ops.aten.permute.default(primals_990, [1, 0])
        addmm_321: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_991, view_862, permute_431);  primals_991 = permute_431 = None
        view_863: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_321, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_323: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_863, add_320);  view_863 = add_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_259: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_323, primals_992);  add_323 = None
        add_324: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_259, primals_993);  mul_259 = primals_993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_864: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_324, [32768, 128])
        permute_432: "f32[128, 512]" = torch.ops.aten.permute.default(primals_994, [1, 0])
        addmm_322: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_995, view_864, permute_432);  primals_995 = permute_432 = None
        view_865: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_322, [256, 128, 512]);  addmm_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_84: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_865);  view_865 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_866: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_84, [32768, 512])
        permute_433: "f32[512, 128]" = torch.ops.aten.permute.default(primals_996, [1, 0])
        addmm_323: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_997, view_866, permute_433);  primals_997 = permute_433 = None
        view_867: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_323, [256, 128, 128]);  addmm_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_325: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_867, add_324);  view_867 = add_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_260: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_325, primals_998)
        add_326: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_260, primals_999);  mul_260 = primals_999 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_868: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_326, [32768, 128])
        permute_434: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1000, [1, 0])
        addmm_324: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_1001, view_868, permute_434);  primals_1001 = permute_434 = None
        view_869: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_324, [256, 128, 512]);  addmm_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_85: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_869);  view_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_870: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_85, [32768, 512])
        permute_435: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1002, [1, 0])
        addmm_325: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1003, view_870, permute_435);  primals_1003 = permute_435 = None
        view_871: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_325, [256, 128, 128]);  addmm_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_327: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_871, add_326);  view_871 = add_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_261: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_327, primals_1004)
        add_328: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_261, primals_1005);  mul_261 = primals_1005 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_872: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_328, [32768, 128])
        permute_436: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1006, [1, 0])
        addmm_326: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_1007, view_872, permute_436);  primals_1007 = permute_436 = None
        view_873: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_326, [256, 128, 512]);  addmm_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_86: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_873);  view_873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_874: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_86, [32768, 512])
        permute_437: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1008, [1, 0])
        addmm_327: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1009, view_874, permute_437);  primals_1009 = permute_437 = None
        view_875: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_327, [256, 128, 128]);  addmm_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_329: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_875, add_328);  view_875 = add_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_262: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_329, primals_1010)
        add_330: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_262, primals_1011);  mul_262 = primals_1011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_876: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_330, [32768, 128])
        permute_438: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1012, [1, 0])
        addmm_328: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_1013, view_876, permute_438);  primals_1013 = permute_438 = None
        view_877: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_328, [256, 128, 512]);  addmm_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_87: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_877);  view_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_878: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_87, [32768, 512])
        permute_439: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1014, [1, 0])
        addmm_329: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1015, view_878, permute_439);  primals_1015 = permute_439 = None
        view_879: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_329, [256, 128, 128]);  addmm_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_331: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_879, add_330);  view_879 = add_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_263: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_331, primals_1016)
        add_332: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_263, primals_1017);  mul_263 = primals_1017 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_880: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_332, [32768, 128]);  add_332 = None
        permute_440: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1018, [1, 0])
        addmm_330: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_1019, view_880, permute_440);  primals_1019 = permute_440 = None
        view_881: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_330, [256, 128, 512]);  addmm_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_333: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_881, add_319);  view_881 = add_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_264: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_333, primals_1020)
        add_334: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_264, primals_1021);  mul_264 = primals_1021 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_882: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_334, [32768, 512])
        permute_441: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1022, [1, 0])
        addmm_331: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1023, view_882, permute_441);  primals_1023 = permute_441 = None
        view_883: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_331, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_265: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_883, primals_1024);  view_883 = None
        add_335: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_265, primals_1025);  mul_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_442: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1026, [1, 0])
        addmm_332: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1027, view_882, permute_442);  primals_1027 = permute_442 = None
        view_885: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_332, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_266: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_885, primals_1028);  view_885 = None
        add_336: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_266, primals_1029);  mul_266 = primals_1029 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_886: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_336, [32768, 128]);  add_336 = None
        permute_443: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1030, [1, 0])
        addmm_333: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1031, view_886, permute_443);  primals_1031 = permute_443 = None
        view_887: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_333, [256, 128, 128]);  addmm_333 = None
        view_888: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_887, [256, 128, -1, 32]);  view_887 = None
        permute_444: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_888, [0, 2, 1, 3]);  view_888 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_445: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1032, [1, 0])
        addmm_334: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1033, view_886, permute_445);  primals_1033 = permute_445 = None
        view_890: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_334, [256, 128, 128]);  addmm_334 = None
        view_891: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_890, [256, 128, -1, 32]);  view_890 = None
        permute_446: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_891, [0, 2, 1, 3]);  view_891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_447: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1034, [1, 0])
        addmm_335: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1035, view_882, permute_447);  primals_1035 = permute_447 = None
        view_893: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_335, [256, 128, 128]);  addmm_335 = None
        view_894: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_893, [256, 128, -1, 32]);  view_893 = None
        permute_448: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_894, [0, 2, 1, 3]);  view_894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_267: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_444, 0.4204482076268573);  permute_444 = None
        permute_449: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_446, [0, 1, 3, 2]);  permute_446 = None
        mul_268: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_449, 0.4204482076268573);  permute_449 = None
        expand_89: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_267, [256, 4, 128, 32]);  mul_267 = None
        clone_111: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_895: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_111, [1024, 128, 32]);  clone_111 = None
        expand_90: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_268, [256, 4, 32, 128]);  mul_268 = None
        clone_112: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_896: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_112, [1024, 32, 128]);  clone_112 = None
        bmm_44: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_895, view_896)
        view_897: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_44, [256, 4, 128, 128]);  bmm_44 = None
        add_337: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_897, where);  view_897 = None
        amax_22: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_337, [-1], True)
        sub_22: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_337, amax_22);  amax_22 = None
        exp_22: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_23: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        eq_22: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_337, -inf);  add_337 = None
        logical_not_44: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_22);  eq_22 = None
        any_23: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_44, -1, True);  logical_not_44 = None
        logical_not_45: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_23);  any_23 = None
        where_45: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_45, full_default_3, div_22);  logical_not_45 = div_22 = None
        inductor_lookup_seed_default_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_1: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        gt_22: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default_1, 0.1);  inductor_random_default_1 = None
        mul_269: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_22, where_45)
        mul_270: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_269, 1.1111111111111112);  mul_269 = None
        expand_91: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_270, [256, 4, 128, 128]);  mul_270 = None
        view_898: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_91, [1024, 128, 128]);  expand_91 = None
        expand_92: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_448, [256, 4, 128, 32]);  permute_448 = None
        clone_113: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_899: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_113, [1024, 128, 32]);  clone_113 = None
        bmm_45: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_898, view_899)
        view_900: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_45, [256, 4, 128, 32]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_450: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_900, [0, 2, 1, 3]);  view_900 = None
        clone_114: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_450, memory_format = torch.contiguous_format);  permute_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_901: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_114, [256, 128, -1]);  clone_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_902: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_901, [32768, 128]);  view_901 = None
        permute_451: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1036, [1, 0])
        addmm_336: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1037, view_902, permute_451);  primals_1037 = permute_451 = None
        view_903: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_336, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_338: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_903, add_335);  view_903 = add_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_271: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_338, primals_1038);  add_338 = None
        add_339: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_271, primals_1039);  mul_271 = primals_1039 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_904: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_339, [32768, 128])
        permute_452: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1040, [1, 0])
        addmm_337: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_1041, view_904, permute_452);  primals_1041 = permute_452 = None
        view_905: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_337, [256, 128, 512]);  addmm_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_88: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_905);  view_905 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_906: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_88, [32768, 512])
        permute_453: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1042, [1, 0])
        addmm_338: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1043, view_906, permute_453);  primals_1043 = permute_453 = None
        view_907: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_338, [256, 128, 128]);  addmm_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_340: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_907, add_339);  view_907 = add_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_272: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_340, primals_1044)
        add_341: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_272, primals_1045);  mul_272 = primals_1045 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_908: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_341, [32768, 128])
        permute_454: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1046, [1, 0])
        addmm_339: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_1047, view_908, permute_454);  primals_1047 = permute_454 = None
        view_909: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_339, [256, 128, 512]);  addmm_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_89: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_909);  view_909 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_910: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_89, [32768, 512])
        permute_455: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1048, [1, 0])
        addmm_340: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1049, view_910, permute_455);  primals_1049 = permute_455 = None
        view_911: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_340, [256, 128, 128]);  addmm_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_342: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_911, add_341);  view_911 = add_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_273: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_342, primals_1050)
        add_343: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_273, primals_1051);  mul_273 = primals_1051 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_912: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_343, [32768, 128])
        permute_456: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1052, [1, 0])
        addmm_341: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_1053, view_912, permute_456);  primals_1053 = permute_456 = None
        view_913: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_341, [256, 128, 512]);  addmm_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_90: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_913);  view_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_914: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_90, [32768, 512])
        permute_457: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1054, [1, 0])
        addmm_342: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1055, view_914, permute_457);  primals_1055 = permute_457 = None
        view_915: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_342, [256, 128, 128]);  addmm_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_344: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_915, add_343);  view_915 = add_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_274: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_344, primals_1056)
        add_345: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_274, primals_1057);  mul_274 = primals_1057 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_916: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_345, [32768, 128])
        permute_458: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1058, [1, 0])
        addmm_343: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_1059, view_916, permute_458);  primals_1059 = permute_458 = None
        view_917: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_343, [256, 128, 512]);  addmm_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_91: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_917);  view_917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_918: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_91, [32768, 512])
        permute_459: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1060, [1, 0])
        addmm_344: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1061, view_918, permute_459);  primals_1061 = permute_459 = None
        view_919: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_344, [256, 128, 128]);  addmm_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_346: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_919, add_345);  view_919 = add_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_275: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_346, primals_1062)
        add_347: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_275, primals_1063);  mul_275 = primals_1063 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_920: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_347, [32768, 128]);  add_347 = None
        permute_460: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1064, [1, 0])
        addmm_345: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_1065, view_920, permute_460);  primals_1065 = permute_460 = None
        view_921: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_345, [256, 128, 512]);  addmm_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_348: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_921, add_334);  view_921 = add_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_276: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_348, primals_1066)
        add_349: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_276, primals_1067);  mul_276 = primals_1067 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        view_922: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_349, [32768, 512])
        permute_461: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1068, [1, 0])
        addmm_346: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1069, view_922, permute_461);  primals_1069 = permute_461 = None
        view_923: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_346, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_277: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_923, primals_1070);  view_923 = None
        add_350: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_277, primals_1071);  mul_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:330 in forward, code: layer_input = self.dense(hidden_states)
        permute_462: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1072, [1, 0])
        addmm_347: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1073, view_922, permute_462);  primals_1073 = permute_462 = None
        view_925: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_347, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_278: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_925, primals_1074);  view_925 = None
        add_351: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_278, primals_1075);  mul_278 = primals_1075 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:203 in forward, code: query_layer = self.query(query_tensor).view(*hidden_shape).transpose(1, 2)
        view_926: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_351, [32768, 128]);  add_351 = None
        permute_463: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1076, [1, 0])
        addmm_348: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1077, view_926, permute_463);  primals_1077 = permute_463 = None
        view_927: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_348, [256, 128, 128]);  addmm_348 = None
        view_928: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_927, [256, 128, -1, 32]);  view_927 = None
        permute_464: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_928, [0, 2, 1, 3]);  view_928 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:204 in forward, code: key_layer = self.key(key_tensor).view(*hidden_shape).transpose(1, 2)
        permute_465: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1078, [1, 0])
        addmm_349: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1079, view_926, permute_465);  primals_1079 = permute_465 = None
        view_930: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_349, [256, 128, 128]);  addmm_349 = None
        view_931: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_930, [256, 128, -1, 32]);  view_930 = None
        permute_466: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_931, [0, 2, 1, 3]);  view_931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:205 in forward, code: value_layer = self.value(value_tensor).view(*hidden_shape).transpose(1, 2)
        permute_467: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1080, [1, 0])
        addmm_350: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1081, view_922, permute_467);  primals_1081 = permute_467 = None
        view_933: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_350, [256, 128, 128]);  addmm_350 = None
        view_934: "f32[256, 128, 4, 32]" = torch.ops.aten.reshape.default(view_933, [256, 128, -1, 32]);  view_933 = None
        permute_468: "f32[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_934, [0, 2, 1, 3]);  view_934 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_279: "f32[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_464, 0.4204482076268573);  permute_464 = None
        permute_469: "f32[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_466, [0, 1, 3, 2]);  permute_466 = None
        mul_280: "f32[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_469, 0.4204482076268573);  permute_469 = None
        expand_93: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_279, [256, 4, 128, 32]);  mul_279 = None
        clone_116: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_935: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_116, [1024, 128, 32]);  clone_116 = None
        expand_94: "f32[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_280, [256, 4, 32, 128]);  mul_280 = None
        clone_117: "f32[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
        view_936: "f32[1024, 32, 128]" = torch.ops.aten.reshape.default(clone_117, [1024, 32, 128]);  clone_117 = None
        bmm_46: "f32[1024, 128, 128]" = torch.ops.aten.bmm.default(view_935, view_936)
        view_937: "f32[256, 4, 128, 128]" = torch.ops.aten.reshape.default(bmm_46, [256, 4, 128, 128]);  bmm_46 = None
        add_352: "f32[256, 4, 128, 128]" = torch.ops.aten.add.Tensor(view_937, where);  view_937 = where = None
        amax_23: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(add_352, [-1], True)
        sub_23: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(add_352, amax_23);  amax_23 = None
        exp_23: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_24: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        eq_23: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(add_352, -inf);  add_352 = None
        logical_not_46: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_23);  eq_23 = None
        any_24: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_46, -1, True);  logical_not_46 = None
        logical_not_47: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_24);  any_24 = None
        where_47: "f32[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_47, full_default_3, div_23);  logical_not_47 = full_default_3 = div_23 = None
        inductor_lookup_seed_default_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23);  inductor_seeds_default = None
        inductor_random_default: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default([256, 4, 128, 128], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        gt_23: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_281: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_23, where_47)
        mul_282: "f32[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_281, 1.1111111111111112);  mul_281 = None
        expand_95: "f32[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_282, [256, 4, 128, 128]);  mul_282 = None
        view_938: "f32[1024, 128, 128]" = torch.ops.aten.reshape.default(expand_95, [1024, 128, 128]);  expand_95 = None
        expand_96: "f32[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_468, [256, 4, 128, 32]);  permute_468 = None
        clone_118: "f32[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
        view_939: "f32[1024, 128, 32]" = torch.ops.aten.reshape.default(clone_118, [1024, 128, 32]);  clone_118 = None
        bmm_47: "f32[1024, 128, 32]" = torch.ops.aten.bmm.default(view_938, view_939)
        view_940: "f32[256, 4, 128, 32]" = torch.ops.aten.reshape.default(bmm_47, [256, 4, 128, 32]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_470: "f32[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_940, [0, 2, 1, 3]);  view_940 = None
        clone_119: "f32[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_470, memory_format = torch.contiguous_format);  permute_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:221 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_941: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(clone_119, [256, 128, -1]);  clone_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:235 in forward, code: layer_outputs = self.dense(hidden_states)
        view_942: "f32[32768, 128]" = torch.ops.aten.reshape.default(view_941, [32768, 128]);  view_941 = None
        permute_471: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1082, [1, 0])
        addmm_351: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1083, view_942, permute_471);  primals_1083 = permute_471 = None
        view_943: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_351, [256, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:238 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_353: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_943, add_350);  view_943 = add_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_283: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_353, primals_1084);  add_353 = None
        add_354: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_283, primals_1085);  mul_283 = primals_1085 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_944: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_354, [32768, 128])
        permute_472: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1086, [1, 0])
        addmm_352: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_1087, view_944, permute_472);  primals_1087 = permute_472 = None
        view_945: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_352, [256, 128, 512]);  addmm_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_92: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_945);  view_945 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_946: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_92, [32768, 512])
        permute_473: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1088, [1, 0])
        addmm_353: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1089, view_946, permute_473);  primals_1089 = permute_473 = None
        view_947: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_353, [256, 128, 128]);  addmm_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_355: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_947, add_354);  view_947 = add_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_284: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_355, primals_1090)
        add_356: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_284, primals_1091);  mul_284 = primals_1091 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_948: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_356, [32768, 128])
        permute_474: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1092, [1, 0])
        addmm_354: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_1093, view_948, permute_474);  primals_1093 = permute_474 = None
        view_949: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_354, [256, 128, 512]);  addmm_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_93: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_949);  view_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_950: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_93, [32768, 512])
        permute_475: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1094, [1, 0])
        addmm_355: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1095, view_950, permute_475);  primals_1095 = permute_475 = None
        view_951: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_355, [256, 128, 128]);  addmm_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_357: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_951, add_356);  view_951 = add_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_285: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_357, primals_1096)
        add_358: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_285, primals_1097);  mul_285 = primals_1097 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_952: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_358, [32768, 128])
        permute_476: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1098, [1, 0])
        addmm_356: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_1099, view_952, permute_476);  primals_1099 = permute_476 = None
        view_953: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_356, [256, 128, 512]);  addmm_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_94: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_953);  view_953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:378 in forward, code: layer_outputs = self.dense(hidden_states)
        view_954: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_94, [32768, 512])
        permute_477: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1100, [1, 0])
        addmm_357: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1101, view_954, permute_477);  primals_1101 = permute_477 = None
        view_955: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_357, [256, 128, 128]);  addmm_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:379 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_359: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_955, add_358);  view_955 = add_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_286: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_359, primals_1102)
        add_360: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_286, primals_1103);  mul_286 = primals_1103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:280 in forward, code: hidden_states = self.dense(hidden_states)
        view_956: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_360, [32768, 128])
        permute_478: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1104, [1, 0])
        addmm_358: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_1105, view_956, permute_478);  primals_1105 = permute_478 = None
        view_957: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_358, [256, 128, 512]);  addmm_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        relu_95: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_957);  view_957 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:313 in forward, code: layer_output = self.dense(intermediate_states)
        view_958: "f32[32768, 512]" = torch.ops.aten.reshape.default(relu_95, [32768, 512])
        permute_479: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1106, [1, 0])
        addmm_359: "f32[32768, 128]" = torch.ops.aten.addmm.default(primals_1107, view_958, permute_479);  primals_1107 = permute_479 = None
        view_959: "f32[256, 128, 128]" = torch.ops.aten.reshape.default(addmm_359, [256, 128, 128]);  addmm_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:318 in forward, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        add_361: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_959, add_360);  view_959 = add_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_287: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_361, primals_1108)
        add_362: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_287, primals_1109);  mul_287 = primals_1109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:293 in forward, code: layer_outputs = self.dense(hidden_states)
        view_960: "f32[32768, 128]" = torch.ops.aten.reshape.default(add_362, [32768, 128]);  add_362 = None
        permute_480: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1110, [1, 0])
        addmm_360: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_1111, view_960, permute_480);  primals_1111 = permute_480 = None
        view_961: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_360, [256, 128, 512]);  addmm_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:295 in forward, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        add_363: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_961, add_349);  view_961 = add_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:62 in forward, code: return input_tensor * self.weight + self.bias
        mul_288: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_363, primals_1112)
        add_364: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_288, primals_1113);  mul_288 = primals_1113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:489 in forward, code: hidden_states = self.dense(hidden_states)
        view_962: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_364, [32768, 512]);  add_364 = None
        permute_481: "f32[512, 512]" = torch.ops.aten.permute.default(primals_1114, [1, 0])
        addmm_361: "f32[32768, 512]" = torch.ops.aten.addmm.default(primals_1115, view_962, permute_481);  primals_1115 = permute_481 = None
        view_963: "f32[256, 128, 512]" = torch.ops.aten.reshape.default(addmm_361, [256, 128, 512])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:490 in forward, code: hidden_states = self.transform_act_fn(hidden_states)
        relu_96: "f32[256, 128, 512]" = torch.ops.aten.relu.default(view_963);  view_963 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:491 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(relu_96, [2], correction = 0, keepdim = True)
        getitem: "f32[256, 128, 1]" = var_mean[0]
        getitem_1: "f32[256, 128, 1]" = var_mean[1];  var_mean = None
        add_365: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_365);  add_365 = None
        sub_24: "f32[256, 128, 512]" = torch.ops.aten.sub.Tensor(relu_96, getitem_1);  relu_96 = None
        mul_289: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt);  sub_24 = None
        mul_290: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(mul_289, primals_1116);  mul_289 = None
        add_366: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_290, primals_1117);  mul_290 = primals_1117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:507 in forward, code: hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        permute_482: "f32[128, 30522]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        cat_1: "f32[512, 30522]" = torch.ops.aten.cat.default([permute_482, primals_1118]);  permute_482 = primals_1118 = None
        view_964: "f32[32768, 512]" = torch.ops.aten.reshape.default(add_366, [32768, 512]);  add_366 = None
        constant_pad_nd_default_3: "f32[512, 30524]" = torch.ops.aten.constant_pad_nd.default(cat_1, [0, 2, 0, 0])
        mm_default_2: "f32[32768, 30524]" = torch.ops.aten.mm.default(view_964, constant_pad_nd_default_3);  constant_pad_nd_default_3 = None
        slice_tensor_1: "f32[32768, 30522]" = torch.ops.aten.slice.Tensor(mm_default_2, 1, 0, -2);  mm_default_2 = None
        view_965: "f32[256, 128, 30522]" = torch.ops.aten.reshape.default(slice_tensor_1, [256, 128, 30522]);  slice_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:508 in forward, code: hidden_states += self.decoder.bias
        add_367: "f32[256, 128, 30522]" = torch.ops.aten.add.Tensor(view_965, primals_1119);  view_965 = primals_1119 = None
        view_966: "f32[32768, 30522]" = torch.ops.aten.reshape.default(add_367, [32768, 30522]);  add_367 = None
        view_967: "f32[256, 128, 30522]" = torch.ops.aten.reshape.default(view_966, [256, 128, 30522])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:825 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        view_969: "i64[32768]" = torch.ops.aten.reshape.default(primals_1120, [-1])
        amax_24: "f32[32768, 1]" = torch.ops.aten.amax.default(view_966, [1], True)
        sub_25: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(view_966, amax_24);  view_966 = amax_24 = None
        exp_24: "f32[32768, 30522]" = torch.ops.aten.exp.default(sub_25)
        sum_25: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[32768, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_26: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(sub_25, log);  sub_25 = log = None
        ne: "b8[32768]" = torch.ops.aten.ne.Scalar(view_969, -100)
        full_default_73: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_48: "i64[32768]" = torch.ops.aten.where.self(ne, view_969, full_default_73);  view_969 = full_default_73 = None
        unsqueeze_3: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(where_48, 1);  where_48 = None
        gather: "f32[32768, 1]" = torch.ops.aten.gather.default(sub_26, 1, unsqueeze_3);  unsqueeze_3 = None
        squeeze: "f32[32768]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[32768]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_49: "f32[32768]" = torch.ops.aten.where.self(ne, neg, full_default_2);  neg = full_default_2 = None
        sum_26: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27: "f32[]" = torch.ops.aten.sum.default(where_49);  where_49 = None
        div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = None
        exp_25: "f32[32768, 30522]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:507 in forward, code: hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        permute_483: "f32[512, 32768]" = torch.ops.aten.permute.default(view_964, [1, 0]);  view_964 = None
        permute_484: "f32[30522, 512]" = torch.ops.aten.permute.default(cat_1, [1, 0]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_1: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_95, 0);  relu_95 = None
        le_2: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_94, 0);  relu_94 = None
        le_3: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_93, 0);  relu_93 = None
        le_4: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_92, 0);  relu_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_531: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_938, [0, 2, 1]);  view_938 = None
        permute_532: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_939, [0, 2, 1]);  view_939 = None
        permute_533: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_935, [0, 2, 1]);  view_935 = None
        permute_534: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_936, [0, 2, 1]);  view_936 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_5: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_91, 0);  relu_91 = None
        le_6: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_90, 0);  relu_90 = None
        le_7: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_89, 0);  relu_89 = None
        le_8: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_88, 0);  relu_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_600: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_898, [0, 2, 1]);  view_898 = None
        permute_601: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_899, [0, 2, 1]);  view_899 = None
        permute_602: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_895, [0, 2, 1]);  view_895 = None
        permute_603: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_896, [0, 2, 1]);  view_896 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_9: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_87, 0);  relu_87 = None
        le_10: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_86, 0);  relu_86 = None
        le_11: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_85, 0);  relu_85 = None
        le_12: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_84, 0);  relu_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_669: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_858, [0, 2, 1]);  view_858 = None
        permute_670: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_859, [0, 2, 1]);  view_859 = None
        permute_671: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_855, [0, 2, 1]);  view_855 = None
        permute_672: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_856, [0, 2, 1]);  view_856 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_13: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_83, 0);  relu_83 = None
        le_14: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_82, 0);  relu_82 = None
        le_15: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_81, 0);  relu_81 = None
        le_16: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_80, 0);  relu_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_738: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_818, [0, 2, 1]);  view_818 = None
        permute_739: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_819, [0, 2, 1]);  view_819 = None
        permute_740: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_815, [0, 2, 1]);  view_815 = None
        permute_741: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_816, [0, 2, 1]);  view_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_17: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_79, 0);  relu_79 = None
        le_18: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_78, 0);  relu_78 = None
        le_19: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_77, 0);  relu_77 = None
        le_20: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_76, 0);  relu_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_807: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_778, [0, 2, 1]);  view_778 = None
        permute_808: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_779, [0, 2, 1]);  view_779 = None
        permute_809: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_775, [0, 2, 1]);  view_775 = None
        permute_810: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_776, [0, 2, 1]);  view_776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_21: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_75, 0);  relu_75 = None
        le_22: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_74, 0);  relu_74 = None
        le_23: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_73, 0);  relu_73 = None
        le_24: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_72, 0);  relu_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_876: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_738, [0, 2, 1]);  view_738 = None
        permute_877: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_739, [0, 2, 1]);  view_739 = None
        permute_878: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_735, [0, 2, 1]);  view_735 = None
        permute_879: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_736, [0, 2, 1]);  view_736 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_25: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_71, 0);  relu_71 = None
        le_26: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_70, 0);  relu_70 = None
        le_27: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_69, 0);  relu_69 = None
        le_28: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_68, 0);  relu_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_945: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_698, [0, 2, 1]);  view_698 = None
        permute_946: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_699, [0, 2, 1]);  view_699 = None
        permute_947: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_695, [0, 2, 1]);  view_695 = None
        permute_948: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_696, [0, 2, 1]);  view_696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_29: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_67, 0);  relu_67 = None
        le_30: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_66, 0);  relu_66 = None
        le_31: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_65, 0);  relu_65 = None
        le_32: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_64, 0);  relu_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_1014: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_658, [0, 2, 1]);  view_658 = None
        permute_1015: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_659, [0, 2, 1]);  view_659 = None
        permute_1016: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_655, [0, 2, 1]);  view_655 = None
        permute_1017: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_656, [0, 2, 1]);  view_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_33: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_63, 0);  relu_63 = None
        le_34: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_62, 0);  relu_62 = None
        le_35: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_61, 0);  relu_61 = None
        le_36: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_1083: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_618, [0, 2, 1]);  view_618 = None
        permute_1084: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_619, [0, 2, 1]);  view_619 = None
        permute_1085: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_615, [0, 2, 1]);  view_615 = None
        permute_1086: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_616, [0, 2, 1]);  view_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_37: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_59, 0);  relu_59 = None
        le_38: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_58, 0);  relu_58 = None
        le_39: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_57, 0);  relu_57 = None
        le_40: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_56, 0);  relu_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_1152: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_578, [0, 2, 1]);  view_578 = None
        permute_1153: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_579, [0, 2, 1]);  view_579 = None
        permute_1154: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_575, [0, 2, 1]);  view_575 = None
        permute_1155: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_576, [0, 2, 1]);  view_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_41: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_55, 0);  relu_55 = None
        le_42: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_54, 0);  relu_54 = None
        le_43: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_53, 0);  relu_53 = None
        le_44: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_52, 0);  relu_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_1221: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_538, [0, 2, 1]);  view_538 = None
        permute_1222: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_539, [0, 2, 1]);  view_539 = None
        permute_1223: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_535, [0, 2, 1]);  view_535 = None
        permute_1224: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_536, [0, 2, 1]);  view_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_45: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_51, 0);  relu_51 = None
        le_46: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
        le_47: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
        le_48: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_1290: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_498, [0, 2, 1]);  view_498 = None
        permute_1291: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_499, [0, 2, 1]);  view_499 = None
        permute_1292: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_495, [0, 2, 1]);  view_495 = None
        permute_1293: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_496, [0, 2, 1]);  view_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_49: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
        le_50: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
        le_51: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
        le_52: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_1359: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_458, [0, 2, 1]);  view_458 = None
        permute_1360: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_459, [0, 2, 1]);  view_459 = None
        permute_1361: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_455, [0, 2, 1]);  view_455 = None
        permute_1362: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_456, [0, 2, 1]);  view_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_53: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
        le_54: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
        le_55: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
        le_56: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_1428: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_418, [0, 2, 1]);  view_418 = None
        permute_1429: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_419, [0, 2, 1]);  view_419 = None
        permute_1430: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_415, [0, 2, 1]);  view_415 = None
        permute_1431: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_416, [0, 2, 1]);  view_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_57: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
        le_58: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
        le_59: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
        le_60: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_1497: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_378, [0, 2, 1]);  view_378 = None
        permute_1498: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_379, [0, 2, 1]);  view_379 = None
        permute_1499: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_375, [0, 2, 1]);  view_375 = None
        permute_1500: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_376, [0, 2, 1]);  view_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_61: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        le_62: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        le_63: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        le_64: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_1566: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_338, [0, 2, 1]);  view_338 = None
        permute_1567: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None
        permute_1568: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_335, [0, 2, 1]);  view_335 = None
        permute_1569: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_336, [0, 2, 1]);  view_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_65: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        le_66: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        le_67: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        le_68: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_1635: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_298, [0, 2, 1]);  view_298 = None
        permute_1636: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None
        permute_1637: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_295, [0, 2, 1]);  view_295 = None
        permute_1638: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_69: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        le_70: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        le_71: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        le_72: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_1704: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_258, [0, 2, 1]);  view_258 = None
        permute_1705: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_259, [0, 2, 1]);  view_259 = None
        permute_1706: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
        permute_1707: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_256, [0, 2, 1]);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_73: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        le_74: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        le_75: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        le_76: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_1773: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_218, [0, 2, 1]);  view_218 = None
        permute_1774: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_219, [0, 2, 1]);  view_219 = None
        permute_1775: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_215, [0, 2, 1]);  view_215 = None
        permute_1776: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_216, [0, 2, 1]);  view_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_77: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        le_78: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        le_79: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        le_80: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_1842: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_178, [0, 2, 1]);  view_178 = None
        permute_1843: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None
        permute_1844: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_175, [0, 2, 1]);  view_175 = None
        permute_1845: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_81: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        le_82: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        le_83: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        le_84: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_1911: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_138, [0, 2, 1]);  view_138 = None
        permute_1912: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_139, [0, 2, 1]);  view_139 = None
        permute_1913: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
        permute_1914: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_85: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        le_86: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        le_87: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        le_88: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_1980: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
        permute_1981: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        permute_1982: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
        permute_1983: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_96, [0, 2, 1]);  view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_89: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        le_90: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        le_91: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        le_92: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_2049: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
        permute_2050: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
        permute_2051: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
        permute_2052: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:281 in forward, code: hidden_states = self.intermediate_act_fn(hidden_states)
        le_93: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        le_94: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        le_95: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        le_96: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_2118: "f32[1024, 128, 128]" = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None
        permute_2119: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_19, [0, 2, 1]);  view_19 = None
        permute_2120: "f32[1024, 32, 128]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        permute_2121: "f32[1024, 128, 32]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
        return (div_24, view_967, primals_1, primals_2, primals_4, primals_8, primals_10, primals_12, primals_13, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_59, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_105, primals_106, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_134, primals_136, primals_138, primals_140, primals_142, primals_144, primals_146, primals_148, primals_150, primals_151, primals_152, primals_154, primals_156, primals_158, primals_160, primals_162, primals_164, primals_166, primals_168, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_197, primals_198, primals_200, primals_202, primals_204, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_218, primals_220, primals_222, primals_224, primals_226, primals_228, primals_230, primals_232, primals_234, primals_236, primals_238, primals_240, primals_242, primals_243, primals_244, primals_246, primals_248, primals_250, primals_252, primals_254, primals_256, primals_258, primals_260, primals_262, primals_264, primals_266, primals_268, primals_270, primals_272, primals_274, primals_276, primals_278, primals_280, primals_282, primals_284, primals_286, primals_288, primals_289, primals_290, primals_292, primals_294, primals_296, primals_298, primals_300, primals_302, primals_304, primals_306, primals_308, primals_310, primals_312, primals_314, primals_316, primals_318, primals_320, primals_322, primals_324, primals_326, primals_328, primals_330, primals_332, primals_334, primals_335, primals_336, primals_338, primals_340, primals_342, primals_344, primals_346, primals_348, primals_350, primals_352, primals_354, primals_356, primals_358, primals_360, primals_362, primals_364, primals_366, primals_368, primals_370, primals_372, primals_374, primals_376, primals_378, primals_380, primals_381, primals_382, primals_384, primals_386, primals_388, primals_390, primals_392, primals_394, primals_396, primals_398, primals_400, primals_402, primals_404, primals_406, primals_408, primals_410, primals_412, primals_414, primals_416, primals_418, primals_420, primals_422, primals_424, primals_426, primals_427, primals_428, primals_430, primals_432, primals_434, primals_436, primals_438, primals_440, primals_442, primals_444, primals_446, primals_448, primals_450, primals_452, primals_454, primals_456, primals_458, primals_460, primals_462, primals_464, primals_466, primals_468, primals_470, primals_472, primals_473, primals_474, primals_476, primals_478, primals_480, primals_482, primals_484, primals_486, primals_488, primals_490, primals_492, primals_494, primals_496, primals_498, primals_500, primals_502, primals_504, primals_506, primals_508, primals_510, primals_512, primals_514, primals_516, primals_518, primals_519, primals_520, primals_522, primals_524, primals_526, primals_528, primals_530, primals_532, primals_534, primals_536, primals_538, primals_540, primals_542, primals_544, primals_546, primals_548, primals_550, primals_552, primals_554, primals_556, primals_558, primals_560, primals_562, primals_564, primals_565, primals_566, primals_568, primals_570, primals_572, primals_574, primals_576, primals_578, primals_580, primals_582, primals_584, primals_586, primals_588, primals_590, primals_592, primals_594, primals_596, primals_598, primals_600, primals_602, primals_604, primals_606, primals_608, primals_610, primals_611, primals_612, primals_614, primals_616, primals_618, primals_620, primals_622, primals_624, primals_626, primals_628, primals_630, primals_632, primals_634, primals_636, primals_638, primals_640, primals_642, primals_644, primals_646, primals_648, primals_650, primals_652, primals_654, primals_656, primals_657, primals_658, primals_660, primals_662, primals_664, primals_666, primals_668, primals_670, primals_672, primals_674, primals_676, primals_678, primals_680, primals_682, primals_684, primals_686, primals_688, primals_690, primals_692, primals_694, primals_696, primals_698, primals_700, primals_702, primals_703, primals_704, primals_706, primals_708, primals_710, primals_712, primals_714, primals_716, primals_718, primals_720, primals_722, primals_724, primals_726, primals_728, primals_730, primals_732, primals_734, primals_736, primals_738, primals_740, primals_742, primals_744, primals_746, primals_748, primals_749, primals_750, primals_752, primals_754, primals_756, primals_758, primals_760, primals_762, primals_764, primals_766, primals_768, primals_770, primals_772, primals_774, primals_776, primals_778, primals_780, primals_782, primals_784, primals_786, primals_788, primals_790, primals_792, primals_794, primals_795, primals_796, primals_798, primals_800, primals_802, primals_804, primals_806, primals_808, primals_810, primals_812, primals_814, primals_816, primals_818, primals_820, primals_822, primals_824, primals_826, primals_828, primals_830, primals_832, primals_834, primals_836, primals_838, primals_840, primals_841, primals_842, primals_844, primals_846, primals_848, primals_850, primals_852, primals_854, primals_856, primals_858, primals_860, primals_862, primals_864, primals_866, primals_868, primals_870, primals_872, primals_874, primals_876, primals_878, primals_880, primals_882, primals_884, primals_886, primals_887, primals_888, primals_890, primals_892, primals_894, primals_896, primals_898, primals_900, primals_902, primals_904, primals_906, primals_908, primals_910, primals_912, primals_914, primals_916, primals_918, primals_920, primals_922, primals_924, primals_926, primals_928, primals_930, primals_932, primals_933, primals_934, primals_936, primals_938, primals_940, primals_942, primals_944, primals_946, primals_948, primals_950, primals_952, primals_954, primals_956, primals_958, primals_960, primals_962, primals_964, primals_966, primals_968, primals_970, primals_972, primals_974, primals_976, primals_978, primals_979, primals_980, primals_982, primals_984, primals_986, primals_988, primals_990, primals_992, primals_994, primals_996, primals_998, primals_1000, primals_1002, primals_1004, primals_1006, primals_1008, primals_1010, primals_1012, primals_1014, primals_1016, primals_1018, primals_1020, primals_1022, primals_1024, primals_1025, primals_1026, primals_1028, primals_1030, primals_1032, primals_1034, primals_1036, primals_1038, primals_1040, primals_1042, primals_1044, primals_1046, primals_1048, primals_1050, primals_1052, primals_1054, primals_1056, primals_1058, primals_1060, primals_1062, primals_1064, primals_1066, primals_1068, primals_1070, primals_1071, primals_1072, primals_1074, primals_1076, primals_1078, primals_1080, primals_1082, primals_1084, primals_1086, primals_1088, primals_1090, primals_1092, primals_1094, primals_1096, primals_1098, primals_1100, primals_1102, primals_1104, primals_1106, primals_1108, primals_1110, primals_1112, primals_1114, primals_1116, primals_1120, full_default, view, add_1, ge, view_2, addmm_1, addmm_2, view_6, bmm, amax, sum_1, logical_not_1, gt, view_22, addmm_6, view_24, view_26, add_10, view_28, view_30, add_12, view_32, view_34, add_14, view_36, view_38, add_16, view_40, add_18, view_42, addmm_16, addmm_17, view_46, where_3, gt_1, view_62, addmm_21, view_64, view_66, add_25, view_68, view_70, add_27, view_72, view_74, add_29, view_76, view_78, add_31, view_80, add_33, view_82, addmm_31, addmm_32, view_86, where_5, gt_2, view_102, addmm_36, view_104, view_106, add_40, view_108, view_110, add_42, view_112, view_114, add_44, view_116, view_118, add_46, view_120, add_48, view_122, addmm_46, addmm_47, view_126, where_7, gt_3, view_142, addmm_51, view_144, view_146, add_55, view_148, view_150, add_57, view_152, view_154, add_59, view_156, view_158, add_61, view_160, add_63, view_162, addmm_61, addmm_62, view_166, where_9, gt_4, view_182, addmm_66, view_184, view_186, add_70, view_188, view_190, add_72, view_192, view_194, add_74, view_196, view_198, add_76, view_200, add_78, view_202, addmm_76, addmm_77, view_206, where_11, gt_5, view_222, addmm_81, view_224, view_226, add_85, view_228, view_230, add_87, view_232, view_234, add_89, view_236, view_238, add_91, view_240, add_93, view_242, addmm_91, addmm_92, view_246, where_13, gt_6, view_262, addmm_96, view_264, view_266, add_100, view_268, view_270, add_102, view_272, view_274, add_104, view_276, view_278, add_106, view_280, add_108, view_282, addmm_106, addmm_107, view_286, where_15, gt_7, view_302, addmm_111, view_304, view_306, add_115, view_308, view_310, add_117, view_312, view_314, add_119, view_316, view_318, add_121, view_320, add_123, view_322, addmm_121, addmm_122, view_326, where_17, gt_8, view_342, addmm_126, view_344, view_346, add_130, view_348, view_350, add_132, view_352, view_354, add_134, view_356, view_358, add_136, view_360, add_138, view_362, addmm_136, addmm_137, view_366, where_19, gt_9, view_382, addmm_141, view_384, view_386, add_145, view_388, view_390, add_147, view_392, view_394, add_149, view_396, view_398, add_151, view_400, add_153, view_402, addmm_151, addmm_152, view_406, where_21, gt_10, view_422, addmm_156, view_424, view_426, add_160, view_428, view_430, add_162, view_432, view_434, add_164, view_436, view_438, add_166, view_440, add_168, view_442, addmm_166, addmm_167, view_446, where_23, gt_11, view_462, addmm_171, view_464, view_466, add_175, view_468, view_470, add_177, view_472, view_474, add_179, view_476, view_478, add_181, view_480, add_183, view_482, addmm_181, addmm_182, view_486, where_25, gt_12, view_502, addmm_186, view_504, view_506, add_190, view_508, view_510, add_192, view_512, view_514, add_194, view_516, view_518, add_196, view_520, add_198, view_522, addmm_196, addmm_197, view_526, where_27, gt_13, view_542, addmm_201, view_544, view_546, add_205, view_548, view_550, add_207, view_552, view_554, add_209, view_556, view_558, add_211, view_560, add_213, view_562, addmm_211, addmm_212, view_566, where_29, gt_14, view_582, addmm_216, view_584, view_586, add_220, view_588, view_590, add_222, view_592, view_594, add_224, view_596, view_598, add_226, view_600, add_228, view_602, addmm_226, addmm_227, view_606, where_31, gt_15, view_622, addmm_231, view_624, view_626, add_235, view_628, view_630, add_237, view_632, view_634, add_239, view_636, view_638, add_241, view_640, add_243, view_642, addmm_241, addmm_242, view_646, where_33, gt_16, view_662, addmm_246, view_664, view_666, add_250, view_668, view_670, add_252, view_672, view_674, add_254, view_676, view_678, add_256, view_680, add_258, view_682, addmm_256, addmm_257, view_686, where_35, gt_17, view_702, addmm_261, view_704, view_706, add_265, view_708, view_710, add_267, view_712, view_714, add_269, view_716, view_718, add_271, view_720, add_273, view_722, addmm_271, addmm_272, view_726, where_37, gt_18, view_742, addmm_276, view_744, view_746, add_280, view_748, view_750, add_282, view_752, view_754, add_284, view_756, view_758, add_286, view_760, add_288, view_762, addmm_286, addmm_287, view_766, where_39, gt_19, view_782, addmm_291, view_784, view_786, add_295, view_788, view_790, add_297, view_792, view_794, add_299, view_796, view_798, add_301, view_800, add_303, view_802, addmm_301, addmm_302, view_806, where_41, gt_20, view_822, addmm_306, view_824, view_826, add_310, view_828, view_830, add_312, view_832, view_834, add_314, view_836, view_838, add_316, view_840, add_318, view_842, addmm_316, addmm_317, view_846, where_43, gt_21, view_862, addmm_321, view_864, view_866, add_325, view_868, view_870, add_327, view_872, view_874, add_329, view_876, view_878, add_331, view_880, add_333, view_882, addmm_331, addmm_332, view_886, where_45, gt_22, view_902, addmm_336, view_904, view_906, add_340, view_908, view_910, add_342, view_912, view_914, add_344, view_916, view_918, add_346, view_920, add_348, view_922, addmm_346, addmm_347, view_926, where_47, gt_23, view_942, addmm_351, view_944, view_946, add_355, view_948, view_950, add_357, view_952, view_954, add_359, view_956, view_958, add_361, view_960, add_363, view_962, addmm_361, getitem_1, rsqrt, convert_element_type, exp_25, permute_483, permute_484, le_1, le_2, le_3, le_4, permute_531, permute_532, permute_533, permute_534, le_5, le_6, le_7, le_8, permute_600, permute_601, permute_602, permute_603, le_9, le_10, le_11, le_12, permute_669, permute_670, permute_671, permute_672, le_13, le_14, le_15, le_16, permute_738, permute_739, permute_740, permute_741, le_17, le_18, le_19, le_20, permute_807, permute_808, permute_809, permute_810, le_21, le_22, le_23, le_24, permute_876, permute_877, permute_878, permute_879, le_25, le_26, le_27, le_28, permute_945, permute_946, permute_947, permute_948, le_29, le_30, le_31, le_32, permute_1014, permute_1015, permute_1016, permute_1017, le_33, le_34, le_35, le_36, permute_1083, permute_1084, permute_1085, permute_1086, le_37, le_38, le_39, le_40, permute_1152, permute_1153, permute_1154, permute_1155, le_41, le_42, le_43, le_44, permute_1221, permute_1222, permute_1223, permute_1224, le_45, le_46, le_47, le_48, permute_1290, permute_1291, permute_1292, permute_1293, le_49, le_50, le_51, le_52, permute_1359, permute_1360, permute_1361, permute_1362, le_53, le_54, le_55, le_56, permute_1428, permute_1429, permute_1430, permute_1431, le_57, le_58, le_59, le_60, permute_1497, permute_1498, permute_1499, permute_1500, le_61, le_62, le_63, le_64, permute_1566, permute_1567, permute_1568, permute_1569, le_65, le_66, le_67, le_68, permute_1635, permute_1636, permute_1637, permute_1638, le_69, le_70, le_71, le_72, permute_1704, permute_1705, permute_1706, permute_1707, le_73, le_74, le_75, le_76, permute_1773, permute_1774, permute_1775, permute_1776, le_77, le_78, le_79, le_80, permute_1842, permute_1843, permute_1844, permute_1845, le_81, le_82, le_83, le_84, permute_1911, permute_1912, permute_1913, permute_1914, le_85, le_86, le_87, le_88, permute_1980, permute_1981, permute_1982, permute_1983, le_89, le_90, le_91, le_92, permute_2049, permute_2050, permute_2051, permute_2052, le_93, le_94, le_95, le_96, permute_2118, permute_2119, permute_2120, permute_2121)
