class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[128, 3, 224, 224]", primals_2: "f32[128, 3, 4, 4]", primals_4: "f32[128]", primals_5: "f32[128]", primals_6: "f32[128]", primals_8: "f32[384, 128]", primals_11: "i64[49, 49]", primals_12: "f32[128, 128]", primals_14: "f32[128]", primals_16: "f32[512, 128]", primals_18: "f32[128, 512]", primals_20: "f32[128]", primals_23: "f32[384, 128]", primals_26: "i64[49, 49]", primals_27: "f32[128, 128]", primals_29: "f32[128]", primals_31: "f32[512, 128]", primals_33: "f32[128, 512]", primals_35: "f32[512]", primals_37: "f32[256, 512]", primals_38: "f32[256]", primals_40: "f32[768, 256]", primals_43: "i64[49, 49]", primals_44: "f32[256, 256]", primals_46: "f32[256]", primals_48: "f32[1024, 256]", primals_50: "f32[256, 1024]", primals_52: "f32[256]", primals_55: "f32[768, 256]", primals_58: "i64[49, 49]", primals_59: "f32[256, 256]", primals_61: "f32[256]", primals_63: "f32[1024, 256]", primals_65: "f32[256, 1024]", primals_67: "f32[1024]", primals_69: "f32[512, 1024]", primals_70: "f32[512]", primals_72: "f32[1536, 512]", primals_75: "i64[49, 49]", primals_76: "f32[512, 512]", primals_78: "f32[512]", primals_80: "f32[2048, 512]", primals_82: "f32[512, 2048]", primals_84: "f32[512]", primals_87: "f32[1536, 512]", primals_90: "i64[49, 49]", primals_91: "f32[512, 512]", primals_93: "f32[512]", primals_95: "f32[2048, 512]", primals_97: "f32[512, 2048]", primals_99: "f32[512]", primals_101: "f32[1536, 512]", primals_104: "i64[49, 49]", primals_105: "f32[512, 512]", primals_107: "f32[512]", primals_109: "f32[2048, 512]", primals_111: "f32[512, 2048]", primals_113: "f32[512]", primals_116: "f32[1536, 512]", primals_119: "i64[49, 49]", primals_120: "f32[512, 512]", primals_122: "f32[512]", primals_124: "f32[2048, 512]", primals_126: "f32[512, 2048]", primals_128: "f32[512]", primals_130: "f32[1536, 512]", primals_133: "i64[49, 49]", primals_134: "f32[512, 512]", primals_136: "f32[512]", primals_138: "f32[2048, 512]", primals_140: "f32[512, 2048]", primals_142: "f32[512]", primals_145: "f32[1536, 512]", primals_148: "i64[49, 49]", primals_149: "f32[512, 512]", primals_151: "f32[512]", primals_153: "f32[2048, 512]", primals_155: "f32[512, 2048]", primals_157: "f32[512]", primals_159: "f32[1536, 512]", primals_162: "i64[49, 49]", primals_163: "f32[512, 512]", primals_165: "f32[512]", primals_167: "f32[2048, 512]", primals_169: "f32[512, 2048]", primals_171: "f32[512]", primals_174: "f32[1536, 512]", primals_177: "i64[49, 49]", primals_178: "f32[512, 512]", primals_180: "f32[512]", primals_182: "f32[2048, 512]", primals_184: "f32[512, 2048]", primals_186: "f32[512]", primals_188: "f32[1536, 512]", primals_191: "i64[49, 49]", primals_192: "f32[512, 512]", primals_194: "f32[512]", primals_196: "f32[2048, 512]", primals_198: "f32[512, 2048]", primals_200: "f32[512]", primals_203: "f32[1536, 512]", primals_206: "i64[49, 49]", primals_207: "f32[512, 512]", primals_209: "f32[512]", primals_211: "f32[2048, 512]", primals_213: "f32[512, 2048]", primals_215: "f32[512]", primals_217: "f32[1536, 512]", primals_220: "i64[49, 49]", primals_221: "f32[512, 512]", primals_223: "f32[512]", primals_225: "f32[2048, 512]", primals_227: "f32[512, 2048]", primals_229: "f32[512]", primals_232: "f32[1536, 512]", primals_235: "i64[49, 49]", primals_236: "f32[512, 512]", primals_238: "f32[512]", primals_240: "f32[2048, 512]", primals_242: "f32[512, 2048]", primals_244: "f32[512]", primals_246: "f32[1536, 512]", primals_249: "i64[49, 49]", primals_250: "f32[512, 512]", primals_252: "f32[512]", primals_254: "f32[2048, 512]", primals_256: "f32[512, 2048]", primals_258: "f32[512]", primals_261: "f32[1536, 512]", primals_264: "i64[49, 49]", primals_265: "f32[512, 512]", primals_267: "f32[512]", primals_269: "f32[2048, 512]", primals_271: "f32[512, 2048]", primals_273: "f32[512]", primals_275: "f32[1536, 512]", primals_278: "i64[49, 49]", primals_279: "f32[512, 512]", primals_281: "f32[512]", primals_283: "f32[2048, 512]", primals_285: "f32[512, 2048]", primals_287: "f32[512]", primals_290: "f32[1536, 512]", primals_293: "i64[49, 49]", primals_294: "f32[512, 512]", primals_296: "f32[512]", primals_298: "f32[2048, 512]", primals_300: "f32[512, 2048]", primals_302: "f32[512]", primals_304: "f32[1536, 512]", primals_307: "i64[49, 49]", primals_308: "f32[512, 512]", primals_310: "f32[512]", primals_312: "f32[2048, 512]", primals_314: "f32[512, 2048]", primals_316: "f32[512]", primals_319: "f32[1536, 512]", primals_322: "i64[49, 49]", primals_323: "f32[512, 512]", primals_325: "f32[512]", primals_327: "f32[2048, 512]", primals_329: "f32[512, 2048]", primals_331: "f32[2048]", primals_333: "f32[1024, 2048]", primals_334: "f32[1024]", primals_336: "f32[3072, 1024]", primals_339: "i64[49, 49]", primals_340: "f32[1024, 1024]", primals_342: "f32[1024]", primals_344: "f32[4096, 1024]", primals_346: "f32[1024, 4096]", primals_348: "f32[1024]", primals_350: "f32[3072, 1024]", primals_353: "i64[49, 49]", primals_354: "f32[1024, 1024]", primals_356: "f32[1024]", primals_358: "f32[4096, 1024]", primals_360: "f32[1024, 4096]", primals_362: "f32[1024]", primals_364: "f32[1000, 1024]", convolution: "f32[128, 128, 56, 56]", getitem_1: "f32[128, 56, 56, 1]", rsqrt: "f32[128, 56, 56, 1]", getitem_3: "f32[128, 56, 56, 1]", rsqrt_1: "f32[128, 56, 56, 1]", view_3: "f32[401408, 128]", div: "f32[8192, 4, 49, 49]", view_15: "f32[401408, 128]", mul_5: "f32[128, 3136, 128]", view_21: "f32[401408, 128]", addmm_2: "f32[401408, 512]", view_23: "f32[401408, 512]", mul_10: "f32[128, 56, 56, 128]", view_29: "f32[401408, 128]", div_1: "f32[8192, 4, 49, 49]", view_43: "f32[401408, 128]", fmod_2: "i64[56]", lt: "b8[128, 1, 1, 1]", mul_14: "f32[128, 3136, 128]", view_49: "f32[401408, 128]", addmm_6: "f32[401408, 512]", view_51: "f32[401408, 512]", lt_1: "b8[128, 1, 1]", mul_20: "f32[128, 28, 28, 512]", view_56: "f32[100352, 512]", mm: "f32[100352, 256]", getitem_19: "f32[128, 28, 28, 1]", rsqrt_6: "f32[128, 28, 28, 1]", view_61: "f32[100352, 256]", div_4: "f32[2048, 8, 49, 49]", view_73: "f32[100352, 256]", lt_2: "b8[128, 1, 1, 1]", mul_26: "f32[128, 784, 256]", view_79: "f32[100352, 256]", addmm_10: "f32[100352, 1024]", view_81: "f32[100352, 1024]", lt_3: "b8[128, 1, 1]", mul_32: "f32[128, 28, 28, 256]", view_87: "f32[100352, 256]", div_7: "f32[2048, 8, 49, 49]", view_101: "f32[100352, 256]", fmod_6: "i64[28]", lt_4: "b8[128, 1, 1, 1]", mul_36: "f32[128, 784, 256]", view_107: "f32[100352, 256]", addmm_14: "f32[100352, 1024]", view_109: "f32[100352, 1024]", lt_5: "b8[128, 1, 1]", mul_42: "f32[128, 14, 14, 1024]", view_114: "f32[25088, 1024]", mm_1: "f32[25088, 512]", getitem_35: "f32[128, 14, 14, 1]", rsqrt_11: "f32[128, 14, 14, 1]", view_119: "f32[25088, 512]", div_10: "f32[512, 16, 49, 49]", view_131: "f32[25088, 512]", lt_6: "b8[128, 1, 1, 1]", mul_48: "f32[128, 196, 512]", view_137: "f32[25088, 512]", addmm_18: "f32[25088, 2048]", view_139: "f32[25088, 2048]", lt_7: "b8[128, 1, 1]", mul_54: "f32[128, 14, 14, 512]", view_145: "f32[25088, 512]", div_13: "f32[512, 16, 49, 49]", view_159: "f32[25088, 512]", fmod_10: "i64[14]", lt_8: "b8[128, 1, 1, 1]", mul_58: "f32[128, 196, 512]", view_165: "f32[25088, 512]", addmm_22: "f32[25088, 2048]", view_167: "f32[25088, 2048]", lt_9: "b8[128, 1, 1]", mul_64: "f32[128, 14, 14, 512]", view_173: "f32[25088, 512]", div_16: "f32[512, 16, 49, 49]", view_185: "f32[25088, 512]", lt_10: "b8[128, 1, 1, 1]", mul_68: "f32[128, 196, 512]", view_191: "f32[25088, 512]", addmm_26: "f32[25088, 2048]", view_193: "f32[25088, 2048]", lt_11: "b8[128, 1, 1]", mul_74: "f32[128, 14, 14, 512]", view_199: "f32[25088, 512]", div_19: "f32[512, 16, 49, 49]", view_213: "f32[25088, 512]", lt_12: "b8[128, 1, 1, 1]", mul_78: "f32[128, 196, 512]", view_219: "f32[25088, 512]", addmm_30: "f32[25088, 2048]", view_221: "f32[25088, 2048]", lt_13: "b8[128, 1, 1]", mul_84: "f32[128, 14, 14, 512]", view_227: "f32[25088, 512]", div_22: "f32[512, 16, 49, 49]", view_239: "f32[25088, 512]", lt_14: "b8[128, 1, 1, 1]", mul_88: "f32[128, 196, 512]", view_245: "f32[25088, 512]", addmm_34: "f32[25088, 2048]", view_247: "f32[25088, 2048]", lt_15: "b8[128, 1, 1]", mul_94: "f32[128, 14, 14, 512]", view_253: "f32[25088, 512]", div_25: "f32[512, 16, 49, 49]", view_267: "f32[25088, 512]", lt_16: "b8[128, 1, 1, 1]", mul_98: "f32[128, 196, 512]", view_273: "f32[25088, 512]", addmm_38: "f32[25088, 2048]", view_275: "f32[25088, 2048]", lt_17: "b8[128, 1, 1]", mul_104: "f32[128, 14, 14, 512]", view_281: "f32[25088, 512]", div_28: "f32[512, 16, 49, 49]", view_293: "f32[25088, 512]", lt_18: "b8[128, 1, 1, 1]", mul_108: "f32[128, 196, 512]", view_299: "f32[25088, 512]", addmm_42: "f32[25088, 2048]", view_301: "f32[25088, 2048]", lt_19: "b8[128, 1, 1]", mul_114: "f32[128, 14, 14, 512]", view_307: "f32[25088, 512]", div_31: "f32[512, 16, 49, 49]", view_321: "f32[25088, 512]", lt_20: "b8[128, 1, 1, 1]", mul_118: "f32[128, 196, 512]", view_327: "f32[25088, 512]", addmm_46: "f32[25088, 2048]", view_329: "f32[25088, 2048]", lt_21: "b8[128, 1, 1]", mul_124: "f32[128, 14, 14, 512]", view_335: "f32[25088, 512]", div_34: "f32[512, 16, 49, 49]", view_347: "f32[25088, 512]", lt_22: "b8[128, 1, 1, 1]", mul_128: "f32[128, 196, 512]", view_353: "f32[25088, 512]", addmm_50: "f32[25088, 2048]", view_355: "f32[25088, 2048]", lt_23: "b8[128, 1, 1]", mul_134: "f32[128, 14, 14, 512]", view_361: "f32[25088, 512]", div_37: "f32[512, 16, 49, 49]", view_375: "f32[25088, 512]", lt_24: "b8[128, 1, 1, 1]", mul_138: "f32[128, 196, 512]", view_381: "f32[25088, 512]", addmm_54: "f32[25088, 2048]", view_383: "f32[25088, 2048]", lt_25: "b8[128, 1, 1]", mul_144: "f32[128, 14, 14, 512]", view_389: "f32[25088, 512]", div_40: "f32[512, 16, 49, 49]", view_401: "f32[25088, 512]", lt_26: "b8[128, 1, 1, 1]", mul_148: "f32[128, 196, 512]", view_407: "f32[25088, 512]", addmm_58: "f32[25088, 2048]", view_409: "f32[25088, 2048]", lt_27: "b8[128, 1, 1]", mul_154: "f32[128, 14, 14, 512]", view_415: "f32[25088, 512]", div_43: "f32[512, 16, 49, 49]", view_429: "f32[25088, 512]", lt_28: "b8[128, 1, 1, 1]", mul_158: "f32[128, 196, 512]", view_435: "f32[25088, 512]", addmm_62: "f32[25088, 2048]", view_437: "f32[25088, 2048]", lt_29: "b8[128, 1, 1]", mul_164: "f32[128, 14, 14, 512]", view_443: "f32[25088, 512]", div_46: "f32[512, 16, 49, 49]", view_455: "f32[25088, 512]", lt_30: "b8[128, 1, 1, 1]", mul_168: "f32[128, 196, 512]", view_461: "f32[25088, 512]", addmm_66: "f32[25088, 2048]", view_463: "f32[25088, 2048]", lt_31: "b8[128, 1, 1]", mul_174: "f32[128, 14, 14, 512]", view_469: "f32[25088, 512]", div_49: "f32[512, 16, 49, 49]", view_483: "f32[25088, 512]", lt_32: "b8[128, 1, 1, 1]", mul_178: "f32[128, 196, 512]", view_489: "f32[25088, 512]", addmm_70: "f32[25088, 2048]", view_491: "f32[25088, 2048]", lt_33: "b8[128, 1, 1]", mul_184: "f32[128, 14, 14, 512]", view_497: "f32[25088, 512]", div_52: "f32[512, 16, 49, 49]", view_509: "f32[25088, 512]", lt_34: "b8[128, 1, 1, 1]", mul_188: "f32[128, 196, 512]", view_515: "f32[25088, 512]", addmm_74: "f32[25088, 2048]", view_517: "f32[25088, 2048]", lt_35: "b8[128, 1, 1]", mul_194: "f32[128, 14, 14, 512]", view_523: "f32[25088, 512]", div_55: "f32[512, 16, 49, 49]", view_537: "f32[25088, 512]", lt_36: "b8[128, 1, 1, 1]", mul_198: "f32[128, 196, 512]", view_543: "f32[25088, 512]", addmm_78: "f32[25088, 2048]", view_545: "f32[25088, 2048]", lt_37: "b8[128, 1, 1]", mul_204: "f32[128, 14, 14, 512]", view_551: "f32[25088, 512]", div_58: "f32[512, 16, 49, 49]", view_563: "f32[25088, 512]", lt_38: "b8[128, 1, 1, 1]", mul_208: "f32[128, 196, 512]", view_569: "f32[25088, 512]", addmm_82: "f32[25088, 2048]", view_571: "f32[25088, 2048]", lt_39: "b8[128, 1, 1]", mul_214: "f32[128, 14, 14, 512]", view_577: "f32[25088, 512]", div_61: "f32[512, 16, 49, 49]", view_591: "f32[25088, 512]", lt_40: "b8[128, 1, 1, 1]", mul_218: "f32[128, 196, 512]", view_597: "f32[25088, 512]", addmm_86: "f32[25088, 2048]", view_599: "f32[25088, 2048]", lt_41: "b8[128, 1, 1]", mul_224: "f32[128, 7, 7, 2048]", view_604: "f32[6272, 2048]", mm_2: "f32[6272, 1024]", getitem_163: "f32[128, 7, 7, 1]", rsqrt_48: "f32[128, 7, 7, 1]", view_609: "f32[6272, 1024]", div_64: "f32[128, 32, 49, 49]", view_621: "f32[6272, 1024]", lt_42: "b8[128, 1, 1, 1]", mul_230: "f32[128, 49, 1024]", view_627: "f32[6272, 1024]", addmm_90: "f32[6272, 4096]", view_629: "f32[6272, 4096]", lt_43: "b8[128, 1, 1]", mul_236: "f32[128, 7, 7, 1024]", view_635: "f32[6272, 1024]", div_67: "f32[128, 32, 49, 49]", view_647: "f32[6272, 1024]", lt_44: "b8[128, 1, 1, 1]", mul_240: "f32[128, 49, 1024]", view_653: "f32[6272, 1024]", addmm_94: "f32[6272, 4096]", view_655: "f32[6272, 4096]", lt_45: "b8[128, 1, 1]", mul_246: "f32[128, 7, 7, 1024]", mean: "f32[128, 1024]", div_71: "f32[128, 7, 7, 1]", div_72: "f32[128, 49, 1]", permute_267: "f32[4096, 32, 49]", permute_269: "f32[4096, 32, 49]", permute_270: "f32[4096, 49, 32]", div_73: "f32[128, 7, 7, 1]", div_74: "f32[128, 49, 1]", permute_293: "f32[4096, 32, 49]", permute_295: "f32[4096, 32, 49]", permute_296: "f32[4096, 49, 32]", div_76: "f32[128, 7, 7, 1]", div_77: "f32[128, 196, 1]", permute_324: "f32[8192, 32, 49]", permute_326: "f32[8192, 32, 49]", permute_327: "f32[8192, 49, 32]", div_78: "f32[128, 14, 14, 1]", div_79: "f32[128, 196, 1]", permute_350: "f32[8192, 32, 49]", permute_352: "f32[8192, 32, 49]", permute_353: "f32[8192, 49, 32]", div_80: "f32[128, 14, 14, 1]", div_81: "f32[128, 196, 1]", permute_376: "f32[8192, 32, 49]", permute_378: "f32[8192, 32, 49]", permute_379: "f32[8192, 49, 32]", div_82: "f32[128, 14, 14, 1]", div_83: "f32[128, 196, 1]", permute_402: "f32[8192, 32, 49]", permute_404: "f32[8192, 32, 49]", permute_405: "f32[8192, 49, 32]", div_84: "f32[128, 14, 14, 1]", div_85: "f32[128, 196, 1]", permute_428: "f32[8192, 32, 49]", permute_430: "f32[8192, 32, 49]", permute_431: "f32[8192, 49, 32]", div_86: "f32[128, 14, 14, 1]", div_87: "f32[128, 196, 1]", permute_454: "f32[8192, 32, 49]", permute_456: "f32[8192, 32, 49]", permute_457: "f32[8192, 49, 32]", div_88: "f32[128, 14, 14, 1]", div_89: "f32[128, 196, 1]", permute_480: "f32[8192, 32, 49]", permute_482: "f32[8192, 32, 49]", permute_483: "f32[8192, 49, 32]", div_90: "f32[128, 14, 14, 1]", div_91: "f32[128, 196, 1]", permute_506: "f32[8192, 32, 49]", permute_508: "f32[8192, 32, 49]", permute_509: "f32[8192, 49, 32]", div_92: "f32[128, 14, 14, 1]", div_93: "f32[128, 196, 1]", permute_532: "f32[8192, 32, 49]", permute_534: "f32[8192, 32, 49]", permute_535: "f32[8192, 49, 32]", div_94: "f32[128, 14, 14, 1]", div_95: "f32[128, 196, 1]", permute_558: "f32[8192, 32, 49]", permute_560: "f32[8192, 32, 49]", permute_561: "f32[8192, 49, 32]", div_96: "f32[128, 14, 14, 1]", div_97: "f32[128, 196, 1]", permute_584: "f32[8192, 32, 49]", permute_586: "f32[8192, 32, 49]", permute_587: "f32[8192, 49, 32]", div_98: "f32[128, 14, 14, 1]", div_99: "f32[128, 196, 1]", permute_610: "f32[8192, 32, 49]", permute_612: "f32[8192, 32, 49]", permute_613: "f32[8192, 49, 32]", div_100: "f32[128, 14, 14, 1]", div_101: "f32[128, 196, 1]", permute_636: "f32[8192, 32, 49]", permute_638: "f32[8192, 32, 49]", permute_639: "f32[8192, 49, 32]", div_102: "f32[128, 14, 14, 1]", div_103: "f32[128, 196, 1]", permute_662: "f32[8192, 32, 49]", permute_664: "f32[8192, 32, 49]", permute_665: "f32[8192, 49, 32]", div_104: "f32[128, 14, 14, 1]", div_105: "f32[128, 196, 1]", permute_688: "f32[8192, 32, 49]", permute_690: "f32[8192, 32, 49]", permute_691: "f32[8192, 49, 32]", div_106: "f32[128, 14, 14, 1]", div_107: "f32[128, 196, 1]", permute_714: "f32[8192, 32, 49]", permute_716: "f32[8192, 32, 49]", permute_717: "f32[8192, 49, 32]", div_108: "f32[128, 14, 14, 1]", div_109: "f32[128, 196, 1]", permute_740: "f32[8192, 32, 49]", permute_742: "f32[8192, 32, 49]", permute_743: "f32[8192, 49, 32]", div_110: "f32[128, 14, 14, 1]", div_111: "f32[128, 196, 1]", permute_766: "f32[8192, 32, 49]", permute_768: "f32[8192, 32, 49]", permute_769: "f32[8192, 49, 32]", div_113: "f32[128, 14, 14, 1]", div_114: "f32[128, 784, 1]", permute_797: "f32[16384, 32, 49]", permute_799: "f32[16384, 32, 49]", permute_800: "f32[16384, 49, 32]", div_115: "f32[128, 28, 28, 1]", div_116: "f32[128, 784, 1]", permute_823: "f32[16384, 32, 49]", permute_825: "f32[16384, 32, 49]", permute_826: "f32[16384, 49, 32]", div_118: "f32[128, 28, 28, 1]", div_119: "f32[128, 3136, 1]", permute_854: "f32[32768, 32, 49]", permute_856: "f32[32768, 32, 49]", permute_857: "f32[32768, 49, 32]", div_120: "f32[128, 56, 56, 1]", div_121: "f32[128, 3136, 1]", permute_880: "f32[32768, 32, 49]", permute_882: "f32[32768, 32, 49]", permute_883: "f32[32768, 49, 32]", tangents_1: "f32[128, 1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_247: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_364, [1, 0]);  primals_364 = None
        permute_248: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
        mm_3: "f32[128, 1024]" = torch.ops.aten.mm.default(tangents_1, permute_248);  permute_248 = None
        permute_249: "f32[1000, 128]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_4: "f32[1000, 1024]" = torch.ops.aten.mm.default(permute_249, mean);  permute_249 = mean = None
        sum_25: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_658: "f32[1000]" = torch.ops.aten.reshape.default(sum_25, [1000]);  sum_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:65 in forward, code: return x.mean(self.dim, keepdim=not self.flatten)
        unsqueeze_46: "f32[128, 1, 1024]" = torch.ops.aten.unsqueeze.default(mm_3, 1);  mm_3 = None
        unsqueeze_47: "f32[128, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 2);  unsqueeze_46 = None
        expand_96: "f32[128, 7, 7, 1024]" = torch.ops.aten.expand.default(unsqueeze_47, [128, 7, 7, 1024]);  unsqueeze_47 = None
        div_70: "f32[128, 7, 7, 1024]" = torch.ops.aten.div.Scalar(expand_96, 49);  expand_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:981 in forward_features, code: x = self.norm(x)
        mul_249: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_70, primals_362);  primals_362 = None
        mul_250: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_249, 1024)
        sum_26: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [3], True)
        mul_251: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_249, mul_246);  mul_249 = None
        sum_27: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [3], True);  mul_251 = None
        mul_252: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_246, sum_27);  sum_27 = None
        sub_78: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_250, sum_26);  mul_250 = sum_26 = None
        sub_79: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_78, mul_252);  sub_78 = mul_252 = None
        mul_253: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_71, sub_79);  div_71 = sub_79 = None
        mul_254: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_70, mul_246);  mul_246 = None
        sum_28: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_254, [0, 1, 2]);  mul_254 = None
        sum_29: "f32[1024]" = torch.ops.aten.sum.dim_IntList(div_70, [0, 1, 2]);  div_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_659: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(mul_253, [128, 49, 1024]);  mul_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_45: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_45, torch.float32);  lt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_69: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_45, 0.8999999985098839);  convert_element_type_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_255: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view_659, div_69);  div_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_660: "f32[6272, 1024]" = torch.ops.aten.reshape.default(mul_255, [6272, 1024]);  mul_255 = None
        permute_246: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_360, [1, 0]);  primals_360 = None
        permute_252: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
        mm_5: "f32[6272, 4096]" = torch.ops.aten.mm.default(view_660, permute_252);  permute_252 = None
        permute_253: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_660, [1, 0])
        mm_6: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_253, view_655);  permute_253 = view_655 = None
        sum_30: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_660, [0], True);  view_660 = None
        view_661: "f32[1024]" = torch.ops.aten.reshape.default(sum_30, [1024]);  sum_30 = None
        view_662: "f32[128, 49, 4096]" = torch.ops.aten.reshape.default(mm_5, [128, 49, 4096]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_654: "f32[128, 49, 4096]" = torch.ops.aten.reshape.default(addmm_94, [128, 49, 4096]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_243: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(view_654, 0.7071067811865476)
        erf_23: "f32[128, 49, 4096]" = torch.ops.aten.erf.default(mul_243);  mul_243 = None
        add_253: "f32[128, 49, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_257: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(add_253, 0.5);  add_253 = None
        mul_258: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(view_654, view_654)
        mul_259: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(mul_258, -0.5);  mul_258 = None
        exp_24: "f32[128, 49, 4096]" = torch.ops.aten.exp.default(mul_259);  mul_259 = None
        mul_260: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
        mul_261: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(view_654, mul_260);  view_654 = mul_260 = None
        add_258: "f32[128, 49, 4096]" = torch.ops.aten.add.Tensor(mul_257, mul_261);  mul_257 = mul_261 = None
        mul_262: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(view_662, add_258);  view_662 = add_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_663: "f32[6272, 4096]" = torch.ops.aten.reshape.default(mul_262, [6272, 4096]);  mul_262 = None
        permute_245: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_358, [1, 0]);  primals_358 = None
        permute_256: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_245, [1, 0]);  permute_245 = None
        mm_7: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_663, permute_256);  permute_256 = None
        permute_257: "f32[4096, 6272]" = torch.ops.aten.permute.default(view_663, [1, 0])
        mm_8: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_257, view_653);  permute_257 = view_653 = None
        sum_31: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_663, [0], True);  view_663 = None
        view_664: "f32[4096]" = torch.ops.aten.reshape.default(sum_31, [4096]);  sum_31 = None
        view_665: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(mm_7, [128, 49, 1024]);  mm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_264: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view_665, primals_356);  primals_356 = None
        mul_265: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_264, 1024)
        sum_32: "f32[128, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_264, [2], True)
        mul_266: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_264, mul_240);  mul_264 = None
        sum_33: "f32[128, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True);  mul_266 = None
        mul_267: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_240, sum_33);  sum_33 = None
        sub_81: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(mul_265, sum_32);  mul_265 = sum_32 = None
        sub_82: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(sub_81, mul_267);  sub_81 = mul_267 = None
        mul_268: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(div_72, sub_82);  div_72 = sub_82 = None
        mul_269: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view_665, mul_240);  mul_240 = None
        sum_34: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_269, [0, 1]);  mul_269 = None
        sum_35: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_665, [0, 1]);  view_665 = None
        add_259: "f32[128, 49, 1024]" = torch.ops.aten.add.Tensor(view_659, mul_268);  view_659 = mul_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_666: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(add_259, [128, 7, 7, 1024]);  add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_44: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_44, torch.float32);  lt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_68: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_44, 0.8999999985098839);  convert_element_type_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_270: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_666, div_68);  div_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_667: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.reshape.default(mul_270, [128, 1, 7, 1, 7, 1024]);  mul_270 = None
        permute_260: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_667, [0, 1, 3, 2, 4, 5]);  view_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_668: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_260, [128, 7, 7, 1024]);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_669: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(view_668, [128, 49, 1024]);  view_668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_670: "f32[6272, 1024]" = torch.ops.aten.reshape.default(view_669, [6272, 1024]);  view_669 = None
        permute_243: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_354, [1, 0]);  primals_354 = None
        permute_261: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
        mm_9: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_670, permute_261);  permute_261 = None
        permute_262: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_670, [1, 0])
        mm_10: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_262, view_647);  permute_262 = view_647 = None
        sum_36: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_670, [0], True);  view_670 = None
        view_671: "f32[1024]" = torch.ops.aten.reshape.default(sum_36, [1024]);  sum_36 = None
        view_672: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(mm_9, [128, 49, 1024]);  mm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_673: "f32[128, 49, 32, 32]" = torch.ops.aten.reshape.default(view_672, [128, 49, 32, 32]);  view_672 = None
        permute_265: "f32[128, 32, 49, 32]" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_264: "f32[128, 32, 49, 32]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
        view_674: "f32[4096, 49, 32]" = torch.ops.aten.reshape.default(clone_264, [4096, 49, 32]);  clone_264 = None
        expand_94: "f32[128, 32, 49, 49]" = torch.ops.aten.expand.default(div_67, [128, 32, 49, 49])
        view_643: "f32[4096, 49, 49]" = torch.ops.aten.reshape.default(expand_94, [4096, 49, 49]);  expand_94 = None
        permute_266: "f32[4096, 49, 49]" = torch.ops.aten.permute.default(view_643, [0, 2, 1]);  view_643 = None
        bmm_48: "f32[4096, 49, 32]" = torch.ops.aten.bmm.default(permute_266, view_674);  permute_266 = None
        bmm_49: "f32[4096, 49, 49]" = torch.ops.aten.bmm.default(view_674, permute_267);  view_674 = permute_267 = None
        view_675: "f32[128, 32, 49, 32]" = torch.ops.aten.reshape.default(bmm_48, [128, 32, 49, 32]);  bmm_48 = None
        view_676: "f32[128, 32, 49, 49]" = torch.ops.aten.reshape.default(bmm_49, [128, 32, 49, 49]);  bmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_271: "f32[128, 32, 49, 49]" = torch.ops.aten.mul.Tensor(view_676, div_67);  view_676 = None
        sum_37: "f32[128, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [-1], True)
        neg: "f32[128, 32, 49, 49]" = torch.ops.aten.neg.default(div_67);  div_67 = None
        fma: "f32[128, 32, 49, 49]" = torch.ops.prims.fma.default(neg, sum_37, mul_271);  neg = sum_37 = mul_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_38: "f32[1, 32, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze: "f32[32, 49, 49]" = torch.ops.aten.squeeze.dim(sum_38, 0);  sum_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_268: "f32[49, 49, 32]" = torch.ops.aten.permute.default(squeeze, [1, 2, 0]);  squeeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_677: "f32[2401, 32]" = torch.ops.aten.reshape.default(permute_268, [2401, 32]);  permute_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        full_default: "f32[169, 32]" = torch.ops.aten.full.default([169, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_641: "i64[2401]" = torch.ops.aten.reshape.default(primals_353, [-1]);  primals_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put: "f32[169, 32]" = torch.ops.aten.index_put.default(full_default, [view_641], view_677, True);  view_641 = view_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_678: "f32[4096, 49, 49]" = torch.ops.aten.reshape.default(fma, [4096, 49, 49]);  fma = None
        bmm_50: "f32[4096, 32, 49]" = torch.ops.aten.bmm.default(permute_269, view_678);  permute_269 = None
        bmm_51: "f32[4096, 49, 32]" = torch.ops.aten.bmm.default(view_678, permute_270);  view_678 = permute_270 = None
        view_679: "f32[128, 32, 32, 49]" = torch.ops.aten.reshape.default(bmm_50, [128, 32, 32, 49]);  bmm_50 = None
        view_680: "f32[128, 32, 49, 32]" = torch.ops.aten.reshape.default(bmm_51, [128, 32, 49, 32]);  bmm_51 = None
        permute_271: "f32[128, 32, 49, 32]" = torch.ops.aten.permute.default(view_679, [0, 1, 3, 2]);  view_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_272: "f32[128, 32, 49, 32]" = torch.ops.aten.mul.Tensor(view_680, 0.1767766952966369);  view_680 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat: "f32[384, 32, 49, 32]" = torch.ops.aten.cat.default([mul_272, permute_271, view_675]);  mul_272 = permute_271 = view_675 = None
        view_681: "f32[3, 128, 32, 49, 32]" = torch.ops.aten.reshape.default(cat, [3, 128, 32, 49, 32]);  cat = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_272: "f32[128, 49, 3, 32, 32]" = torch.ops.aten.permute.default(view_681, [1, 3, 0, 2, 4]);  view_681 = None
        clone_265: "f32[128, 49, 3, 32, 32]" = torch.ops.aten.clone.default(permute_272, memory_format = torch.contiguous_format);  permute_272 = None
        view_682: "f32[128, 49, 3072]" = torch.ops.aten.reshape.default(clone_265, [128, 49, 3072]);  clone_265 = None
        view_683: "f32[6272, 3072]" = torch.ops.aten.reshape.default(view_682, [6272, 3072]);  view_682 = None
        permute_238: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_350, [1, 0]);  primals_350 = None
        permute_273: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
        mm_11: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_683, permute_273);  permute_273 = None
        permute_274: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_683, [1, 0])
        mm_12: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_274, view_635);  permute_274 = view_635 = None
        sum_39: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_683, [0], True);  view_683 = None
        view_684: "f32[3072]" = torch.ops.aten.reshape.default(sum_39, [3072]);  sum_39 = None
        view_685: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(mm_11, [128, 49, 1024]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_686: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(view_685, [128, 7, 7, 1024]);  view_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_687: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.reshape.default(view_686, [128, 1, 1, 7, 7, 1024]);  view_686 = None
        permute_277: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_687, [0, 1, 3, 2, 4, 5]);  view_687 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_688: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_277, [128, 7, 7, 1024]);  permute_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_274: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_688, primals_348);  primals_348 = None
        mul_275: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_274, 1024)
        sum_40: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [3], True)
        mul_276: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_274, mul_236);  mul_274 = None
        sum_41: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [3], True);  mul_276 = None
        mul_277: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_236, sum_41);  sum_41 = None
        sub_84: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_275, sum_40);  mul_275 = sum_40 = None
        sub_85: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_84, mul_277);  sub_84 = mul_277 = None
        mul_278: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_73, sub_85);  div_73 = sub_85 = None
        mul_279: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_688, mul_236);  mul_236 = None
        sum_42: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_279, [0, 1, 2]);  mul_279 = None
        sum_43: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_688, [0, 1, 2]);  view_688 = None
        add_260: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_666, mul_278);  view_666 = mul_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_689: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(add_260, [128, 49, 1024]);  add_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_43: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_43, torch.float32);  lt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_66: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_43, 0.9043478220701218);  convert_element_type_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_280: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view_689, div_66);  div_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_690: "f32[6272, 1024]" = torch.ops.aten.reshape.default(mul_280, [6272, 1024]);  mul_280 = None
        permute_236: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_346, [1, 0]);  primals_346 = None
        permute_278: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
        mm_13: "f32[6272, 4096]" = torch.ops.aten.mm.default(view_690, permute_278);  permute_278 = None
        permute_279: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_690, [1, 0])
        mm_14: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_279, view_629);  permute_279 = view_629 = None
        sum_44: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_690, [0], True);  view_690 = None
        view_691: "f32[1024]" = torch.ops.aten.reshape.default(sum_44, [1024]);  sum_44 = None
        view_692: "f32[128, 49, 4096]" = torch.ops.aten.reshape.default(mm_13, [128, 49, 4096]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_628: "f32[128, 49, 4096]" = torch.ops.aten.reshape.default(addmm_90, [128, 49, 4096]);  addmm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_233: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(view_628, 0.7071067811865476)
        erf_22: "f32[128, 49, 4096]" = torch.ops.aten.erf.default(mul_233);  mul_233 = None
        add_245: "f32[128, 49, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_282: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(add_245, 0.5);  add_245 = None
        mul_283: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(view_628, view_628)
        mul_284: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(mul_283, -0.5);  mul_283 = None
        exp_25: "f32[128, 49, 4096]" = torch.ops.aten.exp.default(mul_284);  mul_284 = None
        mul_285: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
        mul_286: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(view_628, mul_285);  view_628 = mul_285 = None
        add_262: "f32[128, 49, 4096]" = torch.ops.aten.add.Tensor(mul_282, mul_286);  mul_282 = mul_286 = None
        mul_287: "f32[128, 49, 4096]" = torch.ops.aten.mul.Tensor(view_692, add_262);  view_692 = add_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_693: "f32[6272, 4096]" = torch.ops.aten.reshape.default(mul_287, [6272, 4096]);  mul_287 = None
        permute_235: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_344, [1, 0]);  primals_344 = None
        permute_282: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
        mm_15: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_693, permute_282);  permute_282 = None
        permute_283: "f32[4096, 6272]" = torch.ops.aten.permute.default(view_693, [1, 0])
        mm_16: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_283, view_627);  permute_283 = view_627 = None
        sum_45: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_693, [0], True);  view_693 = None
        view_694: "f32[4096]" = torch.ops.aten.reshape.default(sum_45, [4096]);  sum_45 = None
        view_695: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(mm_15, [128, 49, 1024]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_289: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view_695, primals_342);  primals_342 = None
        mul_290: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_289, 1024)
        sum_46: "f32[128, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_289, [2], True)
        mul_291: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_289, mul_230);  mul_289 = None
        sum_47: "f32[128, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True);  mul_291 = None
        mul_292: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(mul_230, sum_47);  sum_47 = None
        sub_87: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(mul_290, sum_46);  mul_290 = sum_46 = None
        sub_88: "f32[128, 49, 1024]" = torch.ops.aten.sub.Tensor(sub_87, mul_292);  sub_87 = mul_292 = None
        mul_293: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(div_74, sub_88);  div_74 = sub_88 = None
        mul_294: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view_695, mul_230);  mul_230 = None
        sum_48: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_294, [0, 1]);  mul_294 = None
        sum_49: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_695, [0, 1]);  view_695 = None
        add_263: "f32[128, 49, 1024]" = torch.ops.aten.add.Tensor(view_689, mul_293);  view_689 = mul_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_696: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(add_263, [128, 7, 7, 1024]);  add_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_42: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_42, torch.float32);  lt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_65: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_42, 0.9043478220701218);  convert_element_type_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_295: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_696, div_65);  div_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_697: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.reshape.default(mul_295, [128, 1, 7, 1, 7, 1024]);  mul_295 = None
        permute_286: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.permute.default(view_697, [0, 1, 3, 2, 4, 5]);  view_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_698: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_286, [128, 7, 7, 1024]);  permute_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_699: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(view_698, [128, 49, 1024]);  view_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_700: "f32[6272, 1024]" = torch.ops.aten.reshape.default(view_699, [6272, 1024]);  view_699 = None
        permute_233: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_340, [1, 0]);  primals_340 = None
        permute_287: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        mm_17: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_700, permute_287);  permute_287 = None
        permute_288: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_700, [1, 0])
        mm_18: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_288, view_621);  permute_288 = view_621 = None
        sum_50: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_700, [0], True);  view_700 = None
        view_701: "f32[1024]" = torch.ops.aten.reshape.default(sum_50, [1024]);  sum_50 = None
        view_702: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(mm_17, [128, 49, 1024]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_703: "f32[128, 49, 32, 32]" = torch.ops.aten.reshape.default(view_702, [128, 49, 32, 32]);  view_702 = None
        permute_291: "f32[128, 32, 49, 32]" = torch.ops.aten.permute.default(view_703, [0, 2, 1, 3]);  view_703 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_266: "f32[128, 32, 49, 32]" = torch.ops.aten.clone.default(permute_291, memory_format = torch.contiguous_format);  permute_291 = None
        view_704: "f32[4096, 49, 32]" = torch.ops.aten.reshape.default(clone_266, [4096, 49, 32]);  clone_266 = None
        expand_90: "f32[128, 32, 49, 49]" = torch.ops.aten.expand.default(div_64, [128, 32, 49, 49])
        view_617: "f32[4096, 49, 49]" = torch.ops.aten.reshape.default(expand_90, [4096, 49, 49]);  expand_90 = None
        permute_292: "f32[4096, 49, 49]" = torch.ops.aten.permute.default(view_617, [0, 2, 1]);  view_617 = None
        bmm_52: "f32[4096, 49, 32]" = torch.ops.aten.bmm.default(permute_292, view_704);  permute_292 = None
        bmm_53: "f32[4096, 49, 49]" = torch.ops.aten.bmm.default(view_704, permute_293);  view_704 = permute_293 = None
        view_705: "f32[128, 32, 49, 32]" = torch.ops.aten.reshape.default(bmm_52, [128, 32, 49, 32]);  bmm_52 = None
        view_706: "f32[128, 32, 49, 49]" = torch.ops.aten.reshape.default(bmm_53, [128, 32, 49, 49]);  bmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_296: "f32[128, 32, 49, 49]" = torch.ops.aten.mul.Tensor(view_706, div_64);  view_706 = None
        sum_51: "f32[128, 32, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [-1], True)
        neg_1: "f32[128, 32, 49, 49]" = torch.ops.aten.neg.default(div_64);  div_64 = None
        fma_1: "f32[128, 32, 49, 49]" = torch.ops.prims.fma.default(neg_1, sum_51, mul_296);  neg_1 = sum_51 = mul_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_52: "f32[1, 32, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_1, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_1: "f32[32, 49, 49]" = torch.ops.aten.squeeze.dim(sum_52, 0);  sum_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_294: "f32[49, 49, 32]" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_707: "f32[2401, 32]" = torch.ops.aten.reshape.default(permute_294, [2401, 32]);  permute_294 = None
        view_615: "i64[2401]" = torch.ops.aten.reshape.default(primals_339, [-1]);  primals_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_1: "f32[169, 32]" = torch.ops.aten.index_put.default(full_default, [view_615], view_707, True);  full_default = view_615 = view_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_708: "f32[4096, 49, 49]" = torch.ops.aten.reshape.default(fma_1, [4096, 49, 49]);  fma_1 = None
        bmm_54: "f32[4096, 32, 49]" = torch.ops.aten.bmm.default(permute_295, view_708);  permute_295 = None
        bmm_55: "f32[4096, 49, 32]" = torch.ops.aten.bmm.default(view_708, permute_296);  view_708 = permute_296 = None
        view_709: "f32[128, 32, 32, 49]" = torch.ops.aten.reshape.default(bmm_54, [128, 32, 32, 49]);  bmm_54 = None
        view_710: "f32[128, 32, 49, 32]" = torch.ops.aten.reshape.default(bmm_55, [128, 32, 49, 32]);  bmm_55 = None
        permute_297: "f32[128, 32, 49, 32]" = torch.ops.aten.permute.default(view_709, [0, 1, 3, 2]);  view_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_297: "f32[128, 32, 49, 32]" = torch.ops.aten.mul.Tensor(view_710, 0.1767766952966369);  view_710 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_1: "f32[384, 32, 49, 32]" = torch.ops.aten.cat.default([mul_297, permute_297, view_705]);  mul_297 = permute_297 = view_705 = None
        view_711: "f32[3, 128, 32, 49, 32]" = torch.ops.aten.reshape.default(cat_1, [3, 128, 32, 49, 32]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_298: "f32[128, 49, 3, 32, 32]" = torch.ops.aten.permute.default(view_711, [1, 3, 0, 2, 4]);  view_711 = None
        clone_267: "f32[128, 49, 3, 32, 32]" = torch.ops.aten.clone.default(permute_298, memory_format = torch.contiguous_format);  permute_298 = None
        view_712: "f32[128, 49, 3072]" = torch.ops.aten.reshape.default(clone_267, [128, 49, 3072]);  clone_267 = None
        view_713: "f32[6272, 3072]" = torch.ops.aten.reshape.default(view_712, [6272, 3072]);  view_712 = None
        permute_228: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_336, [1, 0]);  primals_336 = None
        permute_299: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
        mm_19: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_713, permute_299);  permute_299 = None
        permute_300: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_713, [1, 0])
        mm_20: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_300, view_609);  permute_300 = view_609 = None
        sum_53: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_713, [0], True);  view_713 = None
        view_714: "f32[3072]" = torch.ops.aten.reshape.default(sum_53, [3072]);  sum_53 = None
        view_715: "f32[128, 49, 1024]" = torch.ops.aten.reshape.default(mm_19, [128, 49, 1024]);  mm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_716: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(view_715, [128, 7, 7, 1024]);  view_715 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_717: "f32[128, 1, 1, 7, 7, 1024]" = torch.ops.aten.reshape.default(view_716, [128, 1, 1, 7, 7, 1024]);  view_716 = None
        permute_303: "f32[128, 1, 7, 1, 7, 1024]" = torch.ops.aten.permute.default(view_717, [0, 1, 3, 2, 4, 5]);  view_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_718: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(permute_303, [128, 7, 7, 1024]);  permute_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_299: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_718, primals_334);  primals_334 = None
        mul_300: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_299, 1024)
        sum_54: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        view_605: "f32[128, 7, 7, 1024]" = torch.ops.aten.reshape.default(mm_2, [128, 7, 7, 1024]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        sub_70: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(view_605, getitem_163);  view_605 = getitem_163 = None
        mul_226: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_48);  sub_70 = None
        mul_301: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_299, mul_226);  mul_299 = None
        sum_55: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [3], True);  mul_301 = None
        mul_302: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_226, sum_55);  sum_55 = None
        sub_90: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_300, sum_54);  mul_300 = sum_54 = None
        sub_91: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_90, mul_302);  sub_90 = mul_302 = None
        div_75: "f32[128, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 1024);  rsqrt_48 = None
        mul_303: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_75, sub_91);  div_75 = sub_91 = None
        mul_304: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_718, mul_226);  mul_226 = None
        sum_56: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1, 2]);  mul_304 = None
        sum_57: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_718, [0, 1, 2]);  view_718 = None
        add_264: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(view_696, mul_303);  view_696 = mul_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        view_719: "f32[6272, 1024]" = torch.ops.aten.reshape.default(add_264, [6272, 1024]);  add_264 = None
        permute_304: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_719, [1, 0])
        mm_21: "f32[1024, 2048]" = torch.ops.aten.mm.default(permute_304, view_604);  permute_304 = view_604 = None
        permute_226: "f32[2048, 1024]" = torch.ops.aten.permute.default(primals_333, [1, 0]);  primals_333 = None
        permute_306: "f32[1024, 2048]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
        mm_22: "f32[6272, 2048]" = torch.ops.aten.mm.default(view_719, permute_306);  view_719 = permute_306 = None
        view_720: "f32[128, 7, 7, 2048]" = torch.ops.aten.reshape.default(mm_22, [128, 7, 7, 2048]);  mm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        mul_306: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(view_720, primals_331);  primals_331 = None
        mul_307: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_306, 2048)
        sum_58: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_306, [3], True)
        mul_308: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_306, mul_224);  mul_306 = None
        sum_59: "f32[128, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_308, [3], True);  mul_308 = None
        mul_309: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(mul_224, sum_59);  sum_59 = None
        sub_93: "f32[128, 7, 7, 2048]" = torch.ops.aten.sub.Tensor(mul_307, sum_58);  mul_307 = sum_58 = None
        sub_94: "f32[128, 7, 7, 2048]" = torch.ops.aten.sub.Tensor(sub_93, mul_309);  sub_93 = mul_309 = None
        mul_310: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(div_76, sub_94);  div_76 = sub_94 = None
        mul_311: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(view_720, mul_224);  mul_224 = None
        sum_60: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_311, [0, 1, 2]);  mul_311 = None
        sum_61: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_720, [0, 1, 2]);  view_720 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_721: "f32[128, 7, 7, 2, 2, 512]" = torch.ops.aten.reshape.default(mul_310, [128, 7, 7, 2, 2, 512]);  mul_310 = None
        permute_308: "f32[128, 7, 2, 7, 2, 512]" = torch.ops.aten.permute.default(view_721, [0, 1, 4, 2, 3, 5]);  view_721 = None
        clone_268: "f32[128, 7, 2, 7, 2, 512]" = torch.ops.aten.clone.default(permute_308, memory_format = torch.contiguous_format);  permute_308 = None
        view_722: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_268, [128, 14, 14, 512]);  clone_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_723: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(view_722, [128, 196, 512]);  view_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_41: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_41, torch.float32);  lt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_63: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_41, 0.9086956530809402);  convert_element_type_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_312: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_723, div_63);  div_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_724: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_312, [25088, 512]);  mul_312 = None
        permute_224: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_329, [1, 0]);  primals_329 = None
        permute_309: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
        mm_23: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_724, permute_309);  permute_309 = None
        permute_310: "f32[512, 25088]" = torch.ops.aten.permute.default(view_724, [1, 0])
        mm_24: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_310, view_599);  permute_310 = view_599 = None
        sum_62: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_724, [0], True);  view_724 = None
        view_725: "f32[512]" = torch.ops.aten.reshape.default(sum_62, [512]);  sum_62 = None
        view_726: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_23, [128, 196, 2048]);  mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_598: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_86, [128, 196, 2048]);  addmm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_221: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_598, 0.7071067811865476)
        erf_21: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
        add_235: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_314: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_235, 0.5);  add_235 = None
        mul_315: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_598, view_598)
        mul_316: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_315, -0.5);  mul_315 = None
        exp_26: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_316);  mul_316 = None
        mul_317: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_318: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_598, mul_317);  view_598 = mul_317 = None
        add_266: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_314, mul_318);  mul_314 = mul_318 = None
        mul_319: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_726, add_266);  view_726 = add_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_727: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_319, [25088, 2048]);  mul_319 = None
        permute_223: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_327, [1, 0]);  primals_327 = None
        permute_313: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
        mm_25: "f32[25088, 512]" = torch.ops.aten.mm.default(view_727, permute_313);  permute_313 = None
        permute_314: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_727, [1, 0])
        mm_26: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_314, view_597);  permute_314 = view_597 = None
        sum_63: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_727, [0], True);  view_727 = None
        view_728: "f32[2048]" = torch.ops.aten.reshape.default(sum_63, [2048]);  sum_63 = None
        view_729: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_25, [128, 196, 512]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_321: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_729, primals_325);  primals_325 = None
        mul_322: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_321, 512)
        sum_64: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True)
        mul_323: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_321, mul_218);  mul_321 = None
        sum_65: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [2], True);  mul_323 = None
        mul_324: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_218, sum_65);  sum_65 = None
        sub_96: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_322, sum_64);  mul_322 = sum_64 = None
        sub_97: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_96, mul_324);  sub_96 = mul_324 = None
        mul_325: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_77, sub_97);  div_77 = sub_97 = None
        mul_326: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_729, mul_218);  mul_218 = None
        sum_66: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_326, [0, 1]);  mul_326 = None
        sum_67: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_729, [0, 1]);  view_729 = None
        add_267: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_723, mul_325);  view_723 = mul_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_730: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_267, [128, 14, 14, 512]);  add_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_40: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_40, torch.float32);  lt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_62: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_40, 0.9086956530809402);  convert_element_type_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_327: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_730, div_62);  div_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_8: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_58: "i64[14]" = torch.ops.aten.add.Tensor(iota_8, 3);  iota_8 = None
        fmod_8: "i64[14]" = torch.ops.aten.fmod.Scalar(add_58, 14);  add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_68: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_327, [None, None, fmod_8]);  mul_327 = None
        index_69: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_68, [None, fmod_8]);  index_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_731: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_69, [128, 2, 7, 2, 7, 512]);  index_69 = None
        permute_317: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_731, [0, 1, 3, 2, 4, 5]);  view_731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_269: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_317, memory_format = torch.contiguous_format);  permute_317 = None
        view_732: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_269, [512, 7, 7, 512]);  clone_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_733: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_732, [512, 49, 512]);  view_732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_734: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_733, [25088, 512]);  view_733 = None
        permute_221: "f32[512, 512]" = torch.ops.aten.permute.default(primals_323, [1, 0]);  primals_323 = None
        permute_318: "f32[512, 512]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
        mm_27: "f32[25088, 512]" = torch.ops.aten.mm.default(view_734, permute_318);  permute_318 = None
        permute_319: "f32[512, 25088]" = torch.ops.aten.permute.default(view_734, [1, 0])
        mm_28: "f32[512, 512]" = torch.ops.aten.mm.default(permute_319, view_591);  permute_319 = view_591 = None
        sum_68: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_734, [0], True);  view_734 = None
        view_735: "f32[512]" = torch.ops.aten.reshape.default(sum_68, [512]);  sum_68 = None
        view_736: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_27, [512, 49, 512]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_737: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_736, [512, 49, 16, 32]);  view_736 = None
        permute_322: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_737, [0, 2, 1, 3]);  view_737 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_270: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
        view_738: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_270, [8192, 49, 32]);  clone_270 = None
        expand_86: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_61, [512, 16, 49, 49])
        view_587: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_86, [8192, 49, 49]);  expand_86 = None
        permute_323: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_587, [0, 2, 1]);  view_587 = None
        bmm_56: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_323, view_738);  permute_323 = None
        bmm_57: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_738, permute_324);  view_738 = permute_324 = None
        view_739: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_56, [512, 16, 49, 32]);  bmm_56 = None
        view_740: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_57, [512, 16, 49, 49]);  bmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_328: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_740, div_61);  view_740 = None
        sum_69: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_328, [-1], True)
        neg_2: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_61);  div_61 = None
        fma_2: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_2, sum_69, mul_328);  neg_2 = sum_69 = mul_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_70: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_2, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_2: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_70, 0);  sum_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_325: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_2, [1, 2, 0]);  squeeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_743: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_325, [2401, 16]);  permute_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        full_default_2: "f32[169, 16]" = torch.ops.aten.full.default([169, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_583: "i64[2401]" = torch.ops.aten.reshape.default(primals_322, [-1]);  primals_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_2: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_583], view_743, True);  view_583 = view_743 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_744: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_2, [8192, 49, 49]);  fma_2 = None
        bmm_58: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_326, view_744);  permute_326 = None
        bmm_59: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_744, permute_327);  view_744 = permute_327 = None
        view_745: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_58, [512, 16, 32, 49]);  bmm_58 = None
        view_746: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_59, [512, 16, 49, 32]);  bmm_59 = None
        permute_328: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_745, [0, 1, 3, 2]);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_329: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_746, 0.1767766952966369);  view_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_2: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_329, permute_328, view_739]);  mul_329 = permute_328 = view_739 = None
        view_747: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_2, [3, 512, 16, 49, 32]);  cat_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_329: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_747, [1, 3, 0, 2, 4]);  view_747 = None
        clone_271: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format);  permute_329 = None
        view_748: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_271, [512, 49, 1536]);  clone_271 = None
        view_749: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_748, [25088, 1536]);  view_748 = None
        permute_216: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_319, [1, 0]);  primals_319 = None
        permute_330: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
        mm_29: "f32[25088, 512]" = torch.ops.aten.mm.default(view_749, permute_330);  permute_330 = None
        permute_331: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_749, [1, 0])
        mm_30: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_331, view_577);  permute_331 = view_577 = None
        sum_71: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_749, [0], True);  view_749 = None
        view_750: "f32[1536]" = torch.ops.aten.reshape.default(sum_71, [1536]);  sum_71 = None
        view_751: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_29, [512, 49, 512]);  mm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_752: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_751, [512, 7, 7, 512]);  view_751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_753: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_752, [128, 2, 2, 7, 7, 512]);  view_752 = None
        permute_334: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_753, [0, 1, 3, 2, 4, 5]);  view_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_272: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_334, memory_format = torch.contiguous_format);  permute_334 = None
        view_754: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_272, [128, 14, 14, 512]);  clone_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_70: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_754, [None, None, fmod_10]);  view_754 = None
        index_71: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_70, [None, fmod_10]);  index_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_331: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_71, primals_316);  primals_316 = None
        mul_332: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_331, 512)
        sum_72: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_331, [3], True)
        mul_333: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_331, mul_214);  mul_331 = None
        sum_73: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_333, [3], True);  mul_333 = None
        mul_334: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_214, sum_73);  sum_73 = None
        sub_99: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_332, sum_72);  mul_332 = sum_72 = None
        sub_100: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_99, mul_334);  sub_99 = mul_334 = None
        mul_335: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_78, sub_100);  div_78 = sub_100 = None
        mul_336: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_71, mul_214);  mul_214 = None
        sum_74: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_336, [0, 1, 2]);  mul_336 = None
        sum_75: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_71, [0, 1, 2]);  index_71 = None
        add_272: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_730, mul_335);  view_730 = mul_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_755: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_272, [128, 196, 512]);  add_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_39: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_39, torch.float32);  lt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_60: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_39, 0.9130434766411781);  convert_element_type_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_337: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_755, div_60);  div_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_756: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_337, [25088, 512]);  mul_337 = None
        permute_214: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_314, [1, 0]);  primals_314 = None
        permute_335: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
        mm_31: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_756, permute_335);  permute_335 = None
        permute_336: "f32[512, 25088]" = torch.ops.aten.permute.default(view_756, [1, 0])
        mm_32: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_336, view_571);  permute_336 = view_571 = None
        sum_76: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_756, [0], True);  view_756 = None
        view_757: "f32[512]" = torch.ops.aten.reshape.default(sum_76, [512]);  sum_76 = None
        view_758: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_31, [128, 196, 2048]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_570: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_82, [128, 196, 2048]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_211: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_570, 0.7071067811865476)
        erf_20: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_211);  mul_211 = None
        add_222: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_339: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_222, 0.5);  add_222 = None
        mul_340: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_570, view_570)
        mul_341: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_340, -0.5);  mul_340 = None
        exp_27: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_341);  mul_341 = None
        mul_342: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
        mul_343: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_570, mul_342);  view_570 = mul_342 = None
        add_274: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_339, mul_343);  mul_339 = mul_343 = None
        mul_344: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_758, add_274);  view_758 = add_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_759: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_344, [25088, 2048]);  mul_344 = None
        permute_213: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_312, [1, 0]);  primals_312 = None
        permute_339: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
        mm_33: "f32[25088, 512]" = torch.ops.aten.mm.default(view_759, permute_339);  permute_339 = None
        permute_340: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_759, [1, 0])
        mm_34: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_340, view_569);  permute_340 = view_569 = None
        sum_77: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_759, [0], True);  view_759 = None
        view_760: "f32[2048]" = torch.ops.aten.reshape.default(sum_77, [2048]);  sum_77 = None
        view_761: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_33, [128, 196, 512]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_346: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_761, primals_310);  primals_310 = None
        mul_347: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_346, 512)
        sum_78: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [2], True)
        mul_348: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_346, mul_208);  mul_346 = None
        sum_79: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [2], True);  mul_348 = None
        mul_349: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_208, sum_79);  sum_79 = None
        sub_102: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_347, sum_78);  mul_347 = sum_78 = None
        sub_103: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_102, mul_349);  sub_102 = mul_349 = None
        mul_350: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_79, sub_103);  div_79 = sub_103 = None
        mul_351: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_761, mul_208);  mul_208 = None
        sum_80: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_351, [0, 1]);  mul_351 = None
        sum_81: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_761, [0, 1]);  view_761 = None
        add_275: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_755, mul_350);  view_755 = mul_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_762: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_275, [128, 14, 14, 512]);  add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_38: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_38, torch.float32);  lt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_59: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_38, 0.9130434766411781);  convert_element_type_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_352: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_762, div_59);  div_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_763: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_352, [128, 2, 7, 2, 7, 512]);  mul_352 = None
        permute_343: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_763, [0, 1, 3, 2, 4, 5]);  view_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_273: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_343, memory_format = torch.contiguous_format);  permute_343 = None
        view_764: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_273, [512, 7, 7, 512]);  clone_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_765: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_764, [512, 49, 512]);  view_764 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_766: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_765, [25088, 512]);  view_765 = None
        permute_211: "f32[512, 512]" = torch.ops.aten.permute.default(primals_308, [1, 0]);  primals_308 = None
        permute_344: "f32[512, 512]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        mm_35: "f32[25088, 512]" = torch.ops.aten.mm.default(view_766, permute_344);  permute_344 = None
        permute_345: "f32[512, 25088]" = torch.ops.aten.permute.default(view_766, [1, 0])
        mm_36: "f32[512, 512]" = torch.ops.aten.mm.default(permute_345, view_563);  permute_345 = view_563 = None
        sum_82: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_766, [0], True);  view_766 = None
        view_767: "f32[512]" = torch.ops.aten.reshape.default(sum_82, [512]);  sum_82 = None
        view_768: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_35, [512, 49, 512]);  mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_769: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_768, [512, 49, 16, 32]);  view_768 = None
        permute_348: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_769, [0, 2, 1, 3]);  view_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_274: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
        view_770: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_274, [8192, 49, 32]);  clone_274 = None
        expand_82: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_58, [512, 16, 49, 49])
        view_559: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_82, [8192, 49, 49]);  expand_82 = None
        permute_349: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_559, [0, 2, 1]);  view_559 = None
        bmm_60: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_349, view_770);  permute_349 = None
        bmm_61: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_770, permute_350);  view_770 = permute_350 = None
        view_771: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_60, [512, 16, 49, 32]);  bmm_60 = None
        view_772: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_61, [512, 16, 49, 49]);  bmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_353: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_772, div_58);  view_772 = None
        sum_83: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [-1], True)
        neg_3: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_58);  div_58 = None
        fma_3: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_3, sum_83, mul_353);  neg_3 = sum_83 = mul_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_84: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_3, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_3: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_84, 0);  sum_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_351: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_3, [1, 2, 0]);  squeeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_773: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_351, [2401, 16]);  permute_351 = None
        view_557: "i64[2401]" = torch.ops.aten.reshape.default(primals_307, [-1]);  primals_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_3: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_557], view_773, True);  view_557 = view_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_774: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_3, [8192, 49, 49]);  fma_3 = None
        bmm_62: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_352, view_774);  permute_352 = None
        bmm_63: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_774, permute_353);  view_774 = permute_353 = None
        view_775: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_62, [512, 16, 32, 49]);  bmm_62 = None
        view_776: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_63, [512, 16, 49, 32]);  bmm_63 = None
        permute_354: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_775, [0, 1, 3, 2]);  view_775 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_354: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_776, 0.1767766952966369);  view_776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_3: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_354, permute_354, view_771]);  mul_354 = permute_354 = view_771 = None
        view_777: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_3, [3, 512, 16, 49, 32]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_355: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_777, [1, 3, 0, 2, 4]);  view_777 = None
        clone_275: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
        view_778: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_275, [512, 49, 1536]);  clone_275 = None
        view_779: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_778, [25088, 1536]);  view_778 = None
        permute_206: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_304, [1, 0]);  primals_304 = None
        permute_356: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
        mm_37: "f32[25088, 512]" = torch.ops.aten.mm.default(view_779, permute_356);  permute_356 = None
        permute_357: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_779, [1, 0])
        mm_38: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_357, view_551);  permute_357 = view_551 = None
        sum_85: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_779, [0], True);  view_779 = None
        view_780: "f32[1536]" = torch.ops.aten.reshape.default(sum_85, [1536]);  sum_85 = None
        view_781: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_37, [512, 49, 512]);  mm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_782: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_781, [512, 7, 7, 512]);  view_781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_783: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_782, [128, 2, 2, 7, 7, 512]);  view_782 = None
        permute_360: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_783, [0, 1, 3, 2, 4, 5]);  view_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_276: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_360, memory_format = torch.contiguous_format);  permute_360 = None
        view_784: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_276, [128, 14, 14, 512]);  clone_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_356: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_784, primals_302);  primals_302 = None
        mul_357: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_356, 512)
        sum_86: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [3], True)
        mul_358: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_356, mul_204);  mul_356 = None
        sum_87: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [3], True);  mul_358 = None
        mul_359: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_204, sum_87);  sum_87 = None
        sub_105: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_357, sum_86);  mul_357 = sum_86 = None
        sub_106: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_105, mul_359);  sub_105 = mul_359 = None
        mul_360: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_80, sub_106);  div_80 = sub_106 = None
        mul_361: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_784, mul_204);  mul_204 = None
        sum_88: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 1, 2]);  mul_361 = None
        sum_89: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_784, [0, 1, 2]);  view_784 = None
        add_276: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_762, mul_360);  view_762 = mul_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_785: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_276, [128, 196, 512]);  add_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_37: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_37, torch.float32);  lt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_57: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_37, 0.917391300201416);  convert_element_type_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_362: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_785, div_57);  div_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_786: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_362, [25088, 512]);  mul_362 = None
        permute_204: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_300, [1, 0]);  primals_300 = None
        permute_361: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_204, [1, 0]);  permute_204 = None
        mm_39: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_786, permute_361);  permute_361 = None
        permute_362: "f32[512, 25088]" = torch.ops.aten.permute.default(view_786, [1, 0])
        mm_40: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_362, view_545);  permute_362 = view_545 = None
        sum_90: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_786, [0], True);  view_786 = None
        view_787: "f32[512]" = torch.ops.aten.reshape.default(sum_90, [512]);  sum_90 = None
        view_788: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_39, [128, 196, 2048]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_544: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_78, [128, 196, 2048]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_201: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_544, 0.7071067811865476)
        erf_19: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
        add_214: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_364: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_214, 0.5);  add_214 = None
        mul_365: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_544, view_544)
        mul_366: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_365, -0.5);  mul_365 = None
        exp_28: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_366);  mul_366 = None
        mul_367: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
        mul_368: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_544, mul_367);  view_544 = mul_367 = None
        add_278: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_364, mul_368);  mul_364 = mul_368 = None
        mul_369: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_788, add_278);  view_788 = add_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_789: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_369, [25088, 2048]);  mul_369 = None
        permute_203: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_298, [1, 0]);  primals_298 = None
        permute_365: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
        mm_41: "f32[25088, 512]" = torch.ops.aten.mm.default(view_789, permute_365);  permute_365 = None
        permute_366: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_789, [1, 0])
        mm_42: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_366, view_543);  permute_366 = view_543 = None
        sum_91: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_789, [0], True);  view_789 = None
        view_790: "f32[2048]" = torch.ops.aten.reshape.default(sum_91, [2048]);  sum_91 = None
        view_791: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_41, [128, 196, 512]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_371: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_791, primals_296);  primals_296 = None
        mul_372: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_371, 512)
        sum_92: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True)
        mul_373: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_371, mul_198);  mul_371 = None
        sum_93: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [2], True);  mul_373 = None
        mul_374: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_198, sum_93);  sum_93 = None
        sub_108: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_372, sum_92);  mul_372 = sum_92 = None
        sub_109: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_108, mul_374);  sub_108 = mul_374 = None
        mul_375: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_81, sub_109);  div_81 = sub_109 = None
        mul_376: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_791, mul_198);  mul_198 = None
        sum_94: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 1]);  mul_376 = None
        sum_95: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_791, [0, 1]);  view_791 = None
        add_279: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_785, mul_375);  view_785 = mul_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_792: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_279, [128, 14, 14, 512]);  add_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_36: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_36, torch.float32);  lt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_56: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_36, 0.917391300201416);  convert_element_type_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_377: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_792, div_56);  div_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_72: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_377, [None, None, fmod_8]);  mul_377 = None
        index_73: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_72, [None, fmod_8]);  index_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_793: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_73, [128, 2, 7, 2, 7, 512]);  index_73 = None
        permute_369: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_793, [0, 1, 3, 2, 4, 5]);  view_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_277: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_369, memory_format = torch.contiguous_format);  permute_369 = None
        view_794: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_277, [512, 7, 7, 512]);  clone_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_795: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_794, [512, 49, 512]);  view_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_796: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_795, [25088, 512]);  view_795 = None
        permute_201: "f32[512, 512]" = torch.ops.aten.permute.default(primals_294, [1, 0]);  primals_294 = None
        permute_370: "f32[512, 512]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
        mm_43: "f32[25088, 512]" = torch.ops.aten.mm.default(view_796, permute_370);  permute_370 = None
        permute_371: "f32[512, 25088]" = torch.ops.aten.permute.default(view_796, [1, 0])
        mm_44: "f32[512, 512]" = torch.ops.aten.mm.default(permute_371, view_537);  permute_371 = view_537 = None
        sum_96: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True);  view_796 = None
        view_797: "f32[512]" = torch.ops.aten.reshape.default(sum_96, [512]);  sum_96 = None
        view_798: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_43, [512, 49, 512]);  mm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_799: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_798, [512, 49, 16, 32]);  view_798 = None
        permute_374: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_799, [0, 2, 1, 3]);  view_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_278: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_374, memory_format = torch.contiguous_format);  permute_374 = None
        view_800: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_278, [8192, 49, 32]);  clone_278 = None
        expand_78: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_55, [512, 16, 49, 49])
        view_533: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_78, [8192, 49, 49]);  expand_78 = None
        permute_375: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_533, [0, 2, 1]);  view_533 = None
        bmm_64: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_375, view_800);  permute_375 = None
        bmm_65: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_800, permute_376);  view_800 = permute_376 = None
        view_801: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_64, [512, 16, 49, 32]);  bmm_64 = None
        view_802: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_65, [512, 16, 49, 49]);  bmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_378: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_802, div_55);  view_802 = None
        sum_97: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_378, [-1], True)
        neg_4: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_55);  div_55 = None
        fma_4: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_4, sum_97, mul_378);  neg_4 = sum_97 = mul_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_98: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_4, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_4: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_98, 0);  sum_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_377: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_4, [1, 2, 0]);  squeeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_805: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_377, [2401, 16]);  permute_377 = None
        view_529: "i64[2401]" = torch.ops.aten.reshape.default(primals_293, [-1]);  primals_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_4: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_529], view_805, True);  view_529 = view_805 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_806: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_4, [8192, 49, 49]);  fma_4 = None
        bmm_66: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_378, view_806);  permute_378 = None
        bmm_67: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_806, permute_379);  view_806 = permute_379 = None
        view_807: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_66, [512, 16, 32, 49]);  bmm_66 = None
        view_808: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_67, [512, 16, 49, 32]);  bmm_67 = None
        permute_380: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_807, [0, 1, 3, 2]);  view_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_379: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_808, 0.1767766952966369);  view_808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_4: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_379, permute_380, view_801]);  mul_379 = permute_380 = view_801 = None
        view_809: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_4, [3, 512, 16, 49, 32]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_381: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_809, [1, 3, 0, 2, 4]);  view_809 = None
        clone_279: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_381, memory_format = torch.contiguous_format);  permute_381 = None
        view_810: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_279, [512, 49, 1536]);  clone_279 = None
        view_811: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_810, [25088, 1536]);  view_810 = None
        permute_196: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_290, [1, 0]);  primals_290 = None
        permute_382: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
        mm_45: "f32[25088, 512]" = torch.ops.aten.mm.default(view_811, permute_382);  permute_382 = None
        permute_383: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_811, [1, 0])
        mm_46: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_383, view_523);  permute_383 = view_523 = None
        sum_99: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_811, [0], True);  view_811 = None
        view_812: "f32[1536]" = torch.ops.aten.reshape.default(sum_99, [1536]);  sum_99 = None
        view_813: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_45, [512, 49, 512]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_814: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_813, [512, 7, 7, 512]);  view_813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_815: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_814, [128, 2, 2, 7, 7, 512]);  view_814 = None
        permute_386: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_815, [0, 1, 3, 2, 4, 5]);  view_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_280: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
        view_816: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_280, [128, 14, 14, 512]);  clone_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_74: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_816, [None, None, fmod_10]);  view_816 = None
        index_75: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_74, [None, fmod_10]);  index_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_381: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_75, primals_287);  primals_287 = None
        mul_382: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_381, 512)
        sum_100: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [3], True)
        mul_383: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_381, mul_194);  mul_381 = None
        sum_101: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [3], True);  mul_383 = None
        mul_384: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_194, sum_101);  sum_101 = None
        sub_111: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_382, sum_100);  mul_382 = sum_100 = None
        sub_112: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_111, mul_384);  sub_111 = mul_384 = None
        mul_385: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_82, sub_112);  div_82 = sub_112 = None
        mul_386: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_75, mul_194);  mul_194 = None
        sum_102: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 1, 2]);  mul_386 = None
        sum_103: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_75, [0, 1, 2]);  index_75 = None
        add_284: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_792, mul_385);  view_792 = mul_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_817: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_284, [128, 196, 512]);  add_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_35: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_35, torch.float32);  lt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_54: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_35, 0.9217391312122345);  convert_element_type_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_387: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_817, div_54);  div_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_818: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_387, [25088, 512]);  mul_387 = None
        permute_194: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
        permute_387: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
        mm_47: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_818, permute_387);  permute_387 = None
        permute_388: "f32[512, 25088]" = torch.ops.aten.permute.default(view_818, [1, 0])
        mm_48: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_388, view_517);  permute_388 = view_517 = None
        sum_104: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_818, [0], True);  view_818 = None
        view_819: "f32[512]" = torch.ops.aten.reshape.default(sum_104, [512]);  sum_104 = None
        view_820: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_47, [128, 196, 2048]);  mm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_516: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_74, [128, 196, 2048]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_191: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_516, 0.7071067811865476)
        erf_18: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_191);  mul_191 = None
        add_201: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_389: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_201, 0.5);  add_201 = None
        mul_390: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_516, view_516)
        mul_391: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_390, -0.5);  mul_390 = None
        exp_29: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_391);  mul_391 = None
        mul_392: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
        mul_393: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_516, mul_392);  view_516 = mul_392 = None
        add_286: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_389, mul_393);  mul_389 = mul_393 = None
        mul_394: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_820, add_286);  view_820 = add_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_821: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_394, [25088, 2048]);  mul_394 = None
        permute_193: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_283, [1, 0]);  primals_283 = None
        permute_391: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
        mm_49: "f32[25088, 512]" = torch.ops.aten.mm.default(view_821, permute_391);  permute_391 = None
        permute_392: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_821, [1, 0])
        mm_50: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_392, view_515);  permute_392 = view_515 = None
        sum_105: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_821, [0], True);  view_821 = None
        view_822: "f32[2048]" = torch.ops.aten.reshape.default(sum_105, [2048]);  sum_105 = None
        view_823: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_49, [128, 196, 512]);  mm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_396: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_823, primals_281);  primals_281 = None
        mul_397: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_396, 512)
        sum_106: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_396, [2], True)
        mul_398: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_396, mul_188);  mul_396 = None
        sum_107: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True);  mul_398 = None
        mul_399: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_188, sum_107);  sum_107 = None
        sub_114: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_397, sum_106);  mul_397 = sum_106 = None
        sub_115: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_114, mul_399);  sub_114 = mul_399 = None
        mul_400: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_83, sub_115);  div_83 = sub_115 = None
        mul_401: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_823, mul_188);  mul_188 = None
        sum_108: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_401, [0, 1]);  mul_401 = None
        sum_109: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_823, [0, 1]);  view_823 = None
        add_287: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_817, mul_400);  view_817 = mul_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_824: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_287, [128, 14, 14, 512]);  add_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_34: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_34, torch.float32);  lt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_53: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_34, 0.9217391312122345);  convert_element_type_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_402: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_824, div_53);  div_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_825: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_402, [128, 2, 7, 2, 7, 512]);  mul_402 = None
        permute_395: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_825, [0, 1, 3, 2, 4, 5]);  view_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_281: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_395, memory_format = torch.contiguous_format);  permute_395 = None
        view_826: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_281, [512, 7, 7, 512]);  clone_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_827: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_826, [512, 49, 512]);  view_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_828: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_827, [25088, 512]);  view_827 = None
        permute_191: "f32[512, 512]" = torch.ops.aten.permute.default(primals_279, [1, 0]);  primals_279 = None
        permute_396: "f32[512, 512]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
        mm_51: "f32[25088, 512]" = torch.ops.aten.mm.default(view_828, permute_396);  permute_396 = None
        permute_397: "f32[512, 25088]" = torch.ops.aten.permute.default(view_828, [1, 0])
        mm_52: "f32[512, 512]" = torch.ops.aten.mm.default(permute_397, view_509);  permute_397 = view_509 = None
        sum_110: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_828, [0], True);  view_828 = None
        view_829: "f32[512]" = torch.ops.aten.reshape.default(sum_110, [512]);  sum_110 = None
        view_830: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_51, [512, 49, 512]);  mm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_831: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_830, [512, 49, 16, 32]);  view_830 = None
        permute_400: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_831, [0, 2, 1, 3]);  view_831 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_282: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_400, memory_format = torch.contiguous_format);  permute_400 = None
        view_832: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_282, [8192, 49, 32]);  clone_282 = None
        expand_74: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_52, [512, 16, 49, 49])
        view_505: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_74, [8192, 49, 49]);  expand_74 = None
        permute_401: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_505, [0, 2, 1]);  view_505 = None
        bmm_68: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_401, view_832);  permute_401 = None
        bmm_69: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_832, permute_402);  view_832 = permute_402 = None
        view_833: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_68, [512, 16, 49, 32]);  bmm_68 = None
        view_834: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_69, [512, 16, 49, 49]);  bmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_403: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_834, div_52);  view_834 = None
        sum_111: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_403, [-1], True)
        neg_5: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_52);  div_52 = None
        fma_5: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_5, sum_111, mul_403);  neg_5 = sum_111 = mul_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_112: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_5, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_5: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_112, 0);  sum_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_403: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_5, [1, 2, 0]);  squeeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_835: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_403, [2401, 16]);  permute_403 = None
        view_503: "i64[2401]" = torch.ops.aten.reshape.default(primals_278, [-1]);  primals_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_5: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_503], view_835, True);  view_503 = view_835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_836: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_5, [8192, 49, 49]);  fma_5 = None
        bmm_70: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_404, view_836);  permute_404 = None
        bmm_71: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_836, permute_405);  view_836 = permute_405 = None
        view_837: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_70, [512, 16, 32, 49]);  bmm_70 = None
        view_838: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_71, [512, 16, 49, 32]);  bmm_71 = None
        permute_406: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_837, [0, 1, 3, 2]);  view_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_404: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_838, 0.1767766952966369);  view_838 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_5: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_404, permute_406, view_833]);  mul_404 = permute_406 = view_833 = None
        view_839: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_5, [3, 512, 16, 49, 32]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_407: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_839, [1, 3, 0, 2, 4]);  view_839 = None
        clone_283: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_407, memory_format = torch.contiguous_format);  permute_407 = None
        view_840: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_283, [512, 49, 1536]);  clone_283 = None
        view_841: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_840, [25088, 1536]);  view_840 = None
        permute_186: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_275, [1, 0]);  primals_275 = None
        permute_408: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        mm_53: "f32[25088, 512]" = torch.ops.aten.mm.default(view_841, permute_408);  permute_408 = None
        permute_409: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_841, [1, 0])
        mm_54: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_409, view_497);  permute_409 = view_497 = None
        sum_113: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_841, [0], True);  view_841 = None
        view_842: "f32[1536]" = torch.ops.aten.reshape.default(sum_113, [1536]);  sum_113 = None
        view_843: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_53, [512, 49, 512]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_844: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_843, [512, 7, 7, 512]);  view_843 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_845: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_844, [128, 2, 2, 7, 7, 512]);  view_844 = None
        permute_412: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_845, [0, 1, 3, 2, 4, 5]);  view_845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_284: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_412, memory_format = torch.contiguous_format);  permute_412 = None
        view_846: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_284, [128, 14, 14, 512]);  clone_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_406: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_846, primals_273);  primals_273 = None
        mul_407: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_406, 512)
        sum_114: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_406, [3], True)
        mul_408: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_406, mul_184);  mul_406 = None
        sum_115: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_408, [3], True);  mul_408 = None
        mul_409: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_184, sum_115);  sum_115 = None
        sub_117: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_407, sum_114);  mul_407 = sum_114 = None
        sub_118: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_117, mul_409);  sub_117 = mul_409 = None
        mul_410: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_84, sub_118);  div_84 = sub_118 = None
        mul_411: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_846, mul_184);  mul_184 = None
        sum_116: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_411, [0, 1, 2]);  mul_411 = None
        sum_117: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_846, [0, 1, 2]);  view_846 = None
        add_288: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_824, mul_410);  view_824 = mul_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_847: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_288, [128, 196, 512]);  add_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_33: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_33, torch.float32);  lt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_51: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_33, 0.9260869547724724);  convert_element_type_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_412: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_847, div_51);  div_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_848: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_412, [25088, 512]);  mul_412 = None
        permute_184: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
        permute_413: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
        mm_55: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_848, permute_413);  permute_413 = None
        permute_414: "f32[512, 25088]" = torch.ops.aten.permute.default(view_848, [1, 0])
        mm_56: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_414, view_491);  permute_414 = view_491 = None
        sum_118: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_848, [0], True);  view_848 = None
        view_849: "f32[512]" = torch.ops.aten.reshape.default(sum_118, [512]);  sum_118 = None
        view_850: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_55, [128, 196, 2048]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_490: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_70, [128, 196, 2048]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_181: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_490, 0.7071067811865476)
        erf_17: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_181);  mul_181 = None
        add_193: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_414: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_193, 0.5);  add_193 = None
        mul_415: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_490, view_490)
        mul_416: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_415, -0.5);  mul_415 = None
        exp_30: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_416);  mul_416 = None
        mul_417: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
        mul_418: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_490, mul_417);  view_490 = mul_417 = None
        add_290: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_414, mul_418);  mul_414 = mul_418 = None
        mul_419: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_850, add_290);  view_850 = add_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_851: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_419, [25088, 2048]);  mul_419 = None
        permute_183: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_269, [1, 0]);  primals_269 = None
        permute_417: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
        mm_57: "f32[25088, 512]" = torch.ops.aten.mm.default(view_851, permute_417);  permute_417 = None
        permute_418: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_851, [1, 0])
        mm_58: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_418, view_489);  permute_418 = view_489 = None
        sum_119: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_851, [0], True);  view_851 = None
        view_852: "f32[2048]" = torch.ops.aten.reshape.default(sum_119, [2048]);  sum_119 = None
        view_853: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_57, [128, 196, 512]);  mm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_421: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_853, primals_267);  primals_267 = None
        mul_422: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_421, 512)
        sum_120: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_421, [2], True)
        mul_423: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_421, mul_178);  mul_421 = None
        sum_121: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_423, [2], True);  mul_423 = None
        mul_424: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_178, sum_121);  sum_121 = None
        sub_120: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_422, sum_120);  mul_422 = sum_120 = None
        sub_121: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_120, mul_424);  sub_120 = mul_424 = None
        mul_425: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_85, sub_121);  div_85 = sub_121 = None
        mul_426: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_853, mul_178);  mul_178 = None
        sum_122: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_426, [0, 1]);  mul_426 = None
        sum_123: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_853, [0, 1]);  view_853 = None
        add_291: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_847, mul_425);  view_847 = mul_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_854: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_291, [128, 14, 14, 512]);  add_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_32: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_32, torch.float32);  lt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_50: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_32, 0.9260869547724724);  convert_element_type_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_427: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_854, div_50);  div_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_76: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_427, [None, None, fmod_8]);  mul_427 = None
        index_77: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_76, [None, fmod_8]);  index_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_855: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_77, [128, 2, 7, 2, 7, 512]);  index_77 = None
        permute_421: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_855, [0, 1, 3, 2, 4, 5]);  view_855 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_285: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_421, memory_format = torch.contiguous_format);  permute_421 = None
        view_856: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_285, [512, 7, 7, 512]);  clone_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_857: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_856, [512, 49, 512]);  view_856 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_858: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_857, [25088, 512]);  view_857 = None
        permute_181: "f32[512, 512]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
        permute_422: "f32[512, 512]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
        mm_59: "f32[25088, 512]" = torch.ops.aten.mm.default(view_858, permute_422);  permute_422 = None
        permute_423: "f32[512, 25088]" = torch.ops.aten.permute.default(view_858, [1, 0])
        mm_60: "f32[512, 512]" = torch.ops.aten.mm.default(permute_423, view_483);  permute_423 = view_483 = None
        sum_124: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_858, [0], True);  view_858 = None
        view_859: "f32[512]" = torch.ops.aten.reshape.default(sum_124, [512]);  sum_124 = None
        view_860: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_59, [512, 49, 512]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_861: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_860, [512, 49, 16, 32]);  view_860 = None
        permute_426: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_861, [0, 2, 1, 3]);  view_861 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_286: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
        view_862: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_286, [8192, 49, 32]);  clone_286 = None
        expand_70: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_49, [512, 16, 49, 49])
        view_479: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_70, [8192, 49, 49]);  expand_70 = None
        permute_427: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_479, [0, 2, 1]);  view_479 = None
        bmm_72: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_427, view_862);  permute_427 = None
        bmm_73: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_862, permute_428);  view_862 = permute_428 = None
        view_863: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_72, [512, 16, 49, 32]);  bmm_72 = None
        view_864: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_73, [512, 16, 49, 49]);  bmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_428: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_864, div_49);  view_864 = None
        sum_125: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_428, [-1], True)
        neg_6: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_49);  div_49 = None
        fma_6: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_6, sum_125, mul_428);  neg_6 = sum_125 = mul_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_126: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_6, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_6: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_126, 0);  sum_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_429: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_6, [1, 2, 0]);  squeeze_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_867: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_429, [2401, 16]);  permute_429 = None
        view_475: "i64[2401]" = torch.ops.aten.reshape.default(primals_264, [-1]);  primals_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_6: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_475], view_867, True);  view_475 = view_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_868: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_6, [8192, 49, 49]);  fma_6 = None
        bmm_74: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_430, view_868);  permute_430 = None
        bmm_75: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_868, permute_431);  view_868 = permute_431 = None
        view_869: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_74, [512, 16, 32, 49]);  bmm_74 = None
        view_870: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_75, [512, 16, 49, 32]);  bmm_75 = None
        permute_432: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_869, [0, 1, 3, 2]);  view_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_429: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_870, 0.1767766952966369);  view_870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_6: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_429, permute_432, view_863]);  mul_429 = permute_432 = view_863 = None
        view_871: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_6, [3, 512, 16, 49, 32]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_433: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_871, [1, 3, 0, 2, 4]);  view_871 = None
        clone_287: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
        view_872: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_287, [512, 49, 1536]);  clone_287 = None
        view_873: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_872, [25088, 1536]);  view_872 = None
        permute_176: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
        permute_434: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        mm_61: "f32[25088, 512]" = torch.ops.aten.mm.default(view_873, permute_434);  permute_434 = None
        permute_435: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_873, [1, 0])
        mm_62: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_435, view_469);  permute_435 = view_469 = None
        sum_127: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_873, [0], True);  view_873 = None
        view_874: "f32[1536]" = torch.ops.aten.reshape.default(sum_127, [1536]);  sum_127 = None
        view_875: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_61, [512, 49, 512]);  mm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_876: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_875, [512, 7, 7, 512]);  view_875 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_877: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_876, [128, 2, 2, 7, 7, 512]);  view_876 = None
        permute_438: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_877, [0, 1, 3, 2, 4, 5]);  view_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_288: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_438, memory_format = torch.contiguous_format);  permute_438 = None
        view_878: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_288, [128, 14, 14, 512]);  clone_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_78: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_878, [None, None, fmod_10]);  view_878 = None
        index_79: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_78, [None, fmod_10]);  index_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_431: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_79, primals_258);  primals_258 = None
        mul_432: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_431, 512)
        sum_128: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_431, [3], True)
        mul_433: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_431, mul_174);  mul_431 = None
        sum_129: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_433, [3], True);  mul_433 = None
        mul_434: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_174, sum_129);  sum_129 = None
        sub_123: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_432, sum_128);  mul_432 = sum_128 = None
        sub_124: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_123, mul_434);  sub_123 = mul_434 = None
        mul_435: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_86, sub_124);  div_86 = sub_124 = None
        mul_436: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_79, mul_174);  mul_174 = None
        sum_130: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 1, 2]);  mul_436 = None
        sum_131: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_79, [0, 1, 2]);  index_79 = None
        add_296: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_854, mul_435);  view_854 = mul_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_879: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_296, [128, 196, 512]);  add_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_31: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_31, torch.float32);  lt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_48: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_31, 0.9304347857832909);  convert_element_type_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_437: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_879, div_48);  div_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_880: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_437, [25088, 512]);  mul_437 = None
        permute_174: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_256, [1, 0]);  primals_256 = None
        permute_439: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        mm_63: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_880, permute_439);  permute_439 = None
        permute_440: "f32[512, 25088]" = torch.ops.aten.permute.default(view_880, [1, 0])
        mm_64: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_440, view_463);  permute_440 = view_463 = None
        sum_132: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_880, [0], True);  view_880 = None
        view_881: "f32[512]" = torch.ops.aten.reshape.default(sum_132, [512]);  sum_132 = None
        view_882: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_63, [128, 196, 2048]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_462: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_66, [128, 196, 2048]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_171: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_462, 0.7071067811865476)
        erf_16: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
        add_180: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_439: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_180, 0.5);  add_180 = None
        mul_440: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_462, view_462)
        mul_441: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_440, -0.5);  mul_440 = None
        exp_31: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_441);  mul_441 = None
        mul_442: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
        mul_443: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_462, mul_442);  view_462 = mul_442 = None
        add_298: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_439, mul_443);  mul_439 = mul_443 = None
        mul_444: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_882, add_298);  view_882 = add_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_883: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_444, [25088, 2048]);  mul_444 = None
        permute_173: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_254, [1, 0]);  primals_254 = None
        permute_443: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        mm_65: "f32[25088, 512]" = torch.ops.aten.mm.default(view_883, permute_443);  permute_443 = None
        permute_444: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_883, [1, 0])
        mm_66: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_444, view_461);  permute_444 = view_461 = None
        sum_133: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_883, [0], True);  view_883 = None
        view_884: "f32[2048]" = torch.ops.aten.reshape.default(sum_133, [2048]);  sum_133 = None
        view_885: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_65, [128, 196, 512]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_446: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_885, primals_252);  primals_252 = None
        mul_447: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_446, 512)
        sum_134: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_446, [2], True)
        mul_448: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_446, mul_168);  mul_446 = None
        sum_135: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_448, [2], True);  mul_448 = None
        mul_449: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_168, sum_135);  sum_135 = None
        sub_126: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_447, sum_134);  mul_447 = sum_134 = None
        sub_127: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_126, mul_449);  sub_126 = mul_449 = None
        mul_450: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_87, sub_127);  div_87 = sub_127 = None
        mul_451: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_885, mul_168);  mul_168 = None
        sum_136: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 1]);  mul_451 = None
        sum_137: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_885, [0, 1]);  view_885 = None
        add_299: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_879, mul_450);  view_879 = mul_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_886: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_299, [128, 14, 14, 512]);  add_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_30: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_30, torch.float32);  lt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_47: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_30, 0.9304347857832909);  convert_element_type_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_452: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_886, div_47);  div_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_887: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_452, [128, 2, 7, 2, 7, 512]);  mul_452 = None
        permute_447: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_887, [0, 1, 3, 2, 4, 5]);  view_887 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_289: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_447, memory_format = torch.contiguous_format);  permute_447 = None
        view_888: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_289, [512, 7, 7, 512]);  clone_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_889: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_888, [512, 49, 512]);  view_888 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_890: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_889, [25088, 512]);  view_889 = None
        permute_171: "f32[512, 512]" = torch.ops.aten.permute.default(primals_250, [1, 0]);  primals_250 = None
        permute_448: "f32[512, 512]" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None
        mm_67: "f32[25088, 512]" = torch.ops.aten.mm.default(view_890, permute_448);  permute_448 = None
        permute_449: "f32[512, 25088]" = torch.ops.aten.permute.default(view_890, [1, 0])
        mm_68: "f32[512, 512]" = torch.ops.aten.mm.default(permute_449, view_455);  permute_449 = view_455 = None
        sum_138: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_890, [0], True);  view_890 = None
        view_891: "f32[512]" = torch.ops.aten.reshape.default(sum_138, [512]);  sum_138 = None
        view_892: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_67, [512, 49, 512]);  mm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_893: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_892, [512, 49, 16, 32]);  view_892 = None
        permute_452: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_893, [0, 2, 1, 3]);  view_893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_290: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
        view_894: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_290, [8192, 49, 32]);  clone_290 = None
        expand_66: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_46, [512, 16, 49, 49])
        view_451: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_66, [8192, 49, 49]);  expand_66 = None
        permute_453: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_451, [0, 2, 1]);  view_451 = None
        bmm_76: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_453, view_894);  permute_453 = None
        bmm_77: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_894, permute_454);  view_894 = permute_454 = None
        view_895: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_76, [512, 16, 49, 32]);  bmm_76 = None
        view_896: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_77, [512, 16, 49, 49]);  bmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_453: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_896, div_46);  view_896 = None
        sum_139: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [-1], True)
        neg_7: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_46);  div_46 = None
        fma_7: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_7, sum_139, mul_453);  neg_7 = sum_139 = mul_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_140: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_7, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_7: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_140, 0);  sum_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_455: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_7, [1, 2, 0]);  squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_897: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_455, [2401, 16]);  permute_455 = None
        view_449: "i64[2401]" = torch.ops.aten.reshape.default(primals_249, [-1]);  primals_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_7: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_449], view_897, True);  view_449 = view_897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_898: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_7, [8192, 49, 49]);  fma_7 = None
        bmm_78: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_456, view_898);  permute_456 = None
        bmm_79: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_898, permute_457);  view_898 = permute_457 = None
        view_899: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_78, [512, 16, 32, 49]);  bmm_78 = None
        view_900: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_79, [512, 16, 49, 32]);  bmm_79 = None
        permute_458: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_899, [0, 1, 3, 2]);  view_899 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_454: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_900, 0.1767766952966369);  view_900 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_7: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_454, permute_458, view_895]);  mul_454 = permute_458 = view_895 = None
        view_901: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_7, [3, 512, 16, 49, 32]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_459: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_901, [1, 3, 0, 2, 4]);  view_901 = None
        clone_291: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
        view_902: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_291, [512, 49, 1536]);  clone_291 = None
        view_903: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_902, [25088, 1536]);  view_902 = None
        permute_166: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_246, [1, 0]);  primals_246 = None
        permute_460: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
        mm_69: "f32[25088, 512]" = torch.ops.aten.mm.default(view_903, permute_460);  permute_460 = None
        permute_461: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_903, [1, 0])
        mm_70: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_461, view_443);  permute_461 = view_443 = None
        sum_141: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_903, [0], True);  view_903 = None
        view_904: "f32[1536]" = torch.ops.aten.reshape.default(sum_141, [1536]);  sum_141 = None
        view_905: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_69, [512, 49, 512]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_906: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_905, [512, 7, 7, 512]);  view_905 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_907: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_906, [128, 2, 2, 7, 7, 512]);  view_906 = None
        permute_464: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_907, [0, 1, 3, 2, 4, 5]);  view_907 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_292: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_464, memory_format = torch.contiguous_format);  permute_464 = None
        view_908: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_292, [128, 14, 14, 512]);  clone_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_456: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_908, primals_244);  primals_244 = None
        mul_457: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_456, 512)
        sum_142: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_456, [3], True)
        mul_458: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_456, mul_164);  mul_456 = None
        sum_143: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_458, [3], True);  mul_458 = None
        mul_459: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_164, sum_143);  sum_143 = None
        sub_129: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_457, sum_142);  mul_457 = sum_142 = None
        sub_130: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_129, mul_459);  sub_129 = mul_459 = None
        mul_460: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_88, sub_130);  div_88 = sub_130 = None
        mul_461: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_908, mul_164);  mul_164 = None
        sum_144: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_461, [0, 1, 2]);  mul_461 = None
        sum_145: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_908, [0, 1, 2]);  view_908 = None
        add_300: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_886, mul_460);  view_886 = mul_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_909: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_300, [128, 196, 512]);  add_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_29: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_29, torch.float32);  lt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_45: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_29, 0.9347826093435287);  convert_element_type_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_462: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_909, div_45);  div_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_910: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_462, [25088, 512]);  mul_462 = None
        permute_164: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_242, [1, 0]);  primals_242 = None
        permute_465: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
        mm_71: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_910, permute_465);  permute_465 = None
        permute_466: "f32[512, 25088]" = torch.ops.aten.permute.default(view_910, [1, 0])
        mm_72: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_466, view_437);  permute_466 = view_437 = None
        sum_146: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_910, [0], True);  view_910 = None
        view_911: "f32[512]" = torch.ops.aten.reshape.default(sum_146, [512]);  sum_146 = None
        view_912: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_71, [128, 196, 2048]);  mm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_436: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_62, [128, 196, 2048]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_161: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_436, 0.7071067811865476)
        erf_15: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_161);  mul_161 = None
        add_172: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_464: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_172, 0.5);  add_172 = None
        mul_465: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_436, view_436)
        mul_466: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_465, -0.5);  mul_465 = None
        exp_32: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_466);  mul_466 = None
        mul_467: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
        mul_468: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_436, mul_467);  view_436 = mul_467 = None
        add_302: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_464, mul_468);  mul_464 = mul_468 = None
        mul_469: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_912, add_302);  view_912 = add_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_913: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_469, [25088, 2048]);  mul_469 = None
        permute_163: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_240, [1, 0]);  primals_240 = None
        permute_469: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        mm_73: "f32[25088, 512]" = torch.ops.aten.mm.default(view_913, permute_469);  permute_469 = None
        permute_470: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_913, [1, 0])
        mm_74: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_470, view_435);  permute_470 = view_435 = None
        sum_147: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_913, [0], True);  view_913 = None
        view_914: "f32[2048]" = torch.ops.aten.reshape.default(sum_147, [2048]);  sum_147 = None
        view_915: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_73, [128, 196, 512]);  mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_471: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_915, primals_238);  primals_238 = None
        mul_472: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_471, 512)
        sum_148: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_471, [2], True)
        mul_473: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_471, mul_158);  mul_471 = None
        sum_149: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [2], True);  mul_473 = None
        mul_474: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_158, sum_149);  sum_149 = None
        sub_132: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_472, sum_148);  mul_472 = sum_148 = None
        sub_133: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_132, mul_474);  sub_132 = mul_474 = None
        mul_475: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_89, sub_133);  div_89 = sub_133 = None
        mul_476: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_915, mul_158);  mul_158 = None
        sum_150: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 1]);  mul_476 = None
        sum_151: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_915, [0, 1]);  view_915 = None
        add_303: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_909, mul_475);  view_909 = mul_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_916: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_303, [128, 14, 14, 512]);  add_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_28: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_28, torch.float32);  lt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_44: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_28, 0.9347826093435287);  convert_element_type_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_477: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_916, div_44);  div_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_80: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_477, [None, None, fmod_8]);  mul_477 = None
        index_81: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_80, [None, fmod_8]);  index_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_917: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_81, [128, 2, 7, 2, 7, 512]);  index_81 = None
        permute_473: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_917, [0, 1, 3, 2, 4, 5]);  view_917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_293: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_473, memory_format = torch.contiguous_format);  permute_473 = None
        view_918: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_293, [512, 7, 7, 512]);  clone_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_919: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_918, [512, 49, 512]);  view_918 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_920: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_919, [25088, 512]);  view_919 = None
        permute_161: "f32[512, 512]" = torch.ops.aten.permute.default(primals_236, [1, 0]);  primals_236 = None
        permute_474: "f32[512, 512]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
        mm_75: "f32[25088, 512]" = torch.ops.aten.mm.default(view_920, permute_474);  permute_474 = None
        permute_475: "f32[512, 25088]" = torch.ops.aten.permute.default(view_920, [1, 0])
        mm_76: "f32[512, 512]" = torch.ops.aten.mm.default(permute_475, view_429);  permute_475 = view_429 = None
        sum_152: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_920, [0], True);  view_920 = None
        view_921: "f32[512]" = torch.ops.aten.reshape.default(sum_152, [512]);  sum_152 = None
        view_922: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_75, [512, 49, 512]);  mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_923: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_922, [512, 49, 16, 32]);  view_922 = None
        permute_478: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_923, [0, 2, 1, 3]);  view_923 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_294: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_478, memory_format = torch.contiguous_format);  permute_478 = None
        view_924: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_294, [8192, 49, 32]);  clone_294 = None
        expand_62: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_43, [512, 16, 49, 49])
        view_425: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_62, [8192, 49, 49]);  expand_62 = None
        permute_479: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_425, [0, 2, 1]);  view_425 = None
        bmm_80: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_479, view_924);  permute_479 = None
        bmm_81: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_924, permute_480);  view_924 = permute_480 = None
        view_925: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_80, [512, 16, 49, 32]);  bmm_80 = None
        view_926: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_81, [512, 16, 49, 49]);  bmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_478: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_926, div_43);  view_926 = None
        sum_153: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_478, [-1], True)
        neg_8: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_43);  div_43 = None
        fma_8: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_8, sum_153, mul_478);  neg_8 = sum_153 = mul_478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_154: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_8, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_8: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_154, 0);  sum_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_481: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_8, [1, 2, 0]);  squeeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_929: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_481, [2401, 16]);  permute_481 = None
        view_421: "i64[2401]" = torch.ops.aten.reshape.default(primals_235, [-1]);  primals_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_8: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_421], view_929, True);  view_421 = view_929 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_930: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_8, [8192, 49, 49]);  fma_8 = None
        bmm_82: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_482, view_930);  permute_482 = None
        bmm_83: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_930, permute_483);  view_930 = permute_483 = None
        view_931: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_82, [512, 16, 32, 49]);  bmm_82 = None
        view_932: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_83, [512, 16, 49, 32]);  bmm_83 = None
        permute_484: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_931, [0, 1, 3, 2]);  view_931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_479: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_932, 0.1767766952966369);  view_932 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_8: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_479, permute_484, view_925]);  mul_479 = permute_484 = view_925 = None
        view_933: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_8, [3, 512, 16, 49, 32]);  cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_485: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_933, [1, 3, 0, 2, 4]);  view_933 = None
        clone_295: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
        view_934: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_295, [512, 49, 1536]);  clone_295 = None
        view_935: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_934, [25088, 1536]);  view_934 = None
        permute_156: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_232, [1, 0]);  primals_232 = None
        permute_486: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        mm_77: "f32[25088, 512]" = torch.ops.aten.mm.default(view_935, permute_486);  permute_486 = None
        permute_487: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_935, [1, 0])
        mm_78: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_487, view_415);  permute_487 = view_415 = None
        sum_155: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_935, [0], True);  view_935 = None
        view_936: "f32[1536]" = torch.ops.aten.reshape.default(sum_155, [1536]);  sum_155 = None
        view_937: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_77, [512, 49, 512]);  mm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_938: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_937, [512, 7, 7, 512]);  view_937 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_939: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_938, [128, 2, 2, 7, 7, 512]);  view_938 = None
        permute_490: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_939, [0, 1, 3, 2, 4, 5]);  view_939 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_296: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
        view_940: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_296, [128, 14, 14, 512]);  clone_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_82: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_940, [None, None, fmod_10]);  view_940 = None
        index_83: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_82, [None, fmod_10]);  index_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_481: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_83, primals_229);  primals_229 = None
        mul_482: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_481, 512)
        sum_156: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_481, [3], True)
        mul_483: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_481, mul_154);  mul_481 = None
        sum_157: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [3], True);  mul_483 = None
        mul_484: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_154, sum_157);  sum_157 = None
        sub_135: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_482, sum_156);  mul_482 = sum_156 = None
        sub_136: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_135, mul_484);  sub_135 = mul_484 = None
        mul_485: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_90, sub_136);  div_90 = sub_136 = None
        mul_486: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_83, mul_154);  mul_154 = None
        sum_158: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_486, [0, 1, 2]);  mul_486 = None
        sum_159: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_83, [0, 1, 2]);  index_83 = None
        add_308: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_916, mul_485);  view_916 = mul_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_941: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_308, [128, 196, 512]);  add_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_27: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_27, torch.float32);  lt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_42: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_27, 0.9391304366290569);  convert_element_type_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_487: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_941, div_42);  div_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_942: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_487, [25088, 512]);  mul_487 = None
        permute_154: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
        permute_491: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        mm_79: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_942, permute_491);  permute_491 = None
        permute_492: "f32[512, 25088]" = torch.ops.aten.permute.default(view_942, [1, 0])
        mm_80: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_492, view_409);  permute_492 = view_409 = None
        sum_160: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_942, [0], True);  view_942 = None
        view_943: "f32[512]" = torch.ops.aten.reshape.default(sum_160, [512]);  sum_160 = None
        view_944: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_79, [128, 196, 2048]);  mm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_408: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_58, [128, 196, 2048]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_151: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_408, 0.7071067811865476)
        erf_14: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_151);  mul_151 = None
        add_159: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_489: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_159, 0.5);  add_159 = None
        mul_490: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_408, view_408)
        mul_491: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_490, -0.5);  mul_490 = None
        exp_33: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_491);  mul_491 = None
        mul_492: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
        mul_493: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_408, mul_492);  view_408 = mul_492 = None
        add_310: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_489, mul_493);  mul_489 = mul_493 = None
        mul_494: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_944, add_310);  view_944 = add_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_945: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_494, [25088, 2048]);  mul_494 = None
        permute_153: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
        permute_495: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        mm_81: "f32[25088, 512]" = torch.ops.aten.mm.default(view_945, permute_495);  permute_495 = None
        permute_496: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_945, [1, 0])
        mm_82: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_496, view_407);  permute_496 = view_407 = None
        sum_161: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_945, [0], True);  view_945 = None
        view_946: "f32[2048]" = torch.ops.aten.reshape.default(sum_161, [2048]);  sum_161 = None
        view_947: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_81, [128, 196, 512]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_496: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_947, primals_223);  primals_223 = None
        mul_497: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_496, 512)
        sum_162: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_496, [2], True)
        mul_498: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_496, mul_148);  mul_496 = None
        sum_163: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_498, [2], True);  mul_498 = None
        mul_499: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_148, sum_163);  sum_163 = None
        sub_138: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_497, sum_162);  mul_497 = sum_162 = None
        sub_139: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_138, mul_499);  sub_138 = mul_499 = None
        mul_500: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_91, sub_139);  div_91 = sub_139 = None
        mul_501: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_947, mul_148);  mul_148 = None
        sum_164: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_501, [0, 1]);  mul_501 = None
        sum_165: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_947, [0, 1]);  view_947 = None
        add_311: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_941, mul_500);  view_941 = mul_500 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_948: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_311, [128, 14, 14, 512]);  add_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_26: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_26, torch.float32);  lt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_41: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_26, 0.9391304366290569);  convert_element_type_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_502: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_948, div_41);  div_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_949: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_502, [128, 2, 7, 2, 7, 512]);  mul_502 = None
        permute_499: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_949, [0, 1, 3, 2, 4, 5]);  view_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_297: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_499, memory_format = torch.contiguous_format);  permute_499 = None
        view_950: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_297, [512, 7, 7, 512]);  clone_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_951: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_950, [512, 49, 512]);  view_950 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_952: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_951, [25088, 512]);  view_951 = None
        permute_151: "f32[512, 512]" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
        permute_500: "f32[512, 512]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        mm_83: "f32[25088, 512]" = torch.ops.aten.mm.default(view_952, permute_500);  permute_500 = None
        permute_501: "f32[512, 25088]" = torch.ops.aten.permute.default(view_952, [1, 0])
        mm_84: "f32[512, 512]" = torch.ops.aten.mm.default(permute_501, view_401);  permute_501 = view_401 = None
        sum_166: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_952, [0], True);  view_952 = None
        view_953: "f32[512]" = torch.ops.aten.reshape.default(sum_166, [512]);  sum_166 = None
        view_954: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_83, [512, 49, 512]);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_955: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_954, [512, 49, 16, 32]);  view_954 = None
        permute_504: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_955, [0, 2, 1, 3]);  view_955 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_298: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_504, memory_format = torch.contiguous_format);  permute_504 = None
        view_956: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_298, [8192, 49, 32]);  clone_298 = None
        expand_58: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_40, [512, 16, 49, 49])
        view_397: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_58, [8192, 49, 49]);  expand_58 = None
        permute_505: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_397, [0, 2, 1]);  view_397 = None
        bmm_84: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_505, view_956);  permute_505 = None
        bmm_85: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_956, permute_506);  view_956 = permute_506 = None
        view_957: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_84, [512, 16, 49, 32]);  bmm_84 = None
        view_958: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_85, [512, 16, 49, 49]);  bmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_503: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_958, div_40);  view_958 = None
        sum_167: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_503, [-1], True)
        neg_9: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_40);  div_40 = None
        fma_9: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_9, sum_167, mul_503);  neg_9 = sum_167 = mul_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_168: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_9, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_9: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_168, 0);  sum_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_507: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_9, [1, 2, 0]);  squeeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_959: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_507, [2401, 16]);  permute_507 = None
        view_395: "i64[2401]" = torch.ops.aten.reshape.default(primals_220, [-1]);  primals_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_9: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_395], view_959, True);  view_395 = view_959 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_960: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_9, [8192, 49, 49]);  fma_9 = None
        bmm_86: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_508, view_960);  permute_508 = None
        bmm_87: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_960, permute_509);  view_960 = permute_509 = None
        view_961: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_86, [512, 16, 32, 49]);  bmm_86 = None
        view_962: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_87, [512, 16, 49, 32]);  bmm_87 = None
        permute_510: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_961, [0, 1, 3, 2]);  view_961 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_504: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_962, 0.1767766952966369);  view_962 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_9: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_504, permute_510, view_957]);  mul_504 = permute_510 = view_957 = None
        view_963: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_9, [3, 512, 16, 49, 32]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_511: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_963, [1, 3, 0, 2, 4]);  view_963 = None
        clone_299: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_511, memory_format = torch.contiguous_format);  permute_511 = None
        view_964: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_299, [512, 49, 1536]);  clone_299 = None
        view_965: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_964, [25088, 1536]);  view_964 = None
        permute_146: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_217, [1, 0]);  primals_217 = None
        permute_512: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
        mm_85: "f32[25088, 512]" = torch.ops.aten.mm.default(view_965, permute_512);  permute_512 = None
        permute_513: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_965, [1, 0])
        mm_86: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_513, view_389);  permute_513 = view_389 = None
        sum_169: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_965, [0], True);  view_965 = None
        view_966: "f32[1536]" = torch.ops.aten.reshape.default(sum_169, [1536]);  sum_169 = None
        view_967: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_85, [512, 49, 512]);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_968: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_967, [512, 7, 7, 512]);  view_967 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_969: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_968, [128, 2, 2, 7, 7, 512]);  view_968 = None
        permute_516: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_969, [0, 1, 3, 2, 4, 5]);  view_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_300: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_516, memory_format = torch.contiguous_format);  permute_516 = None
        view_970: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_300, [128, 14, 14, 512]);  clone_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_506: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_970, primals_215);  primals_215 = None
        mul_507: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_506, 512)
        sum_170: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_506, [3], True)
        mul_508: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_506, mul_144);  mul_506 = None
        sum_171: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [3], True);  mul_508 = None
        mul_509: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_144, sum_171);  sum_171 = None
        sub_141: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_507, sum_170);  mul_507 = sum_170 = None
        sub_142: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_141, mul_509);  sub_141 = mul_509 = None
        mul_510: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_92, sub_142);  div_92 = sub_142 = None
        mul_511: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_970, mul_144);  mul_144 = None
        sum_172: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 1, 2]);  mul_511 = None
        sum_173: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_970, [0, 1, 2]);  view_970 = None
        add_312: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_948, mul_510);  view_948 = mul_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_971: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_312, [128, 196, 512]);  add_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_25: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_25, torch.float32);  lt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_39: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_25, 0.9434782639145851);  convert_element_type_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_512: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_971, div_39);  div_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_972: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_512, [25088, 512]);  mul_512 = None
        permute_144: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
        permute_517: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        mm_87: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_972, permute_517);  permute_517 = None
        permute_518: "f32[512, 25088]" = torch.ops.aten.permute.default(view_972, [1, 0])
        mm_88: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_518, view_383);  permute_518 = view_383 = None
        sum_174: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_972, [0], True);  view_972 = None
        view_973: "f32[512]" = torch.ops.aten.reshape.default(sum_174, [512]);  sum_174 = None
        view_974: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_87, [128, 196, 2048]);  mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_382: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_54, [128, 196, 2048]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_141: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_382, 0.7071067811865476)
        erf_13: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
        add_151: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_514: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_151, 0.5);  add_151 = None
        mul_515: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_382, view_382)
        mul_516: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_515, -0.5);  mul_515 = None
        exp_34: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_516);  mul_516 = None
        mul_517: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
        mul_518: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_382, mul_517);  view_382 = mul_517 = None
        add_314: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_514, mul_518);  mul_514 = mul_518 = None
        mul_519: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_974, add_314);  view_974 = add_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_975: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_519, [25088, 2048]);  mul_519 = None
        permute_143: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
        permute_521: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        mm_89: "f32[25088, 512]" = torch.ops.aten.mm.default(view_975, permute_521);  permute_521 = None
        permute_522: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_975, [1, 0])
        mm_90: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_522, view_381);  permute_522 = view_381 = None
        sum_175: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_975, [0], True);  view_975 = None
        view_976: "f32[2048]" = torch.ops.aten.reshape.default(sum_175, [2048]);  sum_175 = None
        view_977: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_89, [128, 196, 512]);  mm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_521: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_977, primals_209);  primals_209 = None
        mul_522: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_521, 512)
        sum_176: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_521, [2], True)
        mul_523: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_521, mul_138);  mul_521 = None
        sum_177: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_523, [2], True);  mul_523 = None
        mul_524: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_138, sum_177);  sum_177 = None
        sub_144: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_522, sum_176);  mul_522 = sum_176 = None
        sub_145: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_144, mul_524);  sub_144 = mul_524 = None
        mul_525: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_93, sub_145);  div_93 = sub_145 = None
        mul_526: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_977, mul_138);  mul_138 = None
        sum_178: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_526, [0, 1]);  mul_526 = None
        sum_179: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_977, [0, 1]);  view_977 = None
        add_315: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_971, mul_525);  view_971 = mul_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_978: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_315, [128, 14, 14, 512]);  add_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_24: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_24, torch.float32);  lt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_38: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_24, 0.9434782639145851);  convert_element_type_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_527: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_978, div_38);  div_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_84: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_527, [None, None, fmod_8]);  mul_527 = None
        index_85: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_84, [None, fmod_8]);  index_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_979: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_85, [128, 2, 7, 2, 7, 512]);  index_85 = None
        permute_525: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_979, [0, 1, 3, 2, 4, 5]);  view_979 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_301: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_525, memory_format = torch.contiguous_format);  permute_525 = None
        view_980: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_301, [512, 7, 7, 512]);  clone_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_981: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_980, [512, 49, 512]);  view_980 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_982: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_981, [25088, 512]);  view_981 = None
        permute_141: "f32[512, 512]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
        permute_526: "f32[512, 512]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        mm_91: "f32[25088, 512]" = torch.ops.aten.mm.default(view_982, permute_526);  permute_526 = None
        permute_527: "f32[512, 25088]" = torch.ops.aten.permute.default(view_982, [1, 0])
        mm_92: "f32[512, 512]" = torch.ops.aten.mm.default(permute_527, view_375);  permute_527 = view_375 = None
        sum_180: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_982, [0], True);  view_982 = None
        view_983: "f32[512]" = torch.ops.aten.reshape.default(sum_180, [512]);  sum_180 = None
        view_984: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_91, [512, 49, 512]);  mm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_985: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_984, [512, 49, 16, 32]);  view_984 = None
        permute_530: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_985, [0, 2, 1, 3]);  view_985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_302: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_530, memory_format = torch.contiguous_format);  permute_530 = None
        view_986: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_302, [8192, 49, 32]);  clone_302 = None
        expand_54: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_37, [512, 16, 49, 49])
        view_371: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_54, [8192, 49, 49]);  expand_54 = None
        permute_531: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_371, [0, 2, 1]);  view_371 = None
        bmm_88: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_531, view_986);  permute_531 = None
        bmm_89: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_986, permute_532);  view_986 = permute_532 = None
        view_987: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_88, [512, 16, 49, 32]);  bmm_88 = None
        view_988: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_89, [512, 16, 49, 49]);  bmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_528: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_988, div_37);  view_988 = None
        sum_181: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_528, [-1], True)
        neg_10: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_37);  div_37 = None
        fma_10: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_10, sum_181, mul_528);  neg_10 = sum_181 = mul_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_182: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_10, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_10: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_182, 0);  sum_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_533: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_10, [1, 2, 0]);  squeeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_991: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_533, [2401, 16]);  permute_533 = None
        view_367: "i64[2401]" = torch.ops.aten.reshape.default(primals_206, [-1]);  primals_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_10: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_367], view_991, True);  view_367 = view_991 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_992: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_10, [8192, 49, 49]);  fma_10 = None
        bmm_90: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_534, view_992);  permute_534 = None
        bmm_91: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_992, permute_535);  view_992 = permute_535 = None
        view_993: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_90, [512, 16, 32, 49]);  bmm_90 = None
        view_994: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_91, [512, 16, 49, 32]);  bmm_91 = None
        permute_536: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_993, [0, 1, 3, 2]);  view_993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_529: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_994, 0.1767766952966369);  view_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_10: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_529, permute_536, view_987]);  mul_529 = permute_536 = view_987 = None
        view_995: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_10, [3, 512, 16, 49, 32]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_537: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_995, [1, 3, 0, 2, 4]);  view_995 = None
        clone_303: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_537, memory_format = torch.contiguous_format);  permute_537 = None
        view_996: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_303, [512, 49, 1536]);  clone_303 = None
        view_997: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_996, [25088, 1536]);  view_996 = None
        permute_136: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
        permute_538: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        mm_93: "f32[25088, 512]" = torch.ops.aten.mm.default(view_997, permute_538);  permute_538 = None
        permute_539: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_997, [1, 0])
        mm_94: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_539, view_361);  permute_539 = view_361 = None
        sum_183: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_997, [0], True);  view_997 = None
        view_998: "f32[1536]" = torch.ops.aten.reshape.default(sum_183, [1536]);  sum_183 = None
        view_999: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_93, [512, 49, 512]);  mm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1000: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_999, [512, 7, 7, 512]);  view_999 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1001: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1000, [128, 2, 2, 7, 7, 512]);  view_1000 = None
        permute_542: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1001, [0, 1, 3, 2, 4, 5]);  view_1001 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_304: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_542, memory_format = torch.contiguous_format);  permute_542 = None
        view_1002: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_304, [128, 14, 14, 512]);  clone_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_86: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_1002, [None, None, fmod_10]);  view_1002 = None
        index_87: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_86, [None, fmod_10]);  index_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_531: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_87, primals_200);  primals_200 = None
        mul_532: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_531, 512)
        sum_184: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_531, [3], True)
        mul_533: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_531, mul_134);  mul_531 = None
        sum_185: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_533, [3], True);  mul_533 = None
        mul_534: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_134, sum_185);  sum_185 = None
        sub_147: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_532, sum_184);  mul_532 = sum_184 = None
        sub_148: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_147, mul_534);  sub_147 = mul_534 = None
        mul_535: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_94, sub_148);  div_94 = sub_148 = None
        mul_536: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_87, mul_134);  mul_134 = None
        sum_186: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 1, 2]);  mul_536 = None
        sum_187: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_87, [0, 1, 2]);  index_87 = None
        add_320: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_978, mul_535);  view_978 = mul_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1003: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_320, [128, 196, 512]);  add_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_23: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_23, torch.float32);  lt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_36: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_23, 0.947826087474823);  convert_element_type_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_537: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1003, div_36);  div_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1004: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_537, [25088, 512]);  mul_537 = None
        permute_134: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
        permute_543: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        mm_95: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_1004, permute_543);  permute_543 = None
        permute_544: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1004, [1, 0])
        mm_96: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_544, view_355);  permute_544 = view_355 = None
        sum_188: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1004, [0], True);  view_1004 = None
        view_1005: "f32[512]" = torch.ops.aten.reshape.default(sum_188, [512]);  sum_188 = None
        view_1006: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_95, [128, 196, 2048]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_354: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_50, [128, 196, 2048]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_131: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_354, 0.7071067811865476)
        erf_12: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
        add_138: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_539: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_138, 0.5);  add_138 = None
        mul_540: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_354, view_354)
        mul_541: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_540, -0.5);  mul_540 = None
        exp_35: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_541);  mul_541 = None
        mul_542: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
        mul_543: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_354, mul_542);  view_354 = mul_542 = None
        add_322: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_539, mul_543);  mul_539 = mul_543 = None
        mul_544: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1006, add_322);  view_1006 = add_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1007: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_544, [25088, 2048]);  mul_544 = None
        permute_133: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
        permute_547: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        mm_97: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1007, permute_547);  permute_547 = None
        permute_548: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1007, [1, 0])
        mm_98: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_548, view_353);  permute_548 = view_353 = None
        sum_189: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1007, [0], True);  view_1007 = None
        view_1008: "f32[2048]" = torch.ops.aten.reshape.default(sum_189, [2048]);  sum_189 = None
        view_1009: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_97, [128, 196, 512]);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_546: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1009, primals_194);  primals_194 = None
        mul_547: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_546, 512)
        sum_190: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_546, [2], True)
        mul_548: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_546, mul_128);  mul_546 = None
        sum_191: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_548, [2], True);  mul_548 = None
        mul_549: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_128, sum_191);  sum_191 = None
        sub_150: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_547, sum_190);  mul_547 = sum_190 = None
        sub_151: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_150, mul_549);  sub_150 = mul_549 = None
        mul_550: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_95, sub_151);  div_95 = sub_151 = None
        mul_551: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1009, mul_128);  mul_128 = None
        sum_192: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_551, [0, 1]);  mul_551 = None
        sum_193: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1009, [0, 1]);  view_1009 = None
        add_323: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_1003, mul_550);  view_1003 = mul_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1010: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_323, [128, 14, 14, 512]);  add_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_22: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_22, torch.float32);  lt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_35: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_22, 0.947826087474823);  convert_element_type_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_552: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1010, div_35);  div_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1011: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_552, [128, 2, 7, 2, 7, 512]);  mul_552 = None
        permute_551: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1011, [0, 1, 3, 2, 4, 5]);  view_1011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_305: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
        view_1012: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_305, [512, 7, 7, 512]);  clone_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1013: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_1012, [512, 49, 512]);  view_1012 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1014: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_1013, [25088, 512]);  view_1013 = None
        permute_131: "f32[512, 512]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
        permute_552: "f32[512, 512]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        mm_99: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1014, permute_552);  permute_552 = None
        permute_553: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1014, [1, 0])
        mm_100: "f32[512, 512]" = torch.ops.aten.mm.default(permute_553, view_347);  permute_553 = view_347 = None
        sum_194: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True);  view_1014 = None
        view_1015: "f32[512]" = torch.ops.aten.reshape.default(sum_194, [512]);  sum_194 = None
        view_1016: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_99, [512, 49, 512]);  mm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1017: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1016, [512, 49, 16, 32]);  view_1016 = None
        permute_556: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1017, [0, 2, 1, 3]);  view_1017 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_306: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_556, memory_format = torch.contiguous_format);  permute_556 = None
        view_1018: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_306, [8192, 49, 32]);  clone_306 = None
        expand_50: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_34, [512, 16, 49, 49])
        view_343: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_50, [8192, 49, 49]);  expand_50 = None
        permute_557: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None
        bmm_92: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_557, view_1018);  permute_557 = None
        bmm_93: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_1018, permute_558);  view_1018 = permute_558 = None
        view_1019: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_92, [512, 16, 49, 32]);  bmm_92 = None
        view_1020: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_93, [512, 16, 49, 49]);  bmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_553: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1020, div_34);  view_1020 = None
        sum_195: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_553, [-1], True)
        neg_11: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_34);  div_34 = None
        fma_11: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_11, sum_195, mul_553);  neg_11 = sum_195 = mul_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_196: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_11, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_11: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_196, 0);  sum_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_559: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_11, [1, 2, 0]);  squeeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1021: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_559, [2401, 16]);  permute_559 = None
        view_341: "i64[2401]" = torch.ops.aten.reshape.default(primals_191, [-1]);  primals_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_11: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_341], view_1021, True);  view_341 = view_1021 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1022: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_11, [8192, 49, 49]);  fma_11 = None
        bmm_94: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_560, view_1022);  permute_560 = None
        bmm_95: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_1022, permute_561);  view_1022 = permute_561 = None
        view_1023: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_94, [512, 16, 32, 49]);  bmm_94 = None
        view_1024: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_95, [512, 16, 49, 32]);  bmm_95 = None
        permute_562: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1023, [0, 1, 3, 2]);  view_1023 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_554: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1024, 0.1767766952966369);  view_1024 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_11: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_554, permute_562, view_1019]);  mul_554 = permute_562 = view_1019 = None
        view_1025: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_11, [3, 512, 16, 49, 32]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_563: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1025, [1, 3, 0, 2, 4]);  view_1025 = None
        clone_307: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_563, memory_format = torch.contiguous_format);  permute_563 = None
        view_1026: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_307, [512, 49, 1536]);  clone_307 = None
        view_1027: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_1026, [25088, 1536]);  view_1026 = None
        permute_126: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
        permute_564: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
        mm_101: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1027, permute_564);  permute_564 = None
        permute_565: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1027, [1, 0])
        mm_102: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_565, view_335);  permute_565 = view_335 = None
        sum_197: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1027, [0], True);  view_1027 = None
        view_1028: "f32[1536]" = torch.ops.aten.reshape.default(sum_197, [1536]);  sum_197 = None
        view_1029: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_101, [512, 49, 512]);  mm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1030: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1029, [512, 7, 7, 512]);  view_1029 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1031: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1030, [128, 2, 2, 7, 7, 512]);  view_1030 = None
        permute_568: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1031, [0, 1, 3, 2, 4, 5]);  view_1031 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_308: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_568, memory_format = torch.contiguous_format);  permute_568 = None
        view_1032: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_308, [128, 14, 14, 512]);  clone_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_556: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1032, primals_186);  primals_186 = None
        mul_557: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_556, 512)
        sum_198: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_556, [3], True)
        mul_558: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_556, mul_124);  mul_556 = None
        sum_199: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_558, [3], True);  mul_558 = None
        mul_559: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_124, sum_199);  sum_199 = None
        sub_153: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_557, sum_198);  mul_557 = sum_198 = None
        sub_154: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_153, mul_559);  sub_153 = mul_559 = None
        mul_560: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_96, sub_154);  div_96 = sub_154 = None
        mul_561: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1032, mul_124);  mul_124 = None
        sum_200: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_561, [0, 1, 2]);  mul_561 = None
        sum_201: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1032, [0, 1, 2]);  view_1032 = None
        add_324: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1010, mul_560);  view_1010 = mul_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1033: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_324, [128, 196, 512]);  add_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_21: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_21, torch.float32);  lt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_33: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_21, 0.9521739110350609);  convert_element_type_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_562: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1033, div_33);  div_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1034: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_562, [25088, 512]);  mul_562 = None
        permute_124: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        permute_569: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
        mm_103: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_1034, permute_569);  permute_569 = None
        permute_570: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1034, [1, 0])
        mm_104: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_570, view_329);  permute_570 = view_329 = None
        sum_202: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1034, [0], True);  view_1034 = None
        view_1035: "f32[512]" = torch.ops.aten.reshape.default(sum_202, [512]);  sum_202 = None
        view_1036: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_103, [128, 196, 2048]);  mm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_328: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_46, [128, 196, 2048]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_121: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_328, 0.7071067811865476)
        erf_11: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_121);  mul_121 = None
        add_130: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_564: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_130, 0.5);  add_130 = None
        mul_565: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_328, view_328)
        mul_566: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_565, -0.5);  mul_565 = None
        exp_36: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_566);  mul_566 = None
        mul_567: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
        mul_568: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_328, mul_567);  view_328 = mul_567 = None
        add_326: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_564, mul_568);  mul_564 = mul_568 = None
        mul_569: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1036, add_326);  view_1036 = add_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1037: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_569, [25088, 2048]);  mul_569 = None
        permute_123: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
        permute_573: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        mm_105: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1037, permute_573);  permute_573 = None
        permute_574: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1037, [1, 0])
        mm_106: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_574, view_327);  permute_574 = view_327 = None
        sum_203: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1037, [0], True);  view_1037 = None
        view_1038: "f32[2048]" = torch.ops.aten.reshape.default(sum_203, [2048]);  sum_203 = None
        view_1039: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_105, [128, 196, 512]);  mm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_571: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1039, primals_180);  primals_180 = None
        mul_572: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_571, 512)
        sum_204: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_571, [2], True)
        mul_573: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_571, mul_118);  mul_571 = None
        sum_205: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_573, [2], True);  mul_573 = None
        mul_574: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_118, sum_205);  sum_205 = None
        sub_156: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_572, sum_204);  mul_572 = sum_204 = None
        sub_157: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_156, mul_574);  sub_156 = mul_574 = None
        mul_575: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_97, sub_157);  div_97 = sub_157 = None
        mul_576: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1039, mul_118);  mul_118 = None
        sum_206: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_576, [0, 1]);  mul_576 = None
        sum_207: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1039, [0, 1]);  view_1039 = None
        add_327: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_1033, mul_575);  view_1033 = mul_575 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1040: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_327, [128, 14, 14, 512]);  add_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_20: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_20, torch.float32);  lt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_32: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_20, 0.9521739110350609);  convert_element_type_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_577: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1040, div_32);  div_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_88: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_577, [None, None, fmod_8]);  mul_577 = None
        index_89: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_88, [None, fmod_8]);  index_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1041: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_89, [128, 2, 7, 2, 7, 512]);  index_89 = None
        permute_577: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1041, [0, 1, 3, 2, 4, 5]);  view_1041 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_309: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_577, memory_format = torch.contiguous_format);  permute_577 = None
        view_1042: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_309, [512, 7, 7, 512]);  clone_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1043: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_1042, [512, 49, 512]);  view_1042 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1044: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_1043, [25088, 512]);  view_1043 = None
        permute_121: "f32[512, 512]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
        permute_578: "f32[512, 512]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_107: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1044, permute_578);  permute_578 = None
        permute_579: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1044, [1, 0])
        mm_108: "f32[512, 512]" = torch.ops.aten.mm.default(permute_579, view_321);  permute_579 = view_321 = None
        sum_208: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1044, [0], True);  view_1044 = None
        view_1045: "f32[512]" = torch.ops.aten.reshape.default(sum_208, [512]);  sum_208 = None
        view_1046: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_107, [512, 49, 512]);  mm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1047: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1046, [512, 49, 16, 32]);  view_1046 = None
        permute_582: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1047, [0, 2, 1, 3]);  view_1047 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_310: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_582, memory_format = torch.contiguous_format);  permute_582 = None
        view_1048: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_310, [8192, 49, 32]);  clone_310 = None
        expand_46: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_31, [512, 16, 49, 49])
        view_317: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_46, [8192, 49, 49]);  expand_46 = None
        permute_583: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_317, [0, 2, 1]);  view_317 = None
        bmm_96: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_583, view_1048);  permute_583 = None
        bmm_97: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_1048, permute_584);  view_1048 = permute_584 = None
        view_1049: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_96, [512, 16, 49, 32]);  bmm_96 = None
        view_1050: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_97, [512, 16, 49, 49]);  bmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_578: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1050, div_31);  view_1050 = None
        sum_209: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_578, [-1], True)
        neg_12: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_31);  div_31 = None
        fma_12: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_12, sum_209, mul_578);  neg_12 = sum_209 = mul_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_210: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_12, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_12: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_210, 0);  sum_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_585: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_12, [1, 2, 0]);  squeeze_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1053: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_585, [2401, 16]);  permute_585 = None
        view_313: "i64[2401]" = torch.ops.aten.reshape.default(primals_177, [-1]);  primals_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_12: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_313], view_1053, True);  view_313 = view_1053 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1054: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_12, [8192, 49, 49]);  fma_12 = None
        bmm_98: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_586, view_1054);  permute_586 = None
        bmm_99: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_1054, permute_587);  view_1054 = permute_587 = None
        view_1055: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_98, [512, 16, 32, 49]);  bmm_98 = None
        view_1056: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_99, [512, 16, 49, 32]);  bmm_99 = None
        permute_588: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1055, [0, 1, 3, 2]);  view_1055 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_579: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1056, 0.1767766952966369);  view_1056 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_12: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_579, permute_588, view_1049]);  mul_579 = permute_588 = view_1049 = None
        view_1057: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_12, [3, 512, 16, 49, 32]);  cat_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_589: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1057, [1, 3, 0, 2, 4]);  view_1057 = None
        clone_311: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_589, memory_format = torch.contiguous_format);  permute_589 = None
        view_1058: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_311, [512, 49, 1536]);  clone_311 = None
        view_1059: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_1058, [25088, 1536]);  view_1058 = None
        permute_116: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
        permute_590: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
        mm_109: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1059, permute_590);  permute_590 = None
        permute_591: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1059, [1, 0])
        mm_110: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_591, view_307);  permute_591 = view_307 = None
        sum_211: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1059, [0], True);  view_1059 = None
        view_1060: "f32[1536]" = torch.ops.aten.reshape.default(sum_211, [1536]);  sum_211 = None
        view_1061: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_109, [512, 49, 512]);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1062: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1061, [512, 7, 7, 512]);  view_1061 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1063: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1062, [128, 2, 2, 7, 7, 512]);  view_1062 = None
        permute_594: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1063, [0, 1, 3, 2, 4, 5]);  view_1063 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_312: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_594, memory_format = torch.contiguous_format);  permute_594 = None
        view_1064: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_312, [128, 14, 14, 512]);  clone_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_90: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_1064, [None, None, fmod_10]);  view_1064 = None
        index_91: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_90, [None, fmod_10]);  index_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_581: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_91, primals_171);  primals_171 = None
        mul_582: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_581, 512)
        sum_212: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_581, [3], True)
        mul_583: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_581, mul_114);  mul_581 = None
        sum_213: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_583, [3], True);  mul_583 = None
        mul_584: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_114, sum_213);  sum_213 = None
        sub_159: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_582, sum_212);  mul_582 = sum_212 = None
        sub_160: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_159, mul_584);  sub_159 = mul_584 = None
        mul_585: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_98, sub_160);  div_98 = sub_160 = None
        mul_586: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_91, mul_114);  mul_114 = None
        sum_214: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_586, [0, 1, 2]);  mul_586 = None
        sum_215: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_91, [0, 1, 2]);  index_91 = None
        add_332: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1040, mul_585);  view_1040 = mul_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1065: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_332, [128, 196, 512]);  add_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_19: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_19, torch.float32);  lt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_30: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_19, 0.9565217345952988);  convert_element_type_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_587: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1065, div_30);  div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1066: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_587, [25088, 512]);  mul_587 = None
        permute_114: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
        permute_595: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
        mm_111: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_1066, permute_595);  permute_595 = None
        permute_596: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1066, [1, 0])
        mm_112: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_596, view_301);  permute_596 = view_301 = None
        sum_216: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1066, [0], True);  view_1066 = None
        view_1067: "f32[512]" = torch.ops.aten.reshape.default(sum_216, [512]);  sum_216 = None
        view_1068: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_111, [128, 196, 2048]);  mm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_300: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_42, [128, 196, 2048]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_111: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_300, 0.7071067811865476)
        erf_10: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_117: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_589: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_117, 0.5);  add_117 = None
        mul_590: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_300, view_300)
        mul_591: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_590, -0.5);  mul_590 = None
        exp_37: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_591);  mul_591 = None
        mul_592: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
        mul_593: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_300, mul_592);  view_300 = mul_592 = None
        add_334: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_589, mul_593);  mul_589 = mul_593 = None
        mul_594: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1068, add_334);  view_1068 = add_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1069: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_594, [25088, 2048]);  mul_594 = None
        permute_113: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
        permute_599: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
        mm_113: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1069, permute_599);  permute_599 = None
        permute_600: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1069, [1, 0])
        mm_114: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_600, view_299);  permute_600 = view_299 = None
        sum_217: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1069, [0], True);  view_1069 = None
        view_1070: "f32[2048]" = torch.ops.aten.reshape.default(sum_217, [2048]);  sum_217 = None
        view_1071: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_113, [128, 196, 512]);  mm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_596: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1071, primals_165);  primals_165 = None
        mul_597: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_596, 512)
        sum_218: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_596, [2], True)
        mul_598: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_596, mul_108);  mul_596 = None
        sum_219: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_598, [2], True);  mul_598 = None
        mul_599: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_108, sum_219);  sum_219 = None
        sub_162: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_597, sum_218);  mul_597 = sum_218 = None
        sub_163: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_162, mul_599);  sub_162 = mul_599 = None
        mul_600: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_99, sub_163);  div_99 = sub_163 = None
        mul_601: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1071, mul_108);  mul_108 = None
        sum_220: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_601, [0, 1]);  mul_601 = None
        sum_221: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1071, [0, 1]);  view_1071 = None
        add_335: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_1065, mul_600);  view_1065 = mul_600 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1072: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_335, [128, 14, 14, 512]);  add_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_18: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_18, torch.float32);  lt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_29: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_18, 0.9565217345952988);  convert_element_type_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_602: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1072, div_29);  div_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1073: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_602, [128, 2, 7, 2, 7, 512]);  mul_602 = None
        permute_603: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1073, [0, 1, 3, 2, 4, 5]);  view_1073 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_313: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_603, memory_format = torch.contiguous_format);  permute_603 = None
        view_1074: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_313, [512, 7, 7, 512]);  clone_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1075: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_1074, [512, 49, 512]);  view_1074 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1076: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_1075, [25088, 512]);  view_1075 = None
        permute_111: "f32[512, 512]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
        permute_604: "f32[512, 512]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        mm_115: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1076, permute_604);  permute_604 = None
        permute_605: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1076, [1, 0])
        mm_116: "f32[512, 512]" = torch.ops.aten.mm.default(permute_605, view_293);  permute_605 = view_293 = None
        sum_222: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1076, [0], True);  view_1076 = None
        view_1077: "f32[512]" = torch.ops.aten.reshape.default(sum_222, [512]);  sum_222 = None
        view_1078: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_115, [512, 49, 512]);  mm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1079: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1078, [512, 49, 16, 32]);  view_1078 = None
        permute_608: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1079, [0, 2, 1, 3]);  view_1079 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_314: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_608, memory_format = torch.contiguous_format);  permute_608 = None
        view_1080: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_314, [8192, 49, 32]);  clone_314 = None
        expand_42: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_28, [512, 16, 49, 49])
        view_289: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_42, [8192, 49, 49]);  expand_42 = None
        permute_609: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_289, [0, 2, 1]);  view_289 = None
        bmm_100: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_609, view_1080);  permute_609 = None
        bmm_101: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_1080, permute_610);  view_1080 = permute_610 = None
        view_1081: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_100, [512, 16, 49, 32]);  bmm_100 = None
        view_1082: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_101, [512, 16, 49, 49]);  bmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_603: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1082, div_28);  view_1082 = None
        sum_223: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_603, [-1], True)
        neg_13: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_28);  div_28 = None
        fma_13: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_13, sum_223, mul_603);  neg_13 = sum_223 = mul_603 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_224: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_13, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_13: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_224, 0);  sum_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_611: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_13, [1, 2, 0]);  squeeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1083: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_611, [2401, 16]);  permute_611 = None
        view_287: "i64[2401]" = torch.ops.aten.reshape.default(primals_162, [-1]);  primals_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_13: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_287], view_1083, True);  view_287 = view_1083 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1084: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_13, [8192, 49, 49]);  fma_13 = None
        bmm_102: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_612, view_1084);  permute_612 = None
        bmm_103: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_1084, permute_613);  view_1084 = permute_613 = None
        view_1085: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_102, [512, 16, 32, 49]);  bmm_102 = None
        view_1086: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_103, [512, 16, 49, 32]);  bmm_103 = None
        permute_614: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1085, [0, 1, 3, 2]);  view_1085 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_604: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1086, 0.1767766952966369);  view_1086 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_13: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_604, permute_614, view_1081]);  mul_604 = permute_614 = view_1081 = None
        view_1087: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_13, [3, 512, 16, 49, 32]);  cat_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_615: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1087, [1, 3, 0, 2, 4]);  view_1087 = None
        clone_315: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_615, memory_format = torch.contiguous_format);  permute_615 = None
        view_1088: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_315, [512, 49, 1536]);  clone_315 = None
        view_1089: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_1088, [25088, 1536]);  view_1088 = None
        permute_106: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
        permute_616: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
        mm_117: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1089, permute_616);  permute_616 = None
        permute_617: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1089, [1, 0])
        mm_118: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_617, view_281);  permute_617 = view_281 = None
        sum_225: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1089, [0], True);  view_1089 = None
        view_1090: "f32[1536]" = torch.ops.aten.reshape.default(sum_225, [1536]);  sum_225 = None
        view_1091: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_117, [512, 49, 512]);  mm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1092: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1091, [512, 7, 7, 512]);  view_1091 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1093: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1092, [128, 2, 2, 7, 7, 512]);  view_1092 = None
        permute_620: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1093, [0, 1, 3, 2, 4, 5]);  view_1093 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_316: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_620, memory_format = torch.contiguous_format);  permute_620 = None
        view_1094: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_316, [128, 14, 14, 512]);  clone_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_606: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1094, primals_157);  primals_157 = None
        mul_607: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_606, 512)
        sum_226: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_606, [3], True)
        mul_608: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_606, mul_104);  mul_606 = None
        sum_227: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_608, [3], True);  mul_608 = None
        mul_609: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_104, sum_227);  sum_227 = None
        sub_165: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_607, sum_226);  mul_607 = sum_226 = None
        sub_166: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_165, mul_609);  sub_165 = mul_609 = None
        mul_610: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_100, sub_166);  div_100 = sub_166 = None
        mul_611: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1094, mul_104);  mul_104 = None
        sum_228: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_611, [0, 1, 2]);  mul_611 = None
        sum_229: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1094, [0, 1, 2]);  view_1094 = None
        add_336: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1072, mul_610);  view_1072 = mul_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1095: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_336, [128, 196, 512]);  add_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_17: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_17, torch.float32);  lt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_27: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_17, 0.960869561880827);  convert_element_type_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_612: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1095, div_27);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1096: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_612, [25088, 512]);  mul_612 = None
        permute_104: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
        permute_621: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
        mm_119: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_1096, permute_621);  permute_621 = None
        permute_622: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1096, [1, 0])
        mm_120: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_622, view_275);  permute_622 = view_275 = None
        sum_230: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1096, [0], True);  view_1096 = None
        view_1097: "f32[512]" = torch.ops.aten.reshape.default(sum_230, [512]);  sum_230 = None
        view_1098: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_119, [128, 196, 2048]);  mm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_274: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_38, [128, 196, 2048]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_101: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_274, 0.7071067811865476)
        erf_9: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_101);  mul_101 = None
        add_109: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_614: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_109, 0.5);  add_109 = None
        mul_615: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_274, view_274)
        mul_616: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_615, -0.5);  mul_615 = None
        exp_38: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_616);  mul_616 = None
        mul_617: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
        mul_618: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_274, mul_617);  view_274 = mul_617 = None
        add_338: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_614, mul_618);  mul_614 = mul_618 = None
        mul_619: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1098, add_338);  view_1098 = add_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1099: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_619, [25088, 2048]);  mul_619 = None
        permute_103: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
        permute_625: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        mm_121: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1099, permute_625);  permute_625 = None
        permute_626: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1099, [1, 0])
        mm_122: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_626, view_273);  permute_626 = view_273 = None
        sum_231: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1099, [0], True);  view_1099 = None
        view_1100: "f32[2048]" = torch.ops.aten.reshape.default(sum_231, [2048]);  sum_231 = None
        view_1101: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_121, [128, 196, 512]);  mm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_621: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1101, primals_151);  primals_151 = None
        mul_622: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_621, 512)
        sum_232: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_621, [2], True)
        mul_623: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_621, mul_98);  mul_621 = None
        sum_233: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_623, [2], True);  mul_623 = None
        mul_624: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_98, sum_233);  sum_233 = None
        sub_168: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_622, sum_232);  mul_622 = sum_232 = None
        sub_169: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_168, mul_624);  sub_168 = mul_624 = None
        mul_625: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_101, sub_169);  div_101 = sub_169 = None
        mul_626: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1101, mul_98);  mul_98 = None
        sum_234: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 1]);  mul_626 = None
        sum_235: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1101, [0, 1]);  view_1101 = None
        add_339: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_1095, mul_625);  view_1095 = mul_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1102: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_339, [128, 14, 14, 512]);  add_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_16: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_16, torch.float32);  lt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_26: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_16, 0.960869561880827);  convert_element_type_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_627: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1102, div_26);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_92: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_627, [None, None, fmod_8]);  mul_627 = None
        index_93: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_92, [None, fmod_8]);  index_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1103: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_93, [128, 2, 7, 2, 7, 512]);  index_93 = None
        permute_629: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1103, [0, 1, 3, 2, 4, 5]);  view_1103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_317: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_629, memory_format = torch.contiguous_format);  permute_629 = None
        view_1104: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_317, [512, 7, 7, 512]);  clone_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1105: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_1104, [512, 49, 512]);  view_1104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1106: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_1105, [25088, 512]);  view_1105 = None
        permute_101: "f32[512, 512]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
        permute_630: "f32[512, 512]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_123: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1106, permute_630);  permute_630 = None
        permute_631: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1106, [1, 0])
        mm_124: "f32[512, 512]" = torch.ops.aten.mm.default(permute_631, view_267);  permute_631 = view_267 = None
        sum_236: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1106, [0], True);  view_1106 = None
        view_1107: "f32[512]" = torch.ops.aten.reshape.default(sum_236, [512]);  sum_236 = None
        view_1108: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_123, [512, 49, 512]);  mm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1109: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1108, [512, 49, 16, 32]);  view_1108 = None
        permute_634: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1109, [0, 2, 1, 3]);  view_1109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_318: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_634, memory_format = torch.contiguous_format);  permute_634 = None
        view_1110: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_318, [8192, 49, 32]);  clone_318 = None
        expand_38: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_25, [512, 16, 49, 49])
        view_263: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_38, [8192, 49, 49]);  expand_38 = None
        permute_635: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_263, [0, 2, 1]);  view_263 = None
        bmm_104: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_635, view_1110);  permute_635 = None
        bmm_105: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_1110, permute_636);  view_1110 = permute_636 = None
        view_1111: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_104, [512, 16, 49, 32]);  bmm_104 = None
        view_1112: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_105, [512, 16, 49, 49]);  bmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_628: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1112, div_25);  view_1112 = None
        sum_237: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_628, [-1], True)
        neg_14: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_25);  div_25 = None
        fma_14: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_14, sum_237, mul_628);  neg_14 = sum_237 = mul_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_238: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_14, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_14: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_238, 0);  sum_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_637: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_14, [1, 2, 0]);  squeeze_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1115: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_637, [2401, 16]);  permute_637 = None
        view_259: "i64[2401]" = torch.ops.aten.reshape.default(primals_148, [-1]);  primals_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_14: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_259], view_1115, True);  view_259 = view_1115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1116: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_14, [8192, 49, 49]);  fma_14 = None
        bmm_106: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_638, view_1116);  permute_638 = None
        bmm_107: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_1116, permute_639);  view_1116 = permute_639 = None
        view_1117: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_106, [512, 16, 32, 49]);  bmm_106 = None
        view_1118: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_107, [512, 16, 49, 32]);  bmm_107 = None
        permute_640: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1117, [0, 1, 3, 2]);  view_1117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_629: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1118, 0.1767766952966369);  view_1118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_14: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_629, permute_640, view_1111]);  mul_629 = permute_640 = view_1111 = None
        view_1119: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_14, [3, 512, 16, 49, 32]);  cat_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_641: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1119, [1, 3, 0, 2, 4]);  view_1119 = None
        clone_319: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_641, memory_format = torch.contiguous_format);  permute_641 = None
        view_1120: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_319, [512, 49, 1536]);  clone_319 = None
        view_1121: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_1120, [25088, 1536]);  view_1120 = None
        permute_96: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
        permute_642: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        mm_125: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1121, permute_642);  permute_642 = None
        permute_643: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1121, [1, 0])
        mm_126: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_643, view_253);  permute_643 = view_253 = None
        sum_239: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1121, [0], True);  view_1121 = None
        view_1122: "f32[1536]" = torch.ops.aten.reshape.default(sum_239, [1536]);  sum_239 = None
        view_1123: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_125, [512, 49, 512]);  mm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1124: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1123, [512, 7, 7, 512]);  view_1123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1125: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1124, [128, 2, 2, 7, 7, 512]);  view_1124 = None
        permute_646: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1125, [0, 1, 3, 2, 4, 5]);  view_1125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_320: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_646, memory_format = torch.contiguous_format);  permute_646 = None
        view_1126: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_320, [128, 14, 14, 512]);  clone_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_94: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_1126, [None, None, fmod_10]);  view_1126 = None
        index_95: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_94, [None, fmod_10]);  index_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_631: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_95, primals_142);  primals_142 = None
        mul_632: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_631, 512)
        sum_240: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_631, [3], True)
        mul_633: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_631, mul_94);  mul_631 = None
        sum_241: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_633, [3], True);  mul_633 = None
        mul_634: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_94, sum_241);  sum_241 = None
        sub_171: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_632, sum_240);  mul_632 = sum_240 = None
        sub_172: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_171, mul_634);  sub_171 = mul_634 = None
        mul_635: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_102, sub_172);  div_102 = sub_172 = None
        mul_636: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_95, mul_94);  mul_94 = None
        sum_242: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_636, [0, 1, 2]);  mul_636 = None
        sum_243: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_95, [0, 1, 2]);  index_95 = None
        add_344: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1102, mul_635);  view_1102 = mul_635 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1127: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_344, [128, 196, 512]);  add_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_15: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_15, torch.float32);  lt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_24: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_15, 0.9652173891663551);  convert_element_type_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_637: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1127, div_24);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1128: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_637, [25088, 512]);  mul_637 = None
        permute_94: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
        permute_647: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
        mm_127: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_1128, permute_647);  permute_647 = None
        permute_648: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1128, [1, 0])
        mm_128: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_648, view_247);  permute_648 = view_247 = None
        sum_244: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1128, [0], True);  view_1128 = None
        view_1129: "f32[512]" = torch.ops.aten.reshape.default(sum_244, [512]);  sum_244 = None
        view_1130: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_127, [128, 196, 2048]);  mm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_246: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_34, [128, 196, 2048]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_91: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_246, 0.7071067811865476)
        erf_8: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_91);  mul_91 = None
        add_96: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_639: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
        mul_640: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_246, view_246)
        mul_641: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_640, -0.5);  mul_640 = None
        exp_39: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_641);  mul_641 = None
        mul_642: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
        mul_643: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_246, mul_642);  view_246 = mul_642 = None
        add_346: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_639, mul_643);  mul_639 = mul_643 = None
        mul_644: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1130, add_346);  view_1130 = add_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1131: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_644, [25088, 2048]);  mul_644 = None
        permute_93: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
        permute_651: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
        mm_129: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1131, permute_651);  permute_651 = None
        permute_652: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1131, [1, 0])
        mm_130: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_652, view_245);  permute_652 = view_245 = None
        sum_245: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1131, [0], True);  view_1131 = None
        view_1132: "f32[2048]" = torch.ops.aten.reshape.default(sum_245, [2048]);  sum_245 = None
        view_1133: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_129, [128, 196, 512]);  mm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_646: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1133, primals_136);  primals_136 = None
        mul_647: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_646, 512)
        sum_246: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_646, [2], True)
        mul_648: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_646, mul_88);  mul_646 = None
        sum_247: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_648, [2], True);  mul_648 = None
        mul_649: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_88, sum_247);  sum_247 = None
        sub_174: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_647, sum_246);  mul_647 = sum_246 = None
        sub_175: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_174, mul_649);  sub_174 = mul_649 = None
        mul_650: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_103, sub_175);  div_103 = sub_175 = None
        mul_651: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1133, mul_88);  mul_88 = None
        sum_248: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_651, [0, 1]);  mul_651 = None
        sum_249: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1133, [0, 1]);  view_1133 = None
        add_347: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_1127, mul_650);  view_1127 = mul_650 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1134: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_347, [128, 14, 14, 512]);  add_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_14: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_14, torch.float32);  lt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_23: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_14, 0.9652173891663551);  convert_element_type_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_652: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1134, div_23);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1135: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_652, [128, 2, 7, 2, 7, 512]);  mul_652 = None
        permute_655: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1135, [0, 1, 3, 2, 4, 5]);  view_1135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_321: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_655, memory_format = torch.contiguous_format);  permute_655 = None
        view_1136: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_321, [512, 7, 7, 512]);  clone_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1137: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_1136, [512, 49, 512]);  view_1136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1138: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_1137, [25088, 512]);  view_1137 = None
        permute_91: "f32[512, 512]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
        permute_656: "f32[512, 512]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
        mm_131: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1138, permute_656);  permute_656 = None
        permute_657: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1138, [1, 0])
        mm_132: "f32[512, 512]" = torch.ops.aten.mm.default(permute_657, view_239);  permute_657 = view_239 = None
        sum_250: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1138, [0], True);  view_1138 = None
        view_1139: "f32[512]" = torch.ops.aten.reshape.default(sum_250, [512]);  sum_250 = None
        view_1140: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_131, [512, 49, 512]);  mm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1141: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1140, [512, 49, 16, 32]);  view_1140 = None
        permute_660: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1141, [0, 2, 1, 3]);  view_1141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_322: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_660, memory_format = torch.contiguous_format);  permute_660 = None
        view_1142: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_322, [8192, 49, 32]);  clone_322 = None
        expand_34: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_22, [512, 16, 49, 49])
        view_235: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_34, [8192, 49, 49]);  expand_34 = None
        permute_661: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
        bmm_108: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_661, view_1142);  permute_661 = None
        bmm_109: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_1142, permute_662);  view_1142 = permute_662 = None
        view_1143: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_108, [512, 16, 49, 32]);  bmm_108 = None
        view_1144: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_109, [512, 16, 49, 49]);  bmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_653: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1144, div_22);  view_1144 = None
        sum_251: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_653, [-1], True)
        neg_15: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_22);  div_22 = None
        fma_15: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_15, sum_251, mul_653);  neg_15 = sum_251 = mul_653 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_252: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_15, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_15: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_252, 0);  sum_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_663: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_15, [1, 2, 0]);  squeeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1145: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_663, [2401, 16]);  permute_663 = None
        view_233: "i64[2401]" = torch.ops.aten.reshape.default(primals_133, [-1]);  primals_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_15: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_233], view_1145, True);  view_233 = view_1145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1146: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_15, [8192, 49, 49]);  fma_15 = None
        bmm_110: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_664, view_1146);  permute_664 = None
        bmm_111: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_1146, permute_665);  view_1146 = permute_665 = None
        view_1147: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_110, [512, 16, 32, 49]);  bmm_110 = None
        view_1148: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_111, [512, 16, 49, 32]);  bmm_111 = None
        permute_666: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1147, [0, 1, 3, 2]);  view_1147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_654: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1148, 0.1767766952966369);  view_1148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_15: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_654, permute_666, view_1143]);  mul_654 = permute_666 = view_1143 = None
        view_1149: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_15, [3, 512, 16, 49, 32]);  cat_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_667: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1149, [1, 3, 0, 2, 4]);  view_1149 = None
        clone_323: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_667, memory_format = torch.contiguous_format);  permute_667 = None
        view_1150: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_323, [512, 49, 1536]);  clone_323 = None
        view_1151: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_1150, [25088, 1536]);  view_1150 = None
        permute_86: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
        permute_668: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_133: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1151, permute_668);  permute_668 = None
        permute_669: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1151, [1, 0])
        mm_134: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_669, view_227);  permute_669 = view_227 = None
        sum_253: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1151, [0], True);  view_1151 = None
        view_1152: "f32[1536]" = torch.ops.aten.reshape.default(sum_253, [1536]);  sum_253 = None
        view_1153: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_133, [512, 49, 512]);  mm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1154: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1153, [512, 7, 7, 512]);  view_1153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1155: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1154, [128, 2, 2, 7, 7, 512]);  view_1154 = None
        permute_672: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1155, [0, 1, 3, 2, 4, 5]);  view_1155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_324: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_672, memory_format = torch.contiguous_format);  permute_672 = None
        view_1156: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_324, [128, 14, 14, 512]);  clone_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_656: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1156, primals_128);  primals_128 = None
        mul_657: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_656, 512)
        sum_254: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_656, [3], True)
        mul_658: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_656, mul_84);  mul_656 = None
        sum_255: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_658, [3], True);  mul_658 = None
        mul_659: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_84, sum_255);  sum_255 = None
        sub_177: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_657, sum_254);  mul_657 = sum_254 = None
        sub_178: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_177, mul_659);  sub_177 = mul_659 = None
        mul_660: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_104, sub_178);  div_104 = sub_178 = None
        mul_661: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1156, mul_84);  mul_84 = None
        sum_256: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 1, 2]);  mul_661 = None
        sum_257: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1156, [0, 1, 2]);  view_1156 = None
        add_348: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1134, mul_660);  view_1134 = mul_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1157: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_348, [128, 196, 512]);  add_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_13: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_13, torch.float32);  lt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_21: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_13, 0.9695652164518833);  convert_element_type_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_662: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1157, div_21);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1158: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_662, [25088, 512]);  mul_662 = None
        permute_84: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
        permute_673: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
        mm_135: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_1158, permute_673);  permute_673 = None
        permute_674: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1158, [1, 0])
        mm_136: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_674, view_221);  permute_674 = view_221 = None
        sum_258: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1158, [0], True);  view_1158 = None
        view_1159: "f32[512]" = torch.ops.aten.reshape.default(sum_258, [512]);  sum_258 = None
        view_1160: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_135, [128, 196, 2048]);  mm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_220: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_30, [128, 196, 2048]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_81: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_220, 0.7071067811865476)
        erf_7: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_81);  mul_81 = None
        add_88: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_664: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
        mul_665: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_220, view_220)
        mul_666: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_665, -0.5);  mul_665 = None
        exp_40: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_666);  mul_666 = None
        mul_667: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
        mul_668: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_220, mul_667);  view_220 = mul_667 = None
        add_350: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_664, mul_668);  mul_664 = mul_668 = None
        mul_669: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1160, add_350);  view_1160 = add_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1161: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_669, [25088, 2048]);  mul_669 = None
        permute_83: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
        permute_677: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
        mm_137: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1161, permute_677);  permute_677 = None
        permute_678: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1161, [1, 0])
        mm_138: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_678, view_219);  permute_678 = view_219 = None
        sum_259: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1161, [0], True);  view_1161 = None
        view_1162: "f32[2048]" = torch.ops.aten.reshape.default(sum_259, [2048]);  sum_259 = None
        view_1163: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_137, [128, 196, 512]);  mm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_671: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1163, primals_122);  primals_122 = None
        mul_672: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_671, 512)
        sum_260: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_671, [2], True)
        mul_673: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_671, mul_78);  mul_671 = None
        sum_261: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_673, [2], True);  mul_673 = None
        mul_674: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_78, sum_261);  sum_261 = None
        sub_180: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_672, sum_260);  mul_672 = sum_260 = None
        sub_181: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_180, mul_674);  sub_180 = mul_674 = None
        mul_675: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_105, sub_181);  div_105 = sub_181 = None
        mul_676: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1163, mul_78);  mul_78 = None
        sum_262: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_676, [0, 1]);  mul_676 = None
        sum_263: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1163, [0, 1]);  view_1163 = None
        add_351: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_1157, mul_675);  view_1157 = mul_675 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1164: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_351, [128, 14, 14, 512]);  add_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_12: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_12, torch.float32);  lt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_20: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_12, 0.9695652164518833);  convert_element_type_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_677: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1164, div_20);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_96: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_677, [None, None, fmod_8]);  mul_677 = None
        index_97: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_96, [None, fmod_8]);  index_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1165: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_97, [128, 2, 7, 2, 7, 512]);  index_97 = None
        permute_681: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1165, [0, 1, 3, 2, 4, 5]);  view_1165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_325: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_681, memory_format = torch.contiguous_format);  permute_681 = None
        view_1166: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_325, [512, 7, 7, 512]);  clone_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1167: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_1166, [512, 49, 512]);  view_1166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1168: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_1167, [25088, 512]);  view_1167 = None
        permute_81: "f32[512, 512]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        permute_682: "f32[512, 512]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        mm_139: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1168, permute_682);  permute_682 = None
        permute_683: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1168, [1, 0])
        mm_140: "f32[512, 512]" = torch.ops.aten.mm.default(permute_683, view_213);  permute_683 = view_213 = None
        sum_264: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1168, [0], True);  view_1168 = None
        view_1169: "f32[512]" = torch.ops.aten.reshape.default(sum_264, [512]);  sum_264 = None
        view_1170: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_139, [512, 49, 512]);  mm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1171: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1170, [512, 49, 16, 32]);  view_1170 = None
        permute_686: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1171, [0, 2, 1, 3]);  view_1171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_326: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_686, memory_format = torch.contiguous_format);  permute_686 = None
        view_1172: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_326, [8192, 49, 32]);  clone_326 = None
        expand_30: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_19, [512, 16, 49, 49])
        view_209: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_30, [8192, 49, 49]);  expand_30 = None
        permute_687: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
        bmm_112: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_687, view_1172);  permute_687 = None
        bmm_113: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_1172, permute_688);  view_1172 = permute_688 = None
        view_1173: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_112, [512, 16, 49, 32]);  bmm_112 = None
        view_1174: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_113, [512, 16, 49, 49]);  bmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_678: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1174, div_19);  view_1174 = None
        sum_265: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_678, [-1], True)
        neg_16: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_19);  div_19 = None
        fma_16: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_16, sum_265, mul_678);  neg_16 = sum_265 = mul_678 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_266: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_16, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_16: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_266, 0);  sum_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_689: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_16, [1, 2, 0]);  squeeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1177: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_689, [2401, 16]);  permute_689 = None
        view_205: "i64[2401]" = torch.ops.aten.reshape.default(primals_119, [-1]);  primals_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_16: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_205], view_1177, True);  view_205 = view_1177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1178: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_16, [8192, 49, 49]);  fma_16 = None
        bmm_114: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_690, view_1178);  permute_690 = None
        bmm_115: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_1178, permute_691);  view_1178 = permute_691 = None
        view_1179: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_114, [512, 16, 32, 49]);  bmm_114 = None
        view_1180: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_115, [512, 16, 49, 32]);  bmm_115 = None
        permute_692: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1179, [0, 1, 3, 2]);  view_1179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_679: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1180, 0.1767766952966369);  view_1180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_16: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_679, permute_692, view_1173]);  mul_679 = permute_692 = view_1173 = None
        view_1181: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_16, [3, 512, 16, 49, 32]);  cat_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_693: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1181, [1, 3, 0, 2, 4]);  view_1181 = None
        clone_327: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_693, memory_format = torch.contiguous_format);  permute_693 = None
        view_1182: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_327, [512, 49, 1536]);  clone_327 = None
        view_1183: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_1182, [25088, 1536]);  view_1182 = None
        permute_76: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
        permute_694: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_141: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1183, permute_694);  permute_694 = None
        permute_695: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1183, [1, 0])
        mm_142: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_695, view_199);  permute_695 = view_199 = None
        sum_267: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1183, [0], True);  view_1183 = None
        view_1184: "f32[1536]" = torch.ops.aten.reshape.default(sum_267, [1536]);  sum_267 = None
        view_1185: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_141, [512, 49, 512]);  mm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1186: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1185, [512, 7, 7, 512]);  view_1185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1187: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1186, [128, 2, 2, 7, 7, 512]);  view_1186 = None
        permute_698: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1187, [0, 1, 3, 2, 4, 5]);  view_1187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_328: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_698, memory_format = torch.contiguous_format);  permute_698 = None
        view_1188: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_328, [128, 14, 14, 512]);  clone_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_98: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_1188, [None, None, fmod_10]);  view_1188 = None
        index_99: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_98, [None, fmod_10]);  index_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_681: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_99, primals_113);  primals_113 = None
        mul_682: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_681, 512)
        sum_268: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_681, [3], True)
        mul_683: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_681, mul_74);  mul_681 = None
        sum_269: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_683, [3], True);  mul_683 = None
        mul_684: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_74, sum_269);  sum_269 = None
        sub_183: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_682, sum_268);  mul_682 = sum_268 = None
        sub_184: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_183, mul_684);  sub_183 = mul_684 = None
        mul_685: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_106, sub_184);  div_106 = sub_184 = None
        mul_686: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_99, mul_74);  mul_74 = None
        sum_270: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_686, [0, 1, 2]);  mul_686 = None
        sum_271: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_99, [0, 1, 2]);  index_99 = None
        add_356: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1164, mul_685);  view_1164 = mul_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1189: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_356, [128, 196, 512]);  add_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_11: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_11, torch.float32);  lt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_18: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_11, 0.9739130418747663);  convert_element_type_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_687: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1189, div_18);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1190: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_687, [25088, 512]);  mul_687 = None
        permute_74: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
        permute_699: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        mm_143: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_1190, permute_699);  permute_699 = None
        permute_700: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1190, [1, 0])
        mm_144: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_700, view_193);  permute_700 = view_193 = None
        sum_272: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1190, [0], True);  view_1190 = None
        view_1191: "f32[512]" = torch.ops.aten.reshape.default(sum_272, [512]);  sum_272 = None
        view_1192: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_143, [128, 196, 2048]);  mm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_192: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_26, [128, 196, 2048]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_71: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_192, 0.7071067811865476)
        erf_6: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_75: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_689: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_75, 0.5);  add_75 = None
        mul_690: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_192, view_192)
        mul_691: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_690, -0.5);  mul_690 = None
        exp_41: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_691);  mul_691 = None
        mul_692: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
        mul_693: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_192, mul_692);  view_192 = mul_692 = None
        add_358: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_689, mul_693);  mul_689 = mul_693 = None
        mul_694: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1192, add_358);  view_1192 = add_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1193: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_694, [25088, 2048]);  mul_694 = None
        permute_73: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        permute_703: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
        mm_145: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1193, permute_703);  permute_703 = None
        permute_704: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1193, [1, 0])
        mm_146: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_704, view_191);  permute_704 = view_191 = None
        sum_273: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1193, [0], True);  view_1193 = None
        view_1194: "f32[2048]" = torch.ops.aten.reshape.default(sum_273, [2048]);  sum_273 = None
        view_1195: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_145, [128, 196, 512]);  mm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_696: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1195, primals_107);  primals_107 = None
        mul_697: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_696, 512)
        sum_274: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_696, [2], True)
        mul_698: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_696, mul_68);  mul_696 = None
        sum_275: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_698, [2], True);  mul_698 = None
        mul_699: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_68, sum_275);  sum_275 = None
        sub_186: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_697, sum_274);  mul_697 = sum_274 = None
        sub_187: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_186, mul_699);  sub_186 = mul_699 = None
        mul_700: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_107, sub_187);  div_107 = sub_187 = None
        mul_701: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1195, mul_68);  mul_68 = None
        sum_276: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_701, [0, 1]);  mul_701 = None
        sum_277: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1195, [0, 1]);  view_1195 = None
        add_359: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_1189, mul_700);  view_1189 = mul_700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1196: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_359, [128, 14, 14, 512]);  add_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_10: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_10, torch.float32);  lt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_17: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_10, 0.9739130418747663);  convert_element_type_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_702: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1196, div_17);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1197: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_702, [128, 2, 7, 2, 7, 512]);  mul_702 = None
        permute_707: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1197, [0, 1, 3, 2, 4, 5]);  view_1197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_329: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_707, memory_format = torch.contiguous_format);  permute_707 = None
        view_1198: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_329, [512, 7, 7, 512]);  clone_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1199: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_1198, [512, 49, 512]);  view_1198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1200: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_1199, [25088, 512]);  view_1199 = None
        permute_71: "f32[512, 512]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        permute_708: "f32[512, 512]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        mm_147: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1200, permute_708);  permute_708 = None
        permute_709: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1200, [1, 0])
        mm_148: "f32[512, 512]" = torch.ops.aten.mm.default(permute_709, view_185);  permute_709 = view_185 = None
        sum_278: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1200, [0], True);  view_1200 = None
        view_1201: "f32[512]" = torch.ops.aten.reshape.default(sum_278, [512]);  sum_278 = None
        view_1202: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_147, [512, 49, 512]);  mm_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1203: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1202, [512, 49, 16, 32]);  view_1202 = None
        permute_712: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1203, [0, 2, 1, 3]);  view_1203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_330: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_712, memory_format = torch.contiguous_format);  permute_712 = None
        view_1204: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_330, [8192, 49, 32]);  clone_330 = None
        expand_26: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_16, [512, 16, 49, 49])
        view_181: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_26, [8192, 49, 49]);  expand_26 = None
        permute_713: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_181, [0, 2, 1]);  view_181 = None
        bmm_116: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_713, view_1204);  permute_713 = None
        bmm_117: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_1204, permute_714);  view_1204 = permute_714 = None
        view_1205: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_116, [512, 16, 49, 32]);  bmm_116 = None
        view_1206: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_117, [512, 16, 49, 49]);  bmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_703: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1206, div_16);  view_1206 = None
        sum_279: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_703, [-1], True)
        neg_17: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_16);  div_16 = None
        fma_17: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_17, sum_279, mul_703);  neg_17 = sum_279 = mul_703 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_280: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_17, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_17: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_280, 0);  sum_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_715: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_17, [1, 2, 0]);  squeeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1207: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_715, [2401, 16]);  permute_715 = None
        view_179: "i64[2401]" = torch.ops.aten.reshape.default(primals_104, [-1]);  primals_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_17: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_179], view_1207, True);  view_179 = view_1207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1208: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_17, [8192, 49, 49]);  fma_17 = None
        bmm_118: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_716, view_1208);  permute_716 = None
        bmm_119: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_1208, permute_717);  view_1208 = permute_717 = None
        view_1209: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_118, [512, 16, 32, 49]);  bmm_118 = None
        view_1210: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_119, [512, 16, 49, 32]);  bmm_119 = None
        permute_718: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1209, [0, 1, 3, 2]);  view_1209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_704: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1210, 0.1767766952966369);  view_1210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_17: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_704, permute_718, view_1205]);  mul_704 = permute_718 = view_1205 = None
        view_1211: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_17, [3, 512, 16, 49, 32]);  cat_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_719: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1211, [1, 3, 0, 2, 4]);  view_1211 = None
        clone_331: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_719, memory_format = torch.contiguous_format);  permute_719 = None
        view_1212: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_331, [512, 49, 1536]);  clone_331 = None
        view_1213: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_1212, [25088, 1536]);  view_1212 = None
        permute_66: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        permute_720: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_149: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1213, permute_720);  permute_720 = None
        permute_721: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1213, [1, 0])
        mm_150: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_721, view_173);  permute_721 = view_173 = None
        sum_281: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1213, [0], True);  view_1213 = None
        view_1214: "f32[1536]" = torch.ops.aten.reshape.default(sum_281, [1536]);  sum_281 = None
        view_1215: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_149, [512, 49, 512]);  mm_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1216: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1215, [512, 7, 7, 512]);  view_1215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1217: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1216, [128, 2, 2, 7, 7, 512]);  view_1216 = None
        permute_724: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1217, [0, 1, 3, 2, 4, 5]);  view_1217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_332: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_724, memory_format = torch.contiguous_format);  permute_724 = None
        view_1218: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_332, [128, 14, 14, 512]);  clone_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_706: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1218, primals_99);  primals_99 = None
        mul_707: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_706, 512)
        sum_282: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_706, [3], True)
        mul_708: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_706, mul_64);  mul_706 = None
        sum_283: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_708, [3], True);  mul_708 = None
        mul_709: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_64, sum_283);  sum_283 = None
        sub_189: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_707, sum_282);  mul_707 = sum_282 = None
        sub_190: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_189, mul_709);  sub_189 = mul_709 = None
        mul_710: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_108, sub_190);  div_108 = sub_190 = None
        mul_711: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1218, mul_64);  mul_64 = None
        sum_284: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_711, [0, 1, 2]);  mul_711 = None
        sum_285: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1218, [0, 1, 2]);  view_1218 = None
        add_360: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1196, mul_710);  view_1196 = mul_710 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1219: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_360, [128, 196, 512]);  add_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_9: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_9, torch.float32);  lt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_15: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_9, 0.9782608672976494);  convert_element_type_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_712: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1219, div_15);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1220: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_712, [25088, 512]);  mul_712 = None
        permute_64: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        permute_725: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_151: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_1220, permute_725);  permute_725 = None
        permute_726: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1220, [1, 0])
        mm_152: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_726, view_167);  permute_726 = view_167 = None
        sum_286: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1220, [0], True);  view_1220 = None
        view_1221: "f32[512]" = torch.ops.aten.reshape.default(sum_286, [512]);  sum_286 = None
        view_1222: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_151, [128, 196, 2048]);  mm_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_166: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_22, [128, 196, 2048]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_61: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_166, 0.7071067811865476)
        erf_5: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_67: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_714: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_67, 0.5);  add_67 = None
        mul_715: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_166, view_166)
        mul_716: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_715, -0.5);  mul_715 = None
        exp_42: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_716);  mul_716 = None
        mul_717: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
        mul_718: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_166, mul_717);  view_166 = mul_717 = None
        add_362: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_714, mul_718);  mul_714 = mul_718 = None
        mul_719: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1222, add_362);  view_1222 = add_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1223: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_719, [25088, 2048]);  mul_719 = None
        permute_63: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
        permute_729: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_153: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1223, permute_729);  permute_729 = None
        permute_730: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1223, [1, 0])
        mm_154: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_730, view_165);  permute_730 = view_165 = None
        sum_287: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1223, [0], True);  view_1223 = None
        view_1224: "f32[2048]" = torch.ops.aten.reshape.default(sum_287, [2048]);  sum_287 = None
        view_1225: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_153, [128, 196, 512]);  mm_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_721: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1225, primals_93);  primals_93 = None
        mul_722: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_721, 512)
        sum_288: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_721, [2], True)
        mul_723: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_721, mul_58);  mul_721 = None
        sum_289: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_723, [2], True);  mul_723 = None
        mul_724: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_58, sum_289);  sum_289 = None
        sub_192: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_722, sum_288);  mul_722 = sum_288 = None
        sub_193: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_192, mul_724);  sub_192 = mul_724 = None
        mul_725: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_109, sub_193);  div_109 = sub_193 = None
        mul_726: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1225, mul_58);  mul_58 = None
        sum_290: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_726, [0, 1]);  mul_726 = None
        sum_291: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1225, [0, 1]);  view_1225 = None
        add_363: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_1219, mul_725);  view_1219 = mul_725 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1226: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_363, [128, 14, 14, 512]);  add_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_8: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_8, torch.float32);  lt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_14: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_8, 0.9782608672976494);  convert_element_type_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_727: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1226, div_14);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_100: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(mul_727, [None, None, fmod_8]);  mul_727 = None
        index_101: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_100, [None, fmod_8]);  index_100 = fmod_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1227: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(index_101, [128, 2, 7, 2, 7, 512]);  index_101 = None
        permute_733: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1227, [0, 1, 3, 2, 4, 5]);  view_1227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_333: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_733, memory_format = torch.contiguous_format);  permute_733 = None
        view_1228: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_333, [512, 7, 7, 512]);  clone_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1229: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_1228, [512, 49, 512]);  view_1228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1230: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_1229, [25088, 512]);  view_1229 = None
        permute_61: "f32[512, 512]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        permute_734: "f32[512, 512]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
        mm_155: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1230, permute_734);  permute_734 = None
        permute_735: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1230, [1, 0])
        mm_156: "f32[512, 512]" = torch.ops.aten.mm.default(permute_735, view_159);  permute_735 = view_159 = None
        sum_292: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1230, [0], True);  view_1230 = None
        view_1231: "f32[512]" = torch.ops.aten.reshape.default(sum_292, [512]);  sum_292 = None
        view_1232: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_155, [512, 49, 512]);  mm_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1233: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1232, [512, 49, 16, 32]);  view_1232 = None
        permute_738: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1233, [0, 2, 1, 3]);  view_1233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_334: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_738, memory_format = torch.contiguous_format);  permute_738 = None
        view_1234: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_334, [8192, 49, 32]);  clone_334 = None
        expand_22: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_13, [512, 16, 49, 49])
        view_155: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_22, [8192, 49, 49]);  expand_22 = None
        permute_739: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
        bmm_120: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_739, view_1234);  permute_739 = None
        bmm_121: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_1234, permute_740);  view_1234 = permute_740 = None
        view_1235: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_120, [512, 16, 49, 32]);  bmm_120 = None
        view_1236: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_121, [512, 16, 49, 49]);  bmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_728: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1236, div_13);  view_1236 = None
        sum_293: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_728, [-1], True)
        neg_18: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_18: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_18, sum_293, mul_728);  neg_18 = sum_293 = mul_728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_294: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_18, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_18: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_294, 0);  sum_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_741: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_18, [1, 2, 0]);  squeeze_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1239: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_741, [2401, 16]);  permute_741 = None
        view_151: "i64[2401]" = torch.ops.aten.reshape.default(primals_90, [-1]);  primals_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_18: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_151], view_1239, True);  view_151 = view_1239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1240: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_18, [8192, 49, 49]);  fma_18 = None
        bmm_122: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_742, view_1240);  permute_742 = None
        bmm_123: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_1240, permute_743);  view_1240 = permute_743 = None
        view_1241: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_122, [512, 16, 32, 49]);  bmm_122 = None
        view_1242: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_123, [512, 16, 49, 32]);  bmm_123 = None
        permute_744: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1241, [0, 1, 3, 2]);  view_1241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_729: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1242, 0.1767766952966369);  view_1242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_18: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_729, permute_744, view_1235]);  mul_729 = permute_744 = view_1235 = None
        view_1243: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_18, [3, 512, 16, 49, 32]);  cat_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_745: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1243, [1, 3, 0, 2, 4]);  view_1243 = None
        clone_335: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_745, memory_format = torch.contiguous_format);  permute_745 = None
        view_1244: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_335, [512, 49, 1536]);  clone_335 = None
        view_1245: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_1244, [25088, 1536]);  view_1244 = None
        permute_56: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        permute_746: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        mm_157: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1245, permute_746);  permute_746 = None
        permute_747: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1245, [1, 0])
        mm_158: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_747, view_145);  permute_747 = view_145 = None
        sum_295: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1245, [0], True);  view_1245 = None
        view_1246: "f32[1536]" = torch.ops.aten.reshape.default(sum_295, [1536]);  sum_295 = None
        view_1247: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_157, [512, 49, 512]);  mm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1248: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1247, [512, 7, 7, 512]);  view_1247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1249: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1248, [128, 2, 2, 7, 7, 512]);  view_1248 = None
        permute_750: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1249, [0, 1, 3, 2, 4, 5]);  view_1249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_336: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_750, memory_format = torch.contiguous_format);  permute_750 = None
        view_1250: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_336, [128, 14, 14, 512]);  clone_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_102: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(view_1250, [None, None, fmod_10]);  view_1250 = None
        index_103: "f32[128, 14, 14, 512]" = torch.ops.aten.index.Tensor(index_102, [None, fmod_10]);  index_102 = fmod_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_731: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_103, primals_84);  primals_84 = None
        mul_732: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_731, 512)
        sum_296: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_731, [3], True)
        mul_733: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_731, mul_54);  mul_731 = None
        sum_297: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_733, [3], True);  mul_733 = None
        mul_734: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_54, sum_297);  sum_297 = None
        sub_195: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_732, sum_296);  mul_732 = sum_296 = None
        sub_196: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_195, mul_734);  sub_195 = mul_734 = None
        mul_735: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_110, sub_196);  div_110 = sub_196 = None
        mul_736: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_103, mul_54);  mul_54 = None
        sum_298: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_736, [0, 1, 2]);  mul_736 = None
        sum_299: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_103, [0, 1, 2]);  index_103 = None
        add_368: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1226, mul_735);  view_1226 = mul_735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1251: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(add_368, [128, 196, 512]);  add_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_7: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_7, torch.float32);  lt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_12: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_7, 0.9826086945831776);  convert_element_type_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_737: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1251, div_12);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1252: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_737, [25088, 512]);  mul_737 = None
        permute_54: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
        permute_751: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_159: "f32[25088, 2048]" = torch.ops.aten.mm.default(view_1252, permute_751);  permute_751 = None
        permute_752: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1252, [1, 0])
        mm_160: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_752, view_139);  permute_752 = view_139 = None
        sum_300: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1252, [0], True);  view_1252 = None
        view_1253: "f32[512]" = torch.ops.aten.reshape.default(sum_300, [512]);  sum_300 = None
        view_1254: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(mm_159, [128, 196, 2048]);  mm_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_138: "f32[128, 196, 2048]" = torch.ops.aten.reshape.default(addmm_18, [128, 196, 2048]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_51: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_138, 0.7071067811865476)
        erf_4: "f32[128, 196, 2048]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_54: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_739: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(add_54, 0.5);  add_54 = None
        mul_740: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_138, view_138)
        mul_741: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(mul_740, -0.5);  mul_740 = None
        exp_43: "f32[128, 196, 2048]" = torch.ops.aten.exp.default(mul_741);  mul_741 = None
        mul_742: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
        mul_743: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_138, mul_742);  view_138 = mul_742 = None
        add_370: "f32[128, 196, 2048]" = torch.ops.aten.add.Tensor(mul_739, mul_743);  mul_739 = mul_743 = None
        mul_744: "f32[128, 196, 2048]" = torch.ops.aten.mul.Tensor(view_1254, add_370);  view_1254 = add_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1255: "f32[25088, 2048]" = torch.ops.aten.reshape.default(mul_744, [25088, 2048]);  mul_744 = None
        permute_53: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        permute_755: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_161: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1255, permute_755);  permute_755 = None
        permute_756: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1255, [1, 0])
        mm_162: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_756, view_137);  permute_756 = view_137 = None
        sum_301: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1255, [0], True);  view_1255 = None
        view_1256: "f32[2048]" = torch.ops.aten.reshape.default(sum_301, [2048]);  sum_301 = None
        view_1257: "f32[128, 196, 512]" = torch.ops.aten.reshape.default(mm_161, [128, 196, 512]);  mm_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_746: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1257, primals_78);  primals_78 = None
        mul_747: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_746, 512)
        sum_302: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_746, [2], True)
        mul_748: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_746, mul_48);  mul_746 = None
        sum_303: "f32[128, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_748, [2], True);  mul_748 = None
        mul_749: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(mul_48, sum_303);  sum_303 = None
        sub_198: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(mul_747, sum_302);  mul_747 = sum_302 = None
        sub_199: "f32[128, 196, 512]" = torch.ops.aten.sub.Tensor(sub_198, mul_749);  sub_198 = mul_749 = None
        mul_750: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(div_111, sub_199);  div_111 = sub_199 = None
        mul_751: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1257, mul_48);  mul_48 = None
        sum_304: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_751, [0, 1]);  mul_751 = None
        sum_305: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1257, [0, 1]);  view_1257 = None
        add_371: "f32[128, 196, 512]" = torch.ops.aten.add.Tensor(view_1251, mul_750);  view_1251 = mul_750 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1258: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(add_371, [128, 14, 14, 512]);  add_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_6: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_6, torch.float32);  lt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_11: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_6, 0.9826086945831776);  convert_element_type_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_752: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1258, div_11);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1259: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.reshape.default(mul_752, [128, 2, 7, 2, 7, 512]);  mul_752 = None
        permute_759: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.permute.default(view_1259, [0, 1, 3, 2, 4, 5]);  view_1259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_337: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.clone.default(permute_759, memory_format = torch.contiguous_format);  permute_759 = None
        view_1260: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(clone_337, [512, 7, 7, 512]);  clone_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1261: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(view_1260, [512, 49, 512]);  view_1260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1262: "f32[25088, 512]" = torch.ops.aten.reshape.default(view_1261, [25088, 512]);  view_1261 = None
        permute_51: "f32[512, 512]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        permute_760: "f32[512, 512]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
        mm_163: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1262, permute_760);  permute_760 = None
        permute_761: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1262, [1, 0])
        mm_164: "f32[512, 512]" = torch.ops.aten.mm.default(permute_761, view_131);  permute_761 = view_131 = None
        sum_306: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1262, [0], True);  view_1262 = None
        view_1263: "f32[512]" = torch.ops.aten.reshape.default(sum_306, [512]);  sum_306 = None
        view_1264: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_163, [512, 49, 512]);  mm_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1265: "f32[512, 49, 16, 32]" = torch.ops.aten.reshape.default(view_1264, [512, 49, 16, 32]);  view_1264 = None
        permute_764: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1265, [0, 2, 1, 3]);  view_1265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_338: "f32[512, 16, 49, 32]" = torch.ops.aten.clone.default(permute_764, memory_format = torch.contiguous_format);  permute_764 = None
        view_1266: "f32[8192, 49, 32]" = torch.ops.aten.reshape.default(clone_338, [8192, 49, 32]);  clone_338 = None
        expand_18: "f32[512, 16, 49, 49]" = torch.ops.aten.expand.default(div_10, [512, 16, 49, 49])
        view_127: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(expand_18, [8192, 49, 49]);  expand_18 = None
        permute_765: "f32[8192, 49, 49]" = torch.ops.aten.permute.default(view_127, [0, 2, 1]);  view_127 = None
        bmm_124: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(permute_765, view_1266);  permute_765 = None
        bmm_125: "f32[8192, 49, 49]" = torch.ops.aten.bmm.default(view_1266, permute_766);  view_1266 = permute_766 = None
        view_1267: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_124, [512, 16, 49, 32]);  bmm_124 = None
        view_1268: "f32[512, 16, 49, 49]" = torch.ops.aten.reshape.default(bmm_125, [512, 16, 49, 49]);  bmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_753: "f32[512, 16, 49, 49]" = torch.ops.aten.mul.Tensor(view_1268, div_10);  view_1268 = None
        sum_307: "f32[512, 16, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_753, [-1], True)
        neg_19: "f32[512, 16, 49, 49]" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_19: "f32[512, 16, 49, 49]" = torch.ops.prims.fma.default(neg_19, sum_307, mul_753);  neg_19 = sum_307 = mul_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_308: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_19, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_19: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_308, 0);  sum_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_767: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_19, [1, 2, 0]);  squeeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1269: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_767, [2401, 16]);  permute_767 = None
        view_125: "i64[2401]" = torch.ops.aten.reshape.default(primals_75, [-1]);  primals_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_19: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_2, [view_125], view_1269, True);  full_default_2 = view_125 = view_1269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1270: "f32[8192, 49, 49]" = torch.ops.aten.reshape.default(fma_19, [8192, 49, 49]);  fma_19 = None
        bmm_126: "f32[8192, 32, 49]" = torch.ops.aten.bmm.default(permute_768, view_1270);  permute_768 = None
        bmm_127: "f32[8192, 49, 32]" = torch.ops.aten.bmm.default(view_1270, permute_769);  view_1270 = permute_769 = None
        view_1271: "f32[512, 16, 32, 49]" = torch.ops.aten.reshape.default(bmm_126, [512, 16, 32, 49]);  bmm_126 = None
        view_1272: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_127, [512, 16, 49, 32]);  bmm_127 = None
        permute_770: "f32[512, 16, 49, 32]" = torch.ops.aten.permute.default(view_1271, [0, 1, 3, 2]);  view_1271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_754: "f32[512, 16, 49, 32]" = torch.ops.aten.mul.Tensor(view_1272, 0.1767766952966369);  view_1272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_19: "f32[1536, 16, 49, 32]" = torch.ops.aten.cat.default([mul_754, permute_770, view_1267]);  mul_754 = permute_770 = view_1267 = None
        view_1273: "f32[3, 512, 16, 49, 32]" = torch.ops.aten.reshape.default(cat_19, [3, 512, 16, 49, 32]);  cat_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_771: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.permute.default(view_1273, [1, 3, 0, 2, 4]);  view_1273 = None
        clone_339: "f32[512, 49, 3, 16, 32]" = torch.ops.aten.clone.default(permute_771, memory_format = torch.contiguous_format);  permute_771 = None
        view_1274: "f32[512, 49, 1536]" = torch.ops.aten.reshape.default(clone_339, [512, 49, 1536]);  clone_339 = None
        view_1275: "f32[25088, 1536]" = torch.ops.aten.reshape.default(view_1274, [25088, 1536]);  view_1274 = None
        permute_46: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        permute_772: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_165: "f32[25088, 512]" = torch.ops.aten.mm.default(view_1275, permute_772);  permute_772 = None
        permute_773: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1275, [1, 0])
        mm_166: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_773, view_119);  permute_773 = view_119 = None
        sum_309: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1275, [0], True);  view_1275 = None
        view_1276: "f32[1536]" = torch.ops.aten.reshape.default(sum_309, [1536]);  sum_309 = None
        view_1277: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(mm_165, [512, 49, 512]);  mm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1278: "f32[512, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1277, [512, 7, 7, 512]);  view_1277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1279: "f32[128, 2, 2, 7, 7, 512]" = torch.ops.aten.reshape.default(view_1278, [128, 2, 2, 7, 7, 512]);  view_1278 = None
        permute_776: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.permute.default(view_1279, [0, 1, 3, 2, 4, 5]);  view_1279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_340: "f32[128, 2, 7, 2, 7, 512]" = torch.ops.aten.clone.default(permute_776, memory_format = torch.contiguous_format);  permute_776 = None
        view_1280: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(clone_340, [128, 14, 14, 512]);  clone_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_756: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1280, primals_70);  primals_70 = None
        mul_757: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_756, 512)
        sum_310: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_756, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        view_115: "f32[128, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_1, [128, 14, 14, 512]);  mm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        sub_15: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(view_115, getitem_35);  view_115 = getitem_35 = None
        mul_44: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = None
        mul_758: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_756, mul_44);  mul_756 = None
        sum_311: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_758, [3], True);  mul_758 = None
        mul_759: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_44, sum_311);  sum_311 = None
        sub_201: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_757, sum_310);  mul_757 = sum_310 = None
        sub_202: "f32[128, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_201, mul_759);  sub_201 = mul_759 = None
        div_112: "f32[128, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 512);  rsqrt_11 = None
        mul_760: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_112, sub_202);  div_112 = sub_202 = None
        mul_761: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1280, mul_44);  mul_44 = None
        sum_312: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_761, [0, 1, 2]);  mul_761 = None
        sum_313: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1280, [0, 1, 2]);  view_1280 = None
        add_372: "f32[128, 14, 14, 512]" = torch.ops.aten.add.Tensor(view_1258, mul_760);  view_1258 = mul_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        view_1281: "f32[25088, 512]" = torch.ops.aten.reshape.default(add_372, [25088, 512]);  add_372 = None
        permute_777: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1281, [1, 0])
        mm_167: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_777, view_114);  permute_777 = view_114 = None
        permute_44: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        permute_779: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_168: "f32[25088, 1024]" = torch.ops.aten.mm.default(view_1281, permute_779);  view_1281 = permute_779 = None
        view_1282: "f32[128, 14, 14, 1024]" = torch.ops.aten.reshape.default(mm_168, [128, 14, 14, 1024]);  mm_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        mul_763: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(view_1282, primals_67);  primals_67 = None
        mul_764: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(mul_763, 1024)
        sum_314: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_763, [3], True)
        mul_765: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(mul_763, mul_42);  mul_763 = None
        sum_315: "f32[128, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_765, [3], True);  mul_765 = None
        mul_766: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(mul_42, sum_315);  sum_315 = None
        sub_204: "f32[128, 14, 14, 1024]" = torch.ops.aten.sub.Tensor(mul_764, sum_314);  mul_764 = sum_314 = None
        sub_205: "f32[128, 14, 14, 1024]" = torch.ops.aten.sub.Tensor(sub_204, mul_766);  sub_204 = mul_766 = None
        mul_767: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(div_113, sub_205);  div_113 = sub_205 = None
        mul_768: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(view_1282, mul_42);  mul_42 = None
        sum_316: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_768, [0, 1, 2]);  mul_768 = None
        sum_317: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1282, [0, 1, 2]);  view_1282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_1283: "f32[128, 14, 14, 2, 2, 256]" = torch.ops.aten.reshape.default(mul_767, [128, 14, 14, 2, 2, 256]);  mul_767 = None
        permute_781: "f32[128, 14, 2, 14, 2, 256]" = torch.ops.aten.permute.default(view_1283, [0, 1, 4, 2, 3, 5]);  view_1283 = None
        clone_341: "f32[128, 14, 2, 14, 2, 256]" = torch.ops.aten.clone.default(permute_781, memory_format = torch.contiguous_format);  permute_781 = None
        view_1284: "f32[128, 28, 28, 256]" = torch.ops.aten.reshape.default(clone_341, [128, 28, 28, 256]);  clone_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1285: "f32[128, 784, 256]" = torch.ops.aten.reshape.default(view_1284, [128, 784, 256]);  view_1284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_5: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_5, torch.float32);  lt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_9: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_5, 0.9869565209373832);  convert_element_type_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_769: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_1285, div_9);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1286: "f32[100352, 256]" = torch.ops.aten.reshape.default(mul_769, [100352, 256]);  mul_769 = None
        permute_42: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
        permute_782: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_169: "f32[100352, 1024]" = torch.ops.aten.mm.default(view_1286, permute_782);  permute_782 = None
        permute_783: "f32[256, 100352]" = torch.ops.aten.permute.default(view_1286, [1, 0])
        mm_170: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_783, view_109);  permute_783 = view_109 = None
        sum_318: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1286, [0], True);  view_1286 = None
        view_1287: "f32[256]" = torch.ops.aten.reshape.default(sum_318, [256]);  sum_318 = None
        view_1288: "f32[128, 784, 1024]" = torch.ops.aten.reshape.default(mm_169, [128, 784, 1024]);  mm_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_108: "f32[128, 784, 1024]" = torch.ops.aten.reshape.default(addmm_14, [128, 784, 1024]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_39: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(view_108, 0.7071067811865476)
        erf_3: "f32[128, 784, 1024]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_44: "f32[128, 784, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_771: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(add_44, 0.5);  add_44 = None
        mul_772: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(view_108, view_108)
        mul_773: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_772, -0.5);  mul_772 = None
        exp_44: "f32[128, 784, 1024]" = torch.ops.aten.exp.default(mul_773);  mul_773 = None
        mul_774: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
        mul_775: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(view_108, mul_774);  view_108 = mul_774 = None
        add_374: "f32[128, 784, 1024]" = torch.ops.aten.add.Tensor(mul_771, mul_775);  mul_771 = mul_775 = None
        mul_776: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(view_1288, add_374);  view_1288 = add_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1289: "f32[100352, 1024]" = torch.ops.aten.reshape.default(mul_776, [100352, 1024]);  mul_776 = None
        permute_41: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        permute_786: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_171: "f32[100352, 256]" = torch.ops.aten.mm.default(view_1289, permute_786);  permute_786 = None
        permute_787: "f32[1024, 100352]" = torch.ops.aten.permute.default(view_1289, [1, 0])
        mm_172: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_787, view_107);  permute_787 = view_107 = None
        sum_319: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1289, [0], True);  view_1289 = None
        view_1290: "f32[1024]" = torch.ops.aten.reshape.default(sum_319, [1024]);  sum_319 = None
        view_1291: "f32[128, 784, 256]" = torch.ops.aten.reshape.default(mm_171, [128, 784, 256]);  mm_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_778: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_1291, primals_61);  primals_61 = None
        mul_779: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_778, 256)
        sum_320: "f32[128, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_778, [2], True)
        mul_780: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_778, mul_36);  mul_778 = None
        sum_321: "f32[128, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_780, [2], True);  mul_780 = None
        mul_781: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_36, sum_321);  sum_321 = None
        sub_207: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(mul_779, sum_320);  mul_779 = sum_320 = None
        sub_208: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(sub_207, mul_781);  sub_207 = mul_781 = None
        mul_782: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(div_114, sub_208);  div_114 = sub_208 = None
        mul_783: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_1291, mul_36);  mul_36 = None
        sum_322: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_783, [0, 1]);  mul_783 = None
        sum_323: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_1291, [0, 1]);  view_1291 = None
        add_375: "f32[128, 784, 256]" = torch.ops.aten.add.Tensor(view_1285, mul_782);  view_1285 = mul_782 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1292: "f32[128, 28, 28, 256]" = torch.ops.aten.reshape.default(add_375, [128, 28, 28, 256]);  add_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_4: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_4, torch.float32);  lt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_8: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_4, 0.9869565209373832);  convert_element_type_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_784: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_1292, div_8);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_4: "i64[28]" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_35: "i64[28]" = torch.ops.aten.add.Tensor(iota_4, 3);  iota_4 = None
        fmod_4: "i64[28]" = torch.ops.aten.fmod.Scalar(add_35, 28);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_104: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(mul_784, [None, None, fmod_4]);  mul_784 = None
        index_105: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(index_104, [None, fmod_4]);  index_104 = fmod_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1293: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.reshape.default(index_105, [128, 4, 7, 4, 7, 256]);  index_105 = None
        permute_790: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_1293, [0, 1, 3, 2, 4, 5]);  view_1293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_342: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_790, memory_format = torch.contiguous_format);  permute_790 = None
        view_1294: "f32[2048, 7, 7, 256]" = torch.ops.aten.reshape.default(clone_342, [2048, 7, 7, 256]);  clone_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1295: "f32[2048, 49, 256]" = torch.ops.aten.reshape.default(view_1294, [2048, 49, 256]);  view_1294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1296: "f32[100352, 256]" = torch.ops.aten.reshape.default(view_1295, [100352, 256]);  view_1295 = None
        permute_39: "f32[256, 256]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        permute_791: "f32[256, 256]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        mm_173: "f32[100352, 256]" = torch.ops.aten.mm.default(view_1296, permute_791);  permute_791 = None
        permute_792: "f32[256, 100352]" = torch.ops.aten.permute.default(view_1296, [1, 0])
        mm_174: "f32[256, 256]" = torch.ops.aten.mm.default(permute_792, view_101);  permute_792 = view_101 = None
        sum_324: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1296, [0], True);  view_1296 = None
        view_1297: "f32[256]" = torch.ops.aten.reshape.default(sum_324, [256]);  sum_324 = None
        view_1298: "f32[2048, 49, 256]" = torch.ops.aten.reshape.default(mm_173, [2048, 49, 256]);  mm_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1299: "f32[2048, 49, 8, 32]" = torch.ops.aten.reshape.default(view_1298, [2048, 49, 8, 32]);  view_1298 = None
        permute_795: "f32[2048, 8, 49, 32]" = torch.ops.aten.permute.default(view_1299, [0, 2, 1, 3]);  view_1299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_343: "f32[2048, 8, 49, 32]" = torch.ops.aten.clone.default(permute_795, memory_format = torch.contiguous_format);  permute_795 = None
        view_1300: "f32[16384, 49, 32]" = torch.ops.aten.reshape.default(clone_343, [16384, 49, 32]);  clone_343 = None
        expand_14: "f32[2048, 8, 49, 49]" = torch.ops.aten.expand.default(div_7, [2048, 8, 49, 49])
        view_97: "f32[16384, 49, 49]" = torch.ops.aten.reshape.default(expand_14, [16384, 49, 49]);  expand_14 = None
        permute_796: "f32[16384, 49, 49]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        bmm_128: "f32[16384, 49, 32]" = torch.ops.aten.bmm.default(permute_796, view_1300);  permute_796 = None
        bmm_129: "f32[16384, 49, 49]" = torch.ops.aten.bmm.default(view_1300, permute_797);  view_1300 = permute_797 = None
        view_1301: "f32[2048, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_128, [2048, 8, 49, 32]);  bmm_128 = None
        view_1302: "f32[2048, 8, 49, 49]" = torch.ops.aten.reshape.default(bmm_129, [2048, 8, 49, 49]);  bmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_785: "f32[2048, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_1302, div_7);  view_1302 = None
        sum_325: "f32[2048, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_785, [-1], True)
        neg_20: "f32[2048, 8, 49, 49]" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_20: "f32[2048, 8, 49, 49]" = torch.ops.prims.fma.default(neg_20, sum_325, mul_785);  neg_20 = sum_325 = mul_785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_326: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_20, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_20: "f32[8, 49, 49]" = torch.ops.aten.squeeze.dim(sum_326, 0);  sum_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_798: "f32[49, 49, 8]" = torch.ops.aten.permute.default(squeeze_20, [1, 2, 0]);  squeeze_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1305: "f32[2401, 8]" = torch.ops.aten.reshape.default(permute_798, [2401, 8]);  permute_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        full_default_20: "f32[169, 8]" = torch.ops.aten.full.default([169, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_93: "i64[2401]" = torch.ops.aten.reshape.default(primals_58, [-1]);  primals_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_20: "f32[169, 8]" = torch.ops.aten.index_put.default(full_default_20, [view_93], view_1305, True);  view_93 = view_1305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1306: "f32[16384, 49, 49]" = torch.ops.aten.reshape.default(fma_20, [16384, 49, 49]);  fma_20 = None
        bmm_130: "f32[16384, 32, 49]" = torch.ops.aten.bmm.default(permute_799, view_1306);  permute_799 = None
        bmm_131: "f32[16384, 49, 32]" = torch.ops.aten.bmm.default(view_1306, permute_800);  view_1306 = permute_800 = None
        view_1307: "f32[2048, 8, 32, 49]" = torch.ops.aten.reshape.default(bmm_130, [2048, 8, 32, 49]);  bmm_130 = None
        view_1308: "f32[2048, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_131, [2048, 8, 49, 32]);  bmm_131 = None
        permute_801: "f32[2048, 8, 49, 32]" = torch.ops.aten.permute.default(view_1307, [0, 1, 3, 2]);  view_1307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_786: "f32[2048, 8, 49, 32]" = torch.ops.aten.mul.Tensor(view_1308, 0.1767766952966369);  view_1308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_20: "f32[6144, 8, 49, 32]" = torch.ops.aten.cat.default([mul_786, permute_801, view_1301]);  mul_786 = permute_801 = view_1301 = None
        view_1309: "f32[3, 2048, 8, 49, 32]" = torch.ops.aten.reshape.default(cat_20, [3, 2048, 8, 49, 32]);  cat_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_802: "f32[2048, 49, 3, 8, 32]" = torch.ops.aten.permute.default(view_1309, [1, 3, 0, 2, 4]);  view_1309 = None
        clone_344: "f32[2048, 49, 3, 8, 32]" = torch.ops.aten.clone.default(permute_802, memory_format = torch.contiguous_format);  permute_802 = None
        view_1310: "f32[2048, 49, 768]" = torch.ops.aten.reshape.default(clone_344, [2048, 49, 768]);  clone_344 = None
        view_1311: "f32[100352, 768]" = torch.ops.aten.reshape.default(view_1310, [100352, 768]);  view_1310 = None
        permute_34: "f32[256, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        permute_803: "f32[768, 256]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_175: "f32[100352, 256]" = torch.ops.aten.mm.default(view_1311, permute_803);  permute_803 = None
        permute_804: "f32[768, 100352]" = torch.ops.aten.permute.default(view_1311, [1, 0])
        mm_176: "f32[768, 256]" = torch.ops.aten.mm.default(permute_804, view_87);  permute_804 = view_87 = None
        sum_327: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1311, [0], True);  view_1311 = None
        view_1312: "f32[768]" = torch.ops.aten.reshape.default(sum_327, [768]);  sum_327 = None
        view_1313: "f32[2048, 49, 256]" = torch.ops.aten.reshape.default(mm_175, [2048, 49, 256]);  mm_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1314: "f32[2048, 7, 7, 256]" = torch.ops.aten.reshape.default(view_1313, [2048, 7, 7, 256]);  view_1313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1315: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.reshape.default(view_1314, [128, 4, 4, 7, 7, 256]);  view_1314 = None
        permute_807: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_1315, [0, 1, 3, 2, 4, 5]);  view_1315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_345: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_807, memory_format = torch.contiguous_format);  permute_807 = None
        view_1316: "f32[128, 28, 28, 256]" = torch.ops.aten.reshape.default(clone_345, [128, 28, 28, 256]);  clone_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_106: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(view_1316, [None, None, fmod_6]);  view_1316 = None
        index_107: "f32[128, 28, 28, 256]" = torch.ops.aten.index.Tensor(index_106, [None, fmod_6]);  index_106 = fmod_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_788: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(index_107, primals_52);  primals_52 = None
        mul_789: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_788, 256)
        sum_328: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_788, [3], True)
        mul_790: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_788, mul_32);  mul_788 = None
        sum_329: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_790, [3], True);  mul_790 = None
        mul_791: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_32, sum_329);  sum_329 = None
        sub_210: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(mul_789, sum_328);  mul_789 = sum_328 = None
        sub_211: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(sub_210, mul_791);  sub_210 = mul_791 = None
        mul_792: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(div_115, sub_211);  div_115 = sub_211 = None
        mul_793: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(index_107, mul_32);  mul_32 = None
        sum_330: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_793, [0, 1, 2]);  mul_793 = None
        sum_331: "f32[256]" = torch.ops.aten.sum.dim_IntList(index_107, [0, 1, 2]);  index_107 = None
        add_380: "f32[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_1292, mul_792);  view_1292 = mul_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1317: "f32[128, 784, 256]" = torch.ops.aten.reshape.default(add_380, [128, 784, 256]);  add_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_3: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_3, torch.float32);  lt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_6: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_3, 0.9913043472915888);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_794: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_1317, div_6);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1318: "f32[100352, 256]" = torch.ops.aten.reshape.default(mul_794, [100352, 256]);  mul_794 = None
        permute_32: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        permute_808: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_177: "f32[100352, 1024]" = torch.ops.aten.mm.default(view_1318, permute_808);  permute_808 = None
        permute_809: "f32[256, 100352]" = torch.ops.aten.permute.default(view_1318, [1, 0])
        mm_178: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_809, view_81);  permute_809 = view_81 = None
        sum_332: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1318, [0], True);  view_1318 = None
        view_1319: "f32[256]" = torch.ops.aten.reshape.default(sum_332, [256]);  sum_332 = None
        view_1320: "f32[128, 784, 1024]" = torch.ops.aten.reshape.default(mm_177, [128, 784, 1024]);  mm_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_80: "f32[128, 784, 1024]" = torch.ops.aten.reshape.default(addmm_10, [128, 784, 1024]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_29: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(view_80, 0.7071067811865476)
        erf_2: "f32[128, 784, 1024]" = torch.ops.aten.erf.default(mul_29);  mul_29 = None
        add_31: "f32[128, 784, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_796: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(add_31, 0.5);  add_31 = None
        mul_797: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(view_80, view_80)
        mul_798: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(mul_797, -0.5);  mul_797 = None
        exp_45: "f32[128, 784, 1024]" = torch.ops.aten.exp.default(mul_798);  mul_798 = None
        mul_799: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
        mul_800: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(view_80, mul_799);  view_80 = mul_799 = None
        add_382: "f32[128, 784, 1024]" = torch.ops.aten.add.Tensor(mul_796, mul_800);  mul_796 = mul_800 = None
        mul_801: "f32[128, 784, 1024]" = torch.ops.aten.mul.Tensor(view_1320, add_382);  view_1320 = add_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1321: "f32[100352, 1024]" = torch.ops.aten.reshape.default(mul_801, [100352, 1024]);  mul_801 = None
        permute_31: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        permute_812: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_179: "f32[100352, 256]" = torch.ops.aten.mm.default(view_1321, permute_812);  permute_812 = None
        permute_813: "f32[1024, 100352]" = torch.ops.aten.permute.default(view_1321, [1, 0])
        mm_180: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_813, view_79);  permute_813 = view_79 = None
        sum_333: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1321, [0], True);  view_1321 = None
        view_1322: "f32[1024]" = torch.ops.aten.reshape.default(sum_333, [1024]);  sum_333 = None
        view_1323: "f32[128, 784, 256]" = torch.ops.aten.reshape.default(mm_179, [128, 784, 256]);  mm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_803: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_1323, primals_46);  primals_46 = None
        mul_804: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_803, 256)
        sum_334: "f32[128, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_803, [2], True)
        mul_805: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_803, mul_26);  mul_803 = None
        sum_335: "f32[128, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_805, [2], True);  mul_805 = None
        mul_806: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(mul_26, sum_335);  sum_335 = None
        sub_213: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(mul_804, sum_334);  mul_804 = sum_334 = None
        sub_214: "f32[128, 784, 256]" = torch.ops.aten.sub.Tensor(sub_213, mul_806);  sub_213 = mul_806 = None
        mul_807: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(div_116, sub_214);  div_116 = sub_214 = None
        mul_808: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_1323, mul_26);  mul_26 = None
        sum_336: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_808, [0, 1]);  mul_808 = None
        sum_337: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_1323, [0, 1]);  view_1323 = None
        add_383: "f32[128, 784, 256]" = torch.ops.aten.add.Tensor(view_1317, mul_807);  view_1317 = mul_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1324: "f32[128, 28, 28, 256]" = torch.ops.aten.reshape.default(add_383, [128, 28, 28, 256]);  add_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_2: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_2, torch.float32);  lt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_5: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_2, 0.9913043472915888);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_809: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_1324, div_5);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1325: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.reshape.default(mul_809, [128, 4, 7, 4, 7, 256]);  mul_809 = None
        permute_816: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.permute.default(view_1325, [0, 1, 3, 2, 4, 5]);  view_1325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_346: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.clone.default(permute_816, memory_format = torch.contiguous_format);  permute_816 = None
        view_1326: "f32[2048, 7, 7, 256]" = torch.ops.aten.reshape.default(clone_346, [2048, 7, 7, 256]);  clone_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1327: "f32[2048, 49, 256]" = torch.ops.aten.reshape.default(view_1326, [2048, 49, 256]);  view_1326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1328: "f32[100352, 256]" = torch.ops.aten.reshape.default(view_1327, [100352, 256]);  view_1327 = None
        permute_29: "f32[256, 256]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        permute_817: "f32[256, 256]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        mm_181: "f32[100352, 256]" = torch.ops.aten.mm.default(view_1328, permute_817);  permute_817 = None
        permute_818: "f32[256, 100352]" = torch.ops.aten.permute.default(view_1328, [1, 0])
        mm_182: "f32[256, 256]" = torch.ops.aten.mm.default(permute_818, view_73);  permute_818 = view_73 = None
        sum_338: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1328, [0], True);  view_1328 = None
        view_1329: "f32[256]" = torch.ops.aten.reshape.default(sum_338, [256]);  sum_338 = None
        view_1330: "f32[2048, 49, 256]" = torch.ops.aten.reshape.default(mm_181, [2048, 49, 256]);  mm_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1331: "f32[2048, 49, 8, 32]" = torch.ops.aten.reshape.default(view_1330, [2048, 49, 8, 32]);  view_1330 = None
        permute_821: "f32[2048, 8, 49, 32]" = torch.ops.aten.permute.default(view_1331, [0, 2, 1, 3]);  view_1331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_347: "f32[2048, 8, 49, 32]" = torch.ops.aten.clone.default(permute_821, memory_format = torch.contiguous_format);  permute_821 = None
        view_1332: "f32[16384, 49, 32]" = torch.ops.aten.reshape.default(clone_347, [16384, 49, 32]);  clone_347 = None
        expand_10: "f32[2048, 8, 49, 49]" = torch.ops.aten.expand.default(div_4, [2048, 8, 49, 49])
        view_69: "f32[16384, 49, 49]" = torch.ops.aten.reshape.default(expand_10, [16384, 49, 49]);  expand_10 = None
        permute_822: "f32[16384, 49, 49]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
        bmm_132: "f32[16384, 49, 32]" = torch.ops.aten.bmm.default(permute_822, view_1332);  permute_822 = None
        bmm_133: "f32[16384, 49, 49]" = torch.ops.aten.bmm.default(view_1332, permute_823);  view_1332 = permute_823 = None
        view_1333: "f32[2048, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_132, [2048, 8, 49, 32]);  bmm_132 = None
        view_1334: "f32[2048, 8, 49, 49]" = torch.ops.aten.reshape.default(bmm_133, [2048, 8, 49, 49]);  bmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_810: "f32[2048, 8, 49, 49]" = torch.ops.aten.mul.Tensor(view_1334, div_4);  view_1334 = None
        sum_339: "f32[2048, 8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_810, [-1], True)
        neg_21: "f32[2048, 8, 49, 49]" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_21: "f32[2048, 8, 49, 49]" = torch.ops.prims.fma.default(neg_21, sum_339, mul_810);  neg_21 = sum_339 = mul_810 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_340: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_21, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_21: "f32[8, 49, 49]" = torch.ops.aten.squeeze.dim(sum_340, 0);  sum_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_824: "f32[49, 49, 8]" = torch.ops.aten.permute.default(squeeze_21, [1, 2, 0]);  squeeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1335: "f32[2401, 8]" = torch.ops.aten.reshape.default(permute_824, [2401, 8]);  permute_824 = None
        view_67: "i64[2401]" = torch.ops.aten.reshape.default(primals_43, [-1]);  primals_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_21: "f32[169, 8]" = torch.ops.aten.index_put.default(full_default_20, [view_67], view_1335, True);  full_default_20 = view_67 = view_1335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1336: "f32[16384, 49, 49]" = torch.ops.aten.reshape.default(fma_21, [16384, 49, 49]);  fma_21 = None
        bmm_134: "f32[16384, 32, 49]" = torch.ops.aten.bmm.default(permute_825, view_1336);  permute_825 = None
        bmm_135: "f32[16384, 49, 32]" = torch.ops.aten.bmm.default(view_1336, permute_826);  view_1336 = permute_826 = None
        view_1337: "f32[2048, 8, 32, 49]" = torch.ops.aten.reshape.default(bmm_134, [2048, 8, 32, 49]);  bmm_134 = None
        view_1338: "f32[2048, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_135, [2048, 8, 49, 32]);  bmm_135 = None
        permute_827: "f32[2048, 8, 49, 32]" = torch.ops.aten.permute.default(view_1337, [0, 1, 3, 2]);  view_1337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_811: "f32[2048, 8, 49, 32]" = torch.ops.aten.mul.Tensor(view_1338, 0.1767766952966369);  view_1338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_21: "f32[6144, 8, 49, 32]" = torch.ops.aten.cat.default([mul_811, permute_827, view_1333]);  mul_811 = permute_827 = view_1333 = None
        view_1339: "f32[3, 2048, 8, 49, 32]" = torch.ops.aten.reshape.default(cat_21, [3, 2048, 8, 49, 32]);  cat_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_828: "f32[2048, 49, 3, 8, 32]" = torch.ops.aten.permute.default(view_1339, [1, 3, 0, 2, 4]);  view_1339 = None
        clone_348: "f32[2048, 49, 3, 8, 32]" = torch.ops.aten.clone.default(permute_828, memory_format = torch.contiguous_format);  permute_828 = None
        view_1340: "f32[2048, 49, 768]" = torch.ops.aten.reshape.default(clone_348, [2048, 49, 768]);  clone_348 = None
        view_1341: "f32[100352, 768]" = torch.ops.aten.reshape.default(view_1340, [100352, 768]);  view_1340 = None
        permute_24: "f32[256, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        permute_829: "f32[768, 256]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_183: "f32[100352, 256]" = torch.ops.aten.mm.default(view_1341, permute_829);  permute_829 = None
        permute_830: "f32[768, 100352]" = torch.ops.aten.permute.default(view_1341, [1, 0])
        mm_184: "f32[768, 256]" = torch.ops.aten.mm.default(permute_830, view_61);  permute_830 = view_61 = None
        sum_341: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1341, [0], True);  view_1341 = None
        view_1342: "f32[768]" = torch.ops.aten.reshape.default(sum_341, [768]);  sum_341 = None
        view_1343: "f32[2048, 49, 256]" = torch.ops.aten.reshape.default(mm_183, [2048, 49, 256]);  mm_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1344: "f32[2048, 7, 7, 256]" = torch.ops.aten.reshape.default(view_1343, [2048, 7, 7, 256]);  view_1343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1345: "f32[128, 4, 4, 7, 7, 256]" = torch.ops.aten.reshape.default(view_1344, [128, 4, 4, 7, 7, 256]);  view_1344 = None
        permute_833: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.permute.default(view_1345, [0, 1, 3, 2, 4, 5]);  view_1345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_349: "f32[128, 4, 7, 4, 7, 256]" = torch.ops.aten.clone.default(permute_833, memory_format = torch.contiguous_format);  permute_833 = None
        view_1346: "f32[128, 28, 28, 256]" = torch.ops.aten.reshape.default(clone_349, [128, 28, 28, 256]);  clone_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_813: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_1346, primals_38);  primals_38 = None
        mul_814: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_813, 256)
        sum_342: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_813, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        view_57: "f32[128, 28, 28, 256]" = torch.ops.aten.reshape.default(mm, [128, 28, 28, 256]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        sub_8: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(view_57, getitem_19);  view_57 = getitem_19 = None
        mul_22: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
        mul_815: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_813, mul_22);  mul_813 = None
        sum_343: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_815, [3], True);  mul_815 = None
        mul_816: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_22, sum_343);  sum_343 = None
        sub_216: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(mul_814, sum_342);  mul_814 = sum_342 = None
        sub_217: "f32[128, 28, 28, 256]" = torch.ops.aten.sub.Tensor(sub_216, mul_816);  sub_216 = mul_816 = None
        div_117: "f32[128, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None
        mul_817: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(div_117, sub_217);  div_117 = sub_217 = None
        mul_818: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_1346, mul_22);  mul_22 = None
        sum_344: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_818, [0, 1, 2]);  mul_818 = None
        sum_345: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_1346, [0, 1, 2]);  view_1346 = None
        add_384: "f32[128, 28, 28, 256]" = torch.ops.aten.add.Tensor(view_1324, mul_817);  view_1324 = mul_817 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        view_1347: "f32[100352, 256]" = torch.ops.aten.reshape.default(add_384, [100352, 256]);  add_384 = None
        permute_834: "f32[256, 100352]" = torch.ops.aten.permute.default(view_1347, [1, 0])
        mm_185: "f32[256, 512]" = torch.ops.aten.mm.default(permute_834, view_56);  permute_834 = view_56 = None
        permute_22: "f32[512, 256]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        permute_836: "f32[256, 512]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_186: "f32[100352, 512]" = torch.ops.aten.mm.default(view_1347, permute_836);  view_1347 = permute_836 = None
        view_1348: "f32[128, 28, 28, 512]" = torch.ops.aten.reshape.default(mm_186, [128, 28, 28, 512]);  mm_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        mul_820: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(view_1348, primals_35);  primals_35 = None
        mul_821: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_820, 512)
        sum_346: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_820, [3], True)
        mul_822: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_820, mul_20);  mul_820 = None
        sum_347: "f32[128, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_822, [3], True);  mul_822 = None
        mul_823: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(mul_20, sum_347);  sum_347 = None
        sub_219: "f32[128, 28, 28, 512]" = torch.ops.aten.sub.Tensor(mul_821, sum_346);  mul_821 = sum_346 = None
        sub_220: "f32[128, 28, 28, 512]" = torch.ops.aten.sub.Tensor(sub_219, mul_823);  sub_219 = mul_823 = None
        mul_824: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(div_118, sub_220);  div_118 = sub_220 = None
        mul_825: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(view_1348, mul_20);  mul_20 = None
        sum_348: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_825, [0, 1, 2]);  mul_825 = None
        sum_349: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1348, [0, 1, 2]);  view_1348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_1349: "f32[128, 28, 28, 2, 2, 128]" = torch.ops.aten.reshape.default(mul_824, [128, 28, 28, 2, 2, 128]);  mul_824 = None
        permute_838: "f32[128, 28, 2, 28, 2, 128]" = torch.ops.aten.permute.default(view_1349, [0, 1, 4, 2, 3, 5]);  view_1349 = None
        clone_350: "f32[128, 28, 2, 28, 2, 128]" = torch.ops.aten.clone.default(permute_838, memory_format = torch.contiguous_format);  permute_838 = None
        view_1350: "f32[128, 56, 56, 128]" = torch.ops.aten.reshape.default(clone_350, [128, 56, 56, 128]);  clone_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1351: "f32[128, 3136, 128]" = torch.ops.aten.reshape.default(view_1350, [128, 3136, 128]);  view_1350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_1: "f32[128, 1, 1]" = torch.ops.prims.convert_element_type.default(lt_1, torch.float32);  lt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_3: "f32[128, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type_1, 0.9956521736457944);  convert_element_type_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_826: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(view_1351, div_3);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1352: "f32[401408, 128]" = torch.ops.aten.reshape.default(mul_826, [401408, 128]);  mul_826 = None
        permute_20: "f32[512, 128]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
        permute_839: "f32[128, 512]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_187: "f32[401408, 512]" = torch.ops.aten.mm.default(view_1352, permute_839);  permute_839 = None
        permute_840: "f32[128, 401408]" = torch.ops.aten.permute.default(view_1352, [1, 0])
        mm_188: "f32[128, 512]" = torch.ops.aten.mm.default(permute_840, view_51);  permute_840 = view_51 = None
        sum_350: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1352, [0], True);  view_1352 = None
        view_1353: "f32[128]" = torch.ops.aten.reshape.default(sum_350, [128]);  sum_350 = None
        view_1354: "f32[128, 3136, 512]" = torch.ops.aten.reshape.default(mm_187, [128, 3136, 512]);  mm_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_50: "f32[128, 3136, 512]" = torch.ops.aten.reshape.default(addmm_6, [128, 3136, 512]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_17: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(view_50, 0.7071067811865476)
        erf_1: "f32[128, 3136, 512]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
        add_21: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_828: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(add_21, 0.5);  add_21 = None
        mul_829: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(view_50, view_50)
        mul_830: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_829, -0.5);  mul_829 = None
        exp_46: "f32[128, 3136, 512]" = torch.ops.aten.exp.default(mul_830);  mul_830 = None
        mul_831: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
        mul_832: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(view_50, mul_831);  view_50 = mul_831 = None
        add_386: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(mul_828, mul_832);  mul_828 = mul_832 = None
        mul_833: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(view_1354, add_386);  view_1354 = add_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1355: "f32[401408, 512]" = torch.ops.aten.reshape.default(mul_833, [401408, 512]);  mul_833 = None
        permute_19: "f32[128, 512]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        permute_843: "f32[512, 128]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_189: "f32[401408, 128]" = torch.ops.aten.mm.default(view_1355, permute_843);  permute_843 = None
        permute_844: "f32[512, 401408]" = torch.ops.aten.permute.default(view_1355, [1, 0])
        mm_190: "f32[512, 128]" = torch.ops.aten.mm.default(permute_844, view_49);  permute_844 = view_49 = None
        sum_351: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1355, [0], True);  view_1355 = None
        view_1356: "f32[512]" = torch.ops.aten.reshape.default(sum_351, [512]);  sum_351 = None
        view_1357: "f32[128, 3136, 128]" = torch.ops.aten.reshape.default(mm_189, [128, 3136, 128]);  mm_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_835: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(view_1357, primals_29);  primals_29 = None
        mul_836: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_835, 128)
        sum_352: "f32[128, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_835, [2], True)
        mul_837: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_835, mul_14);  mul_835 = None
        sum_353: "f32[128, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_837, [2], True);  mul_837 = None
        mul_838: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_14, sum_353);  sum_353 = None
        sub_222: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(mul_836, sum_352);  mul_836 = sum_352 = None
        sub_223: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(sub_222, mul_838);  sub_222 = mul_838 = None
        mul_839: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(div_119, sub_223);  div_119 = sub_223 = None
        mul_840: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(view_1357, mul_14);  mul_14 = None
        sum_354: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_840, [0, 1]);  mul_840 = None
        sum_355: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_1357, [0, 1]);  view_1357 = None
        add_387: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(view_1351, mul_839);  view_1351 = mul_839 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1358: "f32[128, 56, 56, 128]" = torch.ops.aten.reshape.default(add_387, [128, 56, 56, 128]);  add_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type: "f32[128, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(lt, torch.float32);  lt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_2: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(convert_element_type, 0.9956521736457944);  convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_841: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_1358, div_2);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota: "i64[56]" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_12: "i64[56]" = torch.ops.aten.add.Tensor(iota, 3);  iota = None
        fmod: "i64[56]" = torch.ops.aten.fmod.Scalar(add_12, 56);  add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_108: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(mul_841, [None, None, fmod]);  mul_841 = None
        index_109: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_108, [None, fmod]);  index_108 = fmod = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1359: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.reshape.default(index_109, [128, 8, 7, 8, 7, 128]);  index_109 = None
        permute_847: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_1359, [0, 1, 3, 2, 4, 5]);  view_1359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_351: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_847, memory_format = torch.contiguous_format);  permute_847 = None
        view_1360: "f32[8192, 7, 7, 128]" = torch.ops.aten.reshape.default(clone_351, [8192, 7, 7, 128]);  clone_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1361: "f32[8192, 49, 128]" = torch.ops.aten.reshape.default(view_1360, [8192, 49, 128]);  view_1360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1362: "f32[401408, 128]" = torch.ops.aten.reshape.default(view_1361, [401408, 128]);  view_1361 = None
        permute_17: "f32[128, 128]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_848: "f32[128, 128]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        mm_191: "f32[401408, 128]" = torch.ops.aten.mm.default(view_1362, permute_848);  permute_848 = None
        permute_849: "f32[128, 401408]" = torch.ops.aten.permute.default(view_1362, [1, 0])
        mm_192: "f32[128, 128]" = torch.ops.aten.mm.default(permute_849, view_43);  permute_849 = view_43 = None
        sum_356: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1362, [0], True);  view_1362 = None
        view_1363: "f32[128]" = torch.ops.aten.reshape.default(sum_356, [128]);  sum_356 = None
        view_1364: "f32[8192, 49, 128]" = torch.ops.aten.reshape.default(mm_191, [8192, 49, 128]);  mm_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1365: "f32[8192, 49, 4, 32]" = torch.ops.aten.reshape.default(view_1364, [8192, 49, 4, 32]);  view_1364 = None
        permute_852: "f32[8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_1365, [0, 2, 1, 3]);  view_1365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_352: "f32[8192, 4, 49, 32]" = torch.ops.aten.clone.default(permute_852, memory_format = torch.contiguous_format);  permute_852 = None
        view_1366: "f32[32768, 49, 32]" = torch.ops.aten.reshape.default(clone_352, [32768, 49, 32]);  clone_352 = None
        expand_6: "f32[8192, 4, 49, 49]" = torch.ops.aten.expand.default(div_1, [8192, 4, 49, 49])
        view_39: "f32[32768, 49, 49]" = torch.ops.aten.reshape.default(expand_6, [32768, 49, 49]);  expand_6 = None
        permute_853: "f32[32768, 49, 49]" = torch.ops.aten.permute.default(view_39, [0, 2, 1]);  view_39 = None
        bmm_136: "f32[32768, 49, 32]" = torch.ops.aten.bmm.default(permute_853, view_1366);  permute_853 = None
        bmm_137: "f32[32768, 49, 49]" = torch.ops.aten.bmm.default(view_1366, permute_854);  view_1366 = permute_854 = None
        view_1367: "f32[8192, 4, 49, 32]" = torch.ops.aten.reshape.default(bmm_136, [8192, 4, 49, 32]);  bmm_136 = None
        view_1368: "f32[8192, 4, 49, 49]" = torch.ops.aten.reshape.default(bmm_137, [8192, 4, 49, 49]);  bmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_842: "f32[8192, 4, 49, 49]" = torch.ops.aten.mul.Tensor(view_1368, div_1);  view_1368 = None
        sum_357: "f32[8192, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_842, [-1], True)
        neg_22: "f32[8192, 4, 49, 49]" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_22: "f32[8192, 4, 49, 49]" = torch.ops.prims.fma.default(neg_22, sum_357, mul_842);  neg_22 = sum_357 = mul_842 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_358: "f32[1, 4, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_22, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_22: "f32[4, 49, 49]" = torch.ops.aten.squeeze.dim(sum_358, 0);  sum_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_855: "f32[49, 49, 4]" = torch.ops.aten.permute.default(squeeze_22, [1, 2, 0]);  squeeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1371: "f32[2401, 4]" = torch.ops.aten.reshape.default(permute_855, [2401, 4]);  permute_855 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        full_default_22: "f32[169, 4]" = torch.ops.aten.full.default([169, 4], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_35: "i64[2401]" = torch.ops.aten.reshape.default(primals_26, [-1]);  primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_22: "f32[169, 4]" = torch.ops.aten.index_put.default(full_default_22, [view_35], view_1371, True);  view_35 = view_1371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1372: "f32[32768, 49, 49]" = torch.ops.aten.reshape.default(fma_22, [32768, 49, 49]);  fma_22 = None
        bmm_138: "f32[32768, 32, 49]" = torch.ops.aten.bmm.default(permute_856, view_1372);  permute_856 = None
        bmm_139: "f32[32768, 49, 32]" = torch.ops.aten.bmm.default(view_1372, permute_857);  view_1372 = permute_857 = None
        view_1373: "f32[8192, 4, 32, 49]" = torch.ops.aten.reshape.default(bmm_138, [8192, 4, 32, 49]);  bmm_138 = None
        view_1374: "f32[8192, 4, 49, 32]" = torch.ops.aten.reshape.default(bmm_139, [8192, 4, 49, 32]);  bmm_139 = None
        permute_858: "f32[8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_1373, [0, 1, 3, 2]);  view_1373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_843: "f32[8192, 4, 49, 32]" = torch.ops.aten.mul.Tensor(view_1374, 0.1767766952966369);  view_1374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_22: "f32[24576, 4, 49, 32]" = torch.ops.aten.cat.default([mul_843, permute_858, view_1367]);  mul_843 = permute_858 = view_1367 = None
        view_1375: "f32[3, 8192, 4, 49, 32]" = torch.ops.aten.reshape.default(cat_22, [3, 8192, 4, 49, 32]);  cat_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_859: "f32[8192, 49, 3, 4, 32]" = torch.ops.aten.permute.default(view_1375, [1, 3, 0, 2, 4]);  view_1375 = None
        clone_353: "f32[8192, 49, 3, 4, 32]" = torch.ops.aten.clone.default(permute_859, memory_format = torch.contiguous_format);  permute_859 = None
        view_1376: "f32[8192, 49, 384]" = torch.ops.aten.reshape.default(clone_353, [8192, 49, 384]);  clone_353 = None
        view_1377: "f32[401408, 384]" = torch.ops.aten.reshape.default(view_1376, [401408, 384]);  view_1376 = None
        permute_12: "f32[128, 384]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_860: "f32[384, 128]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_193: "f32[401408, 128]" = torch.ops.aten.mm.default(view_1377, permute_860);  permute_860 = None
        permute_861: "f32[384, 401408]" = torch.ops.aten.permute.default(view_1377, [1, 0])
        mm_194: "f32[384, 128]" = torch.ops.aten.mm.default(permute_861, view_29);  permute_861 = view_29 = None
        sum_359: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_1377, [0], True);  view_1377 = None
        view_1378: "f32[384]" = torch.ops.aten.reshape.default(sum_359, [384]);  sum_359 = None
        view_1379: "f32[8192, 49, 128]" = torch.ops.aten.reshape.default(mm_193, [8192, 49, 128]);  mm_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1380: "f32[8192, 7, 7, 128]" = torch.ops.aten.reshape.default(view_1379, [8192, 7, 7, 128]);  view_1379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1381: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.reshape.default(view_1380, [128, 8, 8, 7, 7, 128]);  view_1380 = None
        permute_864: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_1381, [0, 1, 3, 2, 4, 5]);  view_1381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_354: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_864, memory_format = torch.contiguous_format);  permute_864 = None
        view_1382: "f32[128, 56, 56, 128]" = torch.ops.aten.reshape.default(clone_354, [128, 56, 56, 128]);  clone_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_110: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(view_1382, [None, None, fmod_2]);  view_1382 = None
        index_111: "f32[128, 56, 56, 128]" = torch.ops.aten.index.Tensor(index_110, [None, fmod_2]);  index_110 = fmod_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_845: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(index_111, primals_20);  primals_20 = None
        mul_846: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_845, 128)
        sum_360: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_845, [3], True)
        mul_847: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_845, mul_10);  mul_845 = None
        sum_361: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_847, [3], True);  mul_847 = None
        mul_848: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_10, sum_361);  sum_361 = None
        sub_225: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_846, sum_360);  mul_846 = sum_360 = None
        sub_226: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_225, mul_848);  sub_225 = mul_848 = None
        mul_849: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_120, sub_226);  div_120 = sub_226 = None
        mul_850: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(index_111, mul_10);  mul_10 = None
        sum_362: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_850, [0, 1, 2]);  mul_850 = None
        sum_363: "f32[128]" = torch.ops.aten.sum.dim_IntList(index_111, [0, 1, 2]);  index_111 = None
        add_392: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_1358, mul_849);  view_1358 = mul_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1383: "f32[128, 3136, 128]" = torch.ops.aten.reshape.default(add_392, [128, 3136, 128]);  add_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1384: "f32[401408, 128]" = torch.ops.aten.reshape.default(view_1383, [401408, 128])
        permute_10: "f32[512, 128]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_865: "f32[128, 512]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_195: "f32[401408, 512]" = torch.ops.aten.mm.default(view_1384, permute_865);  permute_865 = None
        permute_866: "f32[128, 401408]" = torch.ops.aten.permute.default(view_1384, [1, 0])
        mm_196: "f32[128, 512]" = torch.ops.aten.mm.default(permute_866, view_23);  permute_866 = view_23 = None
        sum_364: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1384, [0], True);  view_1384 = None
        view_1385: "f32[128]" = torch.ops.aten.reshape.default(sum_364, [128]);  sum_364 = None
        view_1386: "f32[128, 3136, 512]" = torch.ops.aten.reshape.default(mm_195, [128, 3136, 512]);  mm_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_22: "f32[128, 3136, 512]" = torch.ops.aten.reshape.default(addmm_2, [128, 3136, 512]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_8: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(view_22, 0.7071067811865476)
        erf: "f32[128, 3136, 512]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_8: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_852: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
        mul_853: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(view_22, view_22)
        mul_854: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(mul_853, -0.5);  mul_853 = None
        exp_47: "f32[128, 3136, 512]" = torch.ops.aten.exp.default(mul_854);  mul_854 = None
        mul_855: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
        mul_856: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(view_22, mul_855);  view_22 = mul_855 = None
        add_394: "f32[128, 3136, 512]" = torch.ops.aten.add.Tensor(mul_852, mul_856);  mul_852 = mul_856 = None
        mul_857: "f32[128, 3136, 512]" = torch.ops.aten.mul.Tensor(view_1386, add_394);  view_1386 = add_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1387: "f32[401408, 512]" = torch.ops.aten.reshape.default(mul_857, [401408, 512]);  mul_857 = None
        permute_9: "f32[128, 512]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_869: "f32[512, 128]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_197: "f32[401408, 128]" = torch.ops.aten.mm.default(view_1387, permute_869);  permute_869 = None
        permute_870: "f32[512, 401408]" = torch.ops.aten.permute.default(view_1387, [1, 0])
        mm_198: "f32[512, 128]" = torch.ops.aten.mm.default(permute_870, view_21);  permute_870 = view_21 = None
        sum_365: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1387, [0], True);  view_1387 = None
        view_1388: "f32[512]" = torch.ops.aten.reshape.default(sum_365, [512]);  sum_365 = None
        view_1389: "f32[128, 3136, 128]" = torch.ops.aten.reshape.default(mm_197, [128, 3136, 128]);  mm_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_859: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(view_1389, primals_14);  primals_14 = None
        mul_860: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_859, 128)
        sum_366: "f32[128, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_859, [2], True)
        mul_861: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_859, mul_5);  mul_859 = None
        sum_367: "f32[128, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_861, [2], True);  mul_861 = None
        mul_862: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(mul_5, sum_367);  sum_367 = None
        sub_228: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(mul_860, sum_366);  mul_860 = sum_366 = None
        sub_229: "f32[128, 3136, 128]" = torch.ops.aten.sub.Tensor(sub_228, mul_862);  sub_228 = mul_862 = None
        mul_863: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(div_121, sub_229);  div_121 = sub_229 = None
        mul_864: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(view_1389, mul_5);  mul_5 = None
        sum_368: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_864, [0, 1]);  mul_864 = None
        sum_369: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_1389, [0, 1]);  view_1389 = None
        add_395: "f32[128, 3136, 128]" = torch.ops.aten.add.Tensor(view_1383, mul_863);  view_1383 = mul_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1390: "f32[128, 56, 56, 128]" = torch.ops.aten.reshape.default(add_395, [128, 56, 56, 128]);  add_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1391: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.reshape.default(view_1390, [128, 8, 7, 8, 7, 128])
        permute_873: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_1391, [0, 1, 3, 2, 4, 5]);  view_1391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_355: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_873, memory_format = torch.contiguous_format);  permute_873 = None
        view_1392: "f32[8192, 7, 7, 128]" = torch.ops.aten.reshape.default(clone_355, [8192, 7, 7, 128]);  clone_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1393: "f32[8192, 49, 128]" = torch.ops.aten.reshape.default(view_1392, [8192, 49, 128]);  view_1392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1394: "f32[401408, 128]" = torch.ops.aten.reshape.default(view_1393, [401408, 128]);  view_1393 = None
        permute_7: "f32[128, 128]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_874: "f32[128, 128]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_199: "f32[401408, 128]" = torch.ops.aten.mm.default(view_1394, permute_874);  permute_874 = None
        permute_875: "f32[128, 401408]" = torch.ops.aten.permute.default(view_1394, [1, 0])
        mm_200: "f32[128, 128]" = torch.ops.aten.mm.default(permute_875, view_15);  permute_875 = view_15 = None
        sum_370: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1394, [0], True);  view_1394 = None
        view_1395: "f32[128]" = torch.ops.aten.reshape.default(sum_370, [128]);  sum_370 = None
        view_1396: "f32[8192, 49, 128]" = torch.ops.aten.reshape.default(mm_199, [8192, 49, 128]);  mm_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1397: "f32[8192, 49, 4, 32]" = torch.ops.aten.reshape.default(view_1396, [8192, 49, 4, 32]);  view_1396 = None
        permute_878: "f32[8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_1397, [0, 2, 1, 3]);  view_1397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_356: "f32[8192, 4, 49, 32]" = torch.ops.aten.clone.default(permute_878, memory_format = torch.contiguous_format);  permute_878 = None
        view_1398: "f32[32768, 49, 32]" = torch.ops.aten.reshape.default(clone_356, [32768, 49, 32]);  clone_356 = None
        expand_2: "f32[8192, 4, 49, 49]" = torch.ops.aten.expand.default(div, [8192, 4, 49, 49])
        view_11: "f32[32768, 49, 49]" = torch.ops.aten.reshape.default(expand_2, [32768, 49, 49]);  expand_2 = None
        permute_879: "f32[32768, 49, 49]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
        bmm_140: "f32[32768, 49, 32]" = torch.ops.aten.bmm.default(permute_879, view_1398);  permute_879 = None
        bmm_141: "f32[32768, 49, 49]" = torch.ops.aten.bmm.default(view_1398, permute_880);  view_1398 = permute_880 = None
        view_1399: "f32[8192, 4, 49, 32]" = torch.ops.aten.reshape.default(bmm_140, [8192, 4, 49, 32]);  bmm_140 = None
        view_1400: "f32[8192, 4, 49, 49]" = torch.ops.aten.reshape.default(bmm_141, [8192, 4, 49, 49]);  bmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        mul_865: "f32[8192, 4, 49, 49]" = torch.ops.aten.mul.Tensor(view_1400, div);  view_1400 = None
        sum_371: "f32[8192, 4, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_865, [-1], True)
        neg_23: "f32[8192, 4, 49, 49]" = torch.ops.aten.neg.default(div);  div = None
        fma_23: "f32[8192, 4, 49, 49]" = torch.ops.prims.fma.default(neg_23, sum_371, mul_865);  neg_23 = sum_371 = mul_865 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_372: "f32[1, 4, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_23, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_23: "f32[4, 49, 49]" = torch.ops.aten.squeeze.dim(sum_372, 0);  sum_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_881: "f32[49, 49, 4]" = torch.ops.aten.permute.default(squeeze_23, [1, 2, 0]);  squeeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1401: "f32[2401, 4]" = torch.ops.aten.reshape.default(permute_881, [2401, 4]);  permute_881 = None
        view_9: "i64[2401]" = torch.ops.aten.reshape.default(primals_11, [-1]);  primals_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_23: "f32[169, 4]" = torch.ops.aten.index_put.default(full_default_22, [view_9], view_1401, True);  full_default_22 = view_9 = view_1401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1402: "f32[32768, 49, 49]" = torch.ops.aten.reshape.default(fma_23, [32768, 49, 49]);  fma_23 = None
        bmm_142: "f32[32768, 32, 49]" = torch.ops.aten.bmm.default(permute_882, view_1402);  permute_882 = None
        bmm_143: "f32[32768, 49, 32]" = torch.ops.aten.bmm.default(view_1402, permute_883);  view_1402 = permute_883 = None
        view_1403: "f32[8192, 4, 32, 49]" = torch.ops.aten.reshape.default(bmm_142, [8192, 4, 32, 49]);  bmm_142 = None
        view_1404: "f32[8192, 4, 49, 32]" = torch.ops.aten.reshape.default(bmm_143, [8192, 4, 49, 32]);  bmm_143 = None
        permute_884: "f32[8192, 4, 49, 32]" = torch.ops.aten.permute.default(view_1403, [0, 1, 3, 2]);  view_1403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_866: "f32[8192, 4, 49, 32]" = torch.ops.aten.mul.Tensor(view_1404, 0.1767766952966369);  view_1404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_23: "f32[24576, 4, 49, 32]" = torch.ops.aten.cat.default([mul_866, permute_884, view_1399]);  mul_866 = permute_884 = view_1399 = None
        view_1405: "f32[3, 8192, 4, 49, 32]" = torch.ops.aten.reshape.default(cat_23, [3, 8192, 4, 49, 32]);  cat_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_885: "f32[8192, 49, 3, 4, 32]" = torch.ops.aten.permute.default(view_1405, [1, 3, 0, 2, 4]);  view_1405 = None
        clone_357: "f32[8192, 49, 3, 4, 32]" = torch.ops.aten.clone.default(permute_885, memory_format = torch.contiguous_format);  permute_885 = None
        view_1406: "f32[8192, 49, 384]" = torch.ops.aten.reshape.default(clone_357, [8192, 49, 384]);  clone_357 = None
        view_1407: "f32[401408, 384]" = torch.ops.aten.reshape.default(view_1406, [401408, 384]);  view_1406 = None
        permute_2: "f32[128, 384]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_886: "f32[384, 128]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_201: "f32[401408, 128]" = torch.ops.aten.mm.default(view_1407, permute_886);  permute_886 = None
        permute_887: "f32[384, 401408]" = torch.ops.aten.permute.default(view_1407, [1, 0])
        mm_202: "f32[384, 128]" = torch.ops.aten.mm.default(permute_887, view_3);  permute_887 = view_3 = None
        sum_373: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_1407, [0], True);  view_1407 = None
        view_1408: "f32[384]" = torch.ops.aten.reshape.default(sum_373, [384]);  sum_373 = None
        view_1409: "f32[8192, 49, 128]" = torch.ops.aten.reshape.default(mm_201, [8192, 49, 128]);  mm_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1410: "f32[8192, 7, 7, 128]" = torch.ops.aten.reshape.default(view_1409, [8192, 7, 7, 128]);  view_1409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1411: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.reshape.default(view_1410, [128, 8, 8, 7, 7, 128]);  view_1410 = None
        permute_890: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.permute.default(view_1411, [0, 1, 3, 2, 4, 5]);  view_1411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_358: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.clone.default(permute_890, memory_format = torch.contiguous_format);  permute_890 = None
        view_1412: "f32[128, 56, 56, 128]" = torch.ops.aten.reshape.default(clone_358, [128, 56, 56, 128]);  clone_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_868: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_1412, primals_6);  primals_6 = None
        mul_869: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_868, 128)
        sum_374: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_868, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/format.py:68 in nchw_to, code: x = x.permute(0, 2, 3, 1)
        permute: "f32[128, 56, 56, 128]" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        sub: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(permute, getitem_1);  permute = getitem_1 = None
        mul: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, primals_4)
        add_1: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_1, primals_5);  mul_1 = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        sub_1: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(add_1, getitem_3);  add_1 = getitem_3 = None
        mul_2: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_870: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_868, mul_2);  mul_868 = None
        sum_375: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_870, [3], True);  mul_870 = None
        mul_871: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_2, sum_375);  sum_375 = None
        sub_231: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_869, sum_374);  mul_869 = sum_374 = None
        sub_232: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_231, mul_871);  sub_231 = mul_871 = None
        div_122: "f32[128, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
        mul_872: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_122, sub_232);  div_122 = sub_232 = None
        mul_873: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_1412, mul_2);  mul_2 = None
        sum_376: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_873, [0, 1, 2]);  mul_873 = None
        sum_377: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_1412, [0, 1, 2]);  view_1412 = None
        add_396: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(view_1390, mul_872);  view_1390 = mul_872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        mul_875: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(add_396, primals_4);  primals_4 = None
        mul_876: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_875, 128)
        sum_378: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_875, [3], True)
        mul_877: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_875, mul);  mul_875 = None
        sum_379: "f32[128, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_877, [3], True);  mul_877 = None
        mul_878: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, sum_379);  sum_379 = None
        sub_234: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_876, sum_378);  mul_876 = sum_378 = None
        sub_235: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_234, mul_878);  sub_234 = mul_878 = None
        div_123: "f32[128, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        mul_879: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_123, sub_235);  div_123 = sub_235 = None
        mul_880: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(add_396, mul);  mul = None
        sum_380: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_880, [0, 1, 2]);  mul_880 = None
        sum_381: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_396, [0, 1, 2]);  add_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/format.py:68 in nchw_to, code: x = x.permute(0, 2, 3, 1)
        permute_891: "f32[128, 128, 56, 56]" = torch.ops.aten.permute.default(mul_879, [0, 3, 1, 2]);  mul_879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_382: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_891, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(permute_891, primals_1, primals_2, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  permute_891 = primals_1 = primals_2 = None
        getitem_179: "f32[128, 3, 4, 4]" = convolution_backward[1];  convolution_backward = None
        return (None, getitem_179, sum_382, sum_380, sum_381, sum_376, sum_377, mm_202, view_1408, index_put_23, None, mm_200, view_1395, sum_368, sum_369, mm_198, view_1388, mm_196, view_1385, sum_362, sum_363, None, mm_194, view_1378, index_put_22, None, mm_192, view_1363, sum_354, sum_355, mm_190, view_1356, mm_188, view_1353, sum_348, sum_349, mm_185, sum_344, sum_345, mm_184, view_1342, index_put_21, None, mm_182, view_1329, sum_336, sum_337, mm_180, view_1322, mm_178, view_1319, sum_330, sum_331, None, mm_176, view_1312, index_put_20, None, mm_174, view_1297, sum_322, sum_323, mm_172, view_1290, mm_170, view_1287, sum_316, sum_317, mm_167, sum_312, sum_313, mm_166, view_1276, index_put_19, None, mm_164, view_1263, sum_304, sum_305, mm_162, view_1256, mm_160, view_1253, sum_298, sum_299, None, mm_158, view_1246, index_put_18, None, mm_156, view_1231, sum_290, sum_291, mm_154, view_1224, mm_152, view_1221, sum_284, sum_285, mm_150, view_1214, index_put_17, None, mm_148, view_1201, sum_276, sum_277, mm_146, view_1194, mm_144, view_1191, sum_270, sum_271, None, mm_142, view_1184, index_put_16, None, mm_140, view_1169, sum_262, sum_263, mm_138, view_1162, mm_136, view_1159, sum_256, sum_257, mm_134, view_1152, index_put_15, None, mm_132, view_1139, sum_248, sum_249, mm_130, view_1132, mm_128, view_1129, sum_242, sum_243, None, mm_126, view_1122, index_put_14, None, mm_124, view_1107, sum_234, sum_235, mm_122, view_1100, mm_120, view_1097, sum_228, sum_229, mm_118, view_1090, index_put_13, None, mm_116, view_1077, sum_220, sum_221, mm_114, view_1070, mm_112, view_1067, sum_214, sum_215, None, mm_110, view_1060, index_put_12, None, mm_108, view_1045, sum_206, sum_207, mm_106, view_1038, mm_104, view_1035, sum_200, sum_201, mm_102, view_1028, index_put_11, None, mm_100, view_1015, sum_192, sum_193, mm_98, view_1008, mm_96, view_1005, sum_186, sum_187, None, mm_94, view_998, index_put_10, None, mm_92, view_983, sum_178, sum_179, mm_90, view_976, mm_88, view_973, sum_172, sum_173, mm_86, view_966, index_put_9, None, mm_84, view_953, sum_164, sum_165, mm_82, view_946, mm_80, view_943, sum_158, sum_159, None, mm_78, view_936, index_put_8, None, mm_76, view_921, sum_150, sum_151, mm_74, view_914, mm_72, view_911, sum_144, sum_145, mm_70, view_904, index_put_7, None, mm_68, view_891, sum_136, sum_137, mm_66, view_884, mm_64, view_881, sum_130, sum_131, None, mm_62, view_874, index_put_6, None, mm_60, view_859, sum_122, sum_123, mm_58, view_852, mm_56, view_849, sum_116, sum_117, mm_54, view_842, index_put_5, None, mm_52, view_829, sum_108, sum_109, mm_50, view_822, mm_48, view_819, sum_102, sum_103, None, mm_46, view_812, index_put_4, None, mm_44, view_797, sum_94, sum_95, mm_42, view_790, mm_40, view_787, sum_88, sum_89, mm_38, view_780, index_put_3, None, mm_36, view_767, sum_80, sum_81, mm_34, view_760, mm_32, view_757, sum_74, sum_75, None, mm_30, view_750, index_put_2, None, mm_28, view_735, sum_66, sum_67, mm_26, view_728, mm_24, view_725, sum_60, sum_61, mm_21, sum_56, sum_57, mm_20, view_714, index_put_1, None, mm_18, view_701, sum_48, sum_49, mm_16, view_694, mm_14, view_691, sum_42, sum_43, mm_12, view_684, index_put, None, mm_10, view_671, sum_34, sum_35, mm_8, view_664, mm_6, view_661, sum_28, sum_29, mm_4, view_658)
