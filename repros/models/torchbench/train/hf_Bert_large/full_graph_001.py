class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[4, 512]", primals_2: "i64[1, 512]", primals_4: "f32[30522, 1024]", primals_7: "f32[1024]", primals_9: "f32[1024, 1024]", primals_11: "f32[1024, 1024]", primals_13: "f32[1024, 1024]", primals_15: "f32[1024, 1024]", primals_17: "f32[1024]", primals_19: "f32[3072, 1024]", primals_21: "f32[1024, 3072]", primals_23: "f32[1024]", primals_25: "f32[1024, 1024]", primals_27: "f32[1024, 1024]", primals_29: "f32[1024, 1024]", primals_31: "f32[1024, 1024]", primals_33: "f32[1024]", primals_35: "f32[3072, 1024]", primals_37: "f32[1024, 3072]", primals_39: "f32[1024]", primals_41: "f32[1024, 1024]", primals_43: "f32[1024, 1024]", primals_45: "f32[1024, 1024]", primals_47: "f32[1024, 1024]", primals_49: "f32[1024]", primals_51: "f32[3072, 1024]", primals_53: "f32[1024, 3072]", primals_55: "f32[1024]", primals_57: "f32[1024, 1024]", primals_59: "f32[1024, 1024]", primals_61: "f32[1024, 1024]", primals_63: "f32[1024, 1024]", primals_65: "f32[1024]", primals_67: "f32[3072, 1024]", primals_69: "f32[1024, 3072]", primals_71: "f32[1024]", primals_73: "f32[1024, 1024]", primals_75: "f32[1024, 1024]", primals_77: "f32[1024, 1024]", primals_79: "f32[1024, 1024]", primals_81: "f32[1024]", primals_83: "f32[3072, 1024]", primals_85: "f32[1024, 3072]", primals_87: "f32[1024]", primals_89: "f32[1024, 1024]", primals_91: "f32[1024, 1024]", primals_93: "f32[1024, 1024]", primals_95: "f32[1024, 1024]", primals_97: "f32[1024]", primals_99: "f32[3072, 1024]", primals_101: "f32[1024, 3072]", primals_103: "f32[1024]", primals_105: "f32[1024, 1024]", primals_107: "f32[1024, 1024]", primals_109: "f32[1024, 1024]", primals_111: "f32[1024, 1024]", primals_113: "f32[1024]", primals_115: "f32[3072, 1024]", primals_117: "f32[1024, 3072]", primals_119: "f32[1024]", primals_121: "f32[1024, 1024]", primals_123: "f32[1024, 1024]", primals_125: "f32[1024, 1024]", primals_127: "f32[1024, 1024]", primals_129: "f32[1024]", primals_131: "f32[3072, 1024]", primals_133: "f32[1024, 3072]", primals_135: "f32[1024]", primals_137: "f32[1024, 1024]", primals_139: "f32[1024, 1024]", primals_141: "f32[1024, 1024]", primals_143: "f32[1024, 1024]", primals_145: "f32[1024]", primals_147: "f32[3072, 1024]", primals_149: "f32[1024, 3072]", primals_151: "f32[1024]", primals_153: "f32[1024, 1024]", primals_155: "f32[1024, 1024]", primals_157: "f32[1024, 1024]", primals_159: "f32[1024, 1024]", primals_161: "f32[1024]", primals_163: "f32[3072, 1024]", primals_165: "f32[1024, 3072]", primals_167: "f32[1024]", primals_169: "f32[1024, 1024]", primals_171: "f32[1024, 1024]", primals_173: "f32[1024, 1024]", primals_175: "f32[1024, 1024]", primals_177: "f32[1024]", primals_179: "f32[3072, 1024]", primals_181: "f32[1024, 3072]", primals_183: "f32[1024]", primals_185: "f32[1024, 1024]", primals_187: "f32[1024, 1024]", primals_189: "f32[1024, 1024]", primals_191: "f32[1024, 1024]", primals_193: "f32[1024]", primals_195: "f32[3072, 1024]", primals_197: "f32[1024, 3072]", primals_199: "f32[1024]", primals_201: "f32[1024, 1024]", primals_203: "f32[1024, 1024]", primals_205: "f32[1024, 1024]", primals_207: "f32[1024, 1024]", primals_209: "f32[1024]", primals_211: "f32[3072, 1024]", primals_213: "f32[1024, 3072]", primals_215: "f32[1024]", primals_217: "f32[1024, 1024]", primals_219: "f32[1024, 1024]", primals_221: "f32[1024, 1024]", primals_223: "f32[1024, 1024]", primals_225: "f32[1024]", primals_227: "f32[3072, 1024]", primals_229: "f32[1024, 3072]", primals_231: "f32[1024]", primals_233: "f32[1024, 1024]", primals_235: "f32[1024, 1024]", primals_237: "f32[1024, 1024]", primals_239: "f32[1024, 1024]", primals_241: "f32[1024]", primals_243: "f32[3072, 1024]", primals_245: "f32[1024, 3072]", primals_247: "f32[1024]", primals_249: "f32[1024, 1024]", primals_251: "f32[1024, 1024]", primals_253: "f32[1024, 1024]", primals_255: "f32[1024, 1024]", primals_257: "f32[1024]", primals_259: "f32[3072, 1024]", primals_261: "f32[1024, 3072]", primals_263: "f32[1024]", primals_265: "f32[1024, 1024]", primals_267: "f32[1024, 1024]", primals_269: "f32[1024, 1024]", primals_271: "f32[1024, 1024]", primals_273: "f32[1024]", primals_275: "f32[3072, 1024]", primals_277: "f32[1024, 3072]", primals_279: "f32[1024]", primals_281: "f32[1024, 1024]", primals_283: "f32[1024, 1024]", primals_285: "f32[1024, 1024]", primals_287: "f32[1024, 1024]", primals_289: "f32[1024]", primals_291: "f32[3072, 1024]", primals_293: "f32[1024, 3072]", primals_295: "f32[1024]", primals_297: "f32[1024, 1024]", primals_299: "f32[1024, 1024]", primals_301: "f32[1024, 1024]", primals_303: "f32[1024, 1024]", primals_305: "f32[1024]", primals_307: "f32[3072, 1024]", primals_309: "f32[1024, 3072]", primals_311: "f32[1024]", primals_313: "f32[1024, 1024]", primals_315: "f32[1024, 1024]", primals_317: "f32[1024, 1024]", primals_319: "f32[1024, 1024]", primals_321: "f32[1024]", primals_323: "f32[3072, 1024]", primals_325: "f32[1024, 3072]", primals_327: "f32[1024]", primals_329: "f32[1024, 1024]", primals_331: "f32[1024, 1024]", primals_333: "f32[1024, 1024]", primals_335: "f32[1024, 1024]", primals_337: "f32[1024]", primals_339: "f32[3072, 1024]", primals_341: "f32[1024, 3072]", primals_343: "f32[1024]", primals_345: "f32[1024, 1024]", primals_347: "f32[1024, 1024]", primals_349: "f32[1024, 1024]", primals_351: "f32[1024, 1024]", primals_353: "f32[1024]", primals_355: "f32[3072, 1024]", primals_357: "f32[1024, 3072]", primals_359: "f32[1024]", primals_361: "f32[1024, 1024]", primals_363: "f32[1024, 1024]", primals_365: "f32[1024, 1024]", primals_367: "f32[1024, 1024]", primals_369: "f32[1024]", primals_371: "f32[3072, 1024]", primals_373: "f32[1024, 3072]", primals_375: "f32[1024]", primals_377: "f32[1024, 1024]", primals_379: "f32[1024, 1024]", primals_381: "f32[1024, 1024]", primals_383: "f32[1024, 1024]", primals_385: "f32[1024]", primals_387: "f32[3072, 1024]", primals_389: "f32[1024, 3072]", primals_391: "f32[1024]", primals_393: "f32[1024, 1024]", primals_395: "f32[1024]", primals_398: "i64[4, 512]", gather: "i64[1, 512]", mul: "f32[4, 512, 1024]", gt: "b8[4, 512, 1024]", ge: "b8[1, 1, 512, 1]", view: "f32[2048, 1024]", bmm: "f32[64, 512, 512]", amax: "f32[4, 16, 512, 1]", sum_1: "f32[4, 16, 512, 1]", logical_not_1: "b8[4, 16, 512, 1]", gt_1: "b8[4, 16, 512, 512]", view_16: "f32[2048, 1024]", gt_2: "b8[4, 512, 1024]", mul_10: "f32[4, 512, 1024]", view_18: "f32[2048, 1024]", addmm_4: "f32[2048, 3072]", view_20: "f32[2048, 3072]", gt_3: "b8[4, 512, 1024]", mul_17: "f32[4, 512, 1024]", view_22: "f32[2048, 1024]", where_3: "f32[4, 16, 512, 512]", gt_4: "b8[4, 16, 512, 512]", view_38: "f32[2048, 1024]", gt_5: "b8[4, 512, 1024]", mul_25: "f32[4, 512, 1024]", view_40: "f32[2048, 1024]", addmm_10: "f32[2048, 3072]", view_42: "f32[2048, 3072]", gt_6: "b8[4, 512, 1024]", mul_32: "f32[4, 512, 1024]", view_44: "f32[2048, 1024]", where_5: "f32[4, 16, 512, 512]", gt_7: "b8[4, 16, 512, 512]", view_60: "f32[2048, 1024]", gt_8: "b8[4, 512, 1024]", mul_40: "f32[4, 512, 1024]", view_62: "f32[2048, 1024]", addmm_16: "f32[2048, 3072]", view_64: "f32[2048, 3072]", gt_9: "b8[4, 512, 1024]", mul_47: "f32[4, 512, 1024]", view_66: "f32[2048, 1024]", where_7: "f32[4, 16, 512, 512]", gt_10: "b8[4, 16, 512, 512]", view_82: "f32[2048, 1024]", gt_11: "b8[4, 512, 1024]", mul_55: "f32[4, 512, 1024]", view_84: "f32[2048, 1024]", addmm_22: "f32[2048, 3072]", view_86: "f32[2048, 3072]", gt_12: "b8[4, 512, 1024]", mul_62: "f32[4, 512, 1024]", view_88: "f32[2048, 1024]", where_9: "f32[4, 16, 512, 512]", gt_13: "b8[4, 16, 512, 512]", view_104: "f32[2048, 1024]", gt_14: "b8[4, 512, 1024]", mul_70: "f32[4, 512, 1024]", view_106: "f32[2048, 1024]", addmm_28: "f32[2048, 3072]", view_108: "f32[2048, 3072]", gt_15: "b8[4, 512, 1024]", mul_77: "f32[4, 512, 1024]", view_110: "f32[2048, 1024]", where_11: "f32[4, 16, 512, 512]", gt_16: "b8[4, 16, 512, 512]", view_126: "f32[2048, 1024]", gt_17: "b8[4, 512, 1024]", mul_85: "f32[4, 512, 1024]", view_128: "f32[2048, 1024]", addmm_34: "f32[2048, 3072]", view_130: "f32[2048, 3072]", gt_18: "b8[4, 512, 1024]", mul_92: "f32[4, 512, 1024]", view_132: "f32[2048, 1024]", where_13: "f32[4, 16, 512, 512]", gt_19: "b8[4, 16, 512, 512]", view_148: "f32[2048, 1024]", gt_20: "b8[4, 512, 1024]", mul_100: "f32[4, 512, 1024]", view_150: "f32[2048, 1024]", addmm_40: "f32[2048, 3072]", view_152: "f32[2048, 3072]", gt_21: "b8[4, 512, 1024]", mul_107: "f32[4, 512, 1024]", view_154: "f32[2048, 1024]", where_15: "f32[4, 16, 512, 512]", gt_22: "b8[4, 16, 512, 512]", view_170: "f32[2048, 1024]", gt_23: "b8[4, 512, 1024]", mul_115: "f32[4, 512, 1024]", view_172: "f32[2048, 1024]", addmm_46: "f32[2048, 3072]", view_174: "f32[2048, 3072]", gt_24: "b8[4, 512, 1024]", mul_122: "f32[4, 512, 1024]", view_176: "f32[2048, 1024]", where_17: "f32[4, 16, 512, 512]", gt_25: "b8[4, 16, 512, 512]", view_192: "f32[2048, 1024]", gt_26: "b8[4, 512, 1024]", mul_130: "f32[4, 512, 1024]", view_194: "f32[2048, 1024]", addmm_52: "f32[2048, 3072]", view_196: "f32[2048, 3072]", gt_27: "b8[4, 512, 1024]", mul_137: "f32[4, 512, 1024]", view_198: "f32[2048, 1024]", where_19: "f32[4, 16, 512, 512]", gt_28: "b8[4, 16, 512, 512]", view_214: "f32[2048, 1024]", gt_29: "b8[4, 512, 1024]", mul_145: "f32[4, 512, 1024]", view_216: "f32[2048, 1024]", addmm_58: "f32[2048, 3072]", view_218: "f32[2048, 3072]", gt_30: "b8[4, 512, 1024]", mul_152: "f32[4, 512, 1024]", view_220: "f32[2048, 1024]", where_21: "f32[4, 16, 512, 512]", gt_31: "b8[4, 16, 512, 512]", view_236: "f32[2048, 1024]", gt_32: "b8[4, 512, 1024]", mul_160: "f32[4, 512, 1024]", view_238: "f32[2048, 1024]", addmm_64: "f32[2048, 3072]", view_240: "f32[2048, 3072]", gt_33: "b8[4, 512, 1024]", mul_167: "f32[4, 512, 1024]", view_242: "f32[2048, 1024]", where_23: "f32[4, 16, 512, 512]", gt_34: "b8[4, 16, 512, 512]", view_258: "f32[2048, 1024]", gt_35: "b8[4, 512, 1024]", mul_175: "f32[4, 512, 1024]", view_260: "f32[2048, 1024]", addmm_70: "f32[2048, 3072]", view_262: "f32[2048, 3072]", gt_36: "b8[4, 512, 1024]", mul_182: "f32[4, 512, 1024]", view_264: "f32[2048, 1024]", where_25: "f32[4, 16, 512, 512]", gt_37: "b8[4, 16, 512, 512]", view_280: "f32[2048, 1024]", gt_38: "b8[4, 512, 1024]", mul_190: "f32[4, 512, 1024]", view_282: "f32[2048, 1024]", addmm_76: "f32[2048, 3072]", view_284: "f32[2048, 3072]", gt_39: "b8[4, 512, 1024]", mul_197: "f32[4, 512, 1024]", view_286: "f32[2048, 1024]", where_27: "f32[4, 16, 512, 512]", gt_40: "b8[4, 16, 512, 512]", view_302: "f32[2048, 1024]", gt_41: "b8[4, 512, 1024]", mul_205: "f32[4, 512, 1024]", view_304: "f32[2048, 1024]", addmm_82: "f32[2048, 3072]", view_306: "f32[2048, 3072]", gt_42: "b8[4, 512, 1024]", mul_212: "f32[4, 512, 1024]", view_308: "f32[2048, 1024]", where_29: "f32[4, 16, 512, 512]", gt_43: "b8[4, 16, 512, 512]", view_324: "f32[2048, 1024]", gt_44: "b8[4, 512, 1024]", mul_220: "f32[4, 512, 1024]", view_326: "f32[2048, 1024]", addmm_88: "f32[2048, 3072]", view_328: "f32[2048, 3072]", gt_45: "b8[4, 512, 1024]", mul_227: "f32[4, 512, 1024]", view_330: "f32[2048, 1024]", where_31: "f32[4, 16, 512, 512]", gt_46: "b8[4, 16, 512, 512]", view_346: "f32[2048, 1024]", gt_47: "b8[4, 512, 1024]", mul_235: "f32[4, 512, 1024]", view_348: "f32[2048, 1024]", addmm_94: "f32[2048, 3072]", view_350: "f32[2048, 3072]", gt_48: "b8[4, 512, 1024]", mul_242: "f32[4, 512, 1024]", view_352: "f32[2048, 1024]", where_33: "f32[4, 16, 512, 512]", gt_49: "b8[4, 16, 512, 512]", view_368: "f32[2048, 1024]", gt_50: "b8[4, 512, 1024]", mul_250: "f32[4, 512, 1024]", view_370: "f32[2048, 1024]", addmm_100: "f32[2048, 3072]", view_372: "f32[2048, 3072]", gt_51: "b8[4, 512, 1024]", mul_257: "f32[4, 512, 1024]", view_374: "f32[2048, 1024]", where_35: "f32[4, 16, 512, 512]", gt_52: "b8[4, 16, 512, 512]", view_390: "f32[2048, 1024]", gt_53: "b8[4, 512, 1024]", mul_265: "f32[4, 512, 1024]", view_392: "f32[2048, 1024]", addmm_106: "f32[2048, 3072]", view_394: "f32[2048, 3072]", gt_54: "b8[4, 512, 1024]", mul_272: "f32[4, 512, 1024]", view_396: "f32[2048, 1024]", where_37: "f32[4, 16, 512, 512]", gt_55: "b8[4, 16, 512, 512]", view_412: "f32[2048, 1024]", gt_56: "b8[4, 512, 1024]", mul_280: "f32[4, 512, 1024]", view_414: "f32[2048, 1024]", addmm_112: "f32[2048, 3072]", view_416: "f32[2048, 3072]", gt_57: "b8[4, 512, 1024]", mul_287: "f32[4, 512, 1024]", view_418: "f32[2048, 1024]", where_39: "f32[4, 16, 512, 512]", gt_58: "b8[4, 16, 512, 512]", view_434: "f32[2048, 1024]", gt_59: "b8[4, 512, 1024]", mul_295: "f32[4, 512, 1024]", view_436: "f32[2048, 1024]", addmm_118: "f32[2048, 3072]", view_438: "f32[2048, 3072]", gt_60: "b8[4, 512, 1024]", mul_302: "f32[4, 512, 1024]", view_440: "f32[2048, 1024]", where_41: "f32[4, 16, 512, 512]", gt_61: "b8[4, 16, 512, 512]", view_456: "f32[2048, 1024]", gt_62: "b8[4, 512, 1024]", mul_310: "f32[4, 512, 1024]", view_458: "f32[2048, 1024]", addmm_124: "f32[2048, 3072]", view_460: "f32[2048, 3072]", gt_63: "b8[4, 512, 1024]", mul_317: "f32[4, 512, 1024]", view_462: "f32[2048, 1024]", where_43: "f32[4, 16, 512, 512]", gt_64: "b8[4, 16, 512, 512]", view_478: "f32[2048, 1024]", gt_65: "b8[4, 512, 1024]", mul_325: "f32[4, 512, 1024]", view_480: "f32[2048, 1024]", addmm_130: "f32[2048, 3072]", view_482: "f32[2048, 3072]", gt_66: "b8[4, 512, 1024]", mul_332: "f32[4, 512, 1024]", view_484: "f32[2048, 1024]", where_45: "f32[4, 16, 512, 512]", gt_67: "b8[4, 16, 512, 512]", view_500: "f32[2048, 1024]", gt_68: "b8[4, 512, 1024]", mul_340: "f32[4, 512, 1024]", view_502: "f32[2048, 1024]", addmm_136: "f32[2048, 3072]", view_504: "f32[2048, 3072]", gt_69: "b8[4, 512, 1024]", mul_347: "f32[4, 512, 1024]", view_506: "f32[2048, 1024]", where_47: "f32[4, 16, 512, 512]", gt_70: "b8[4, 16, 512, 512]", view_522: "f32[2048, 1024]", gt_71: "b8[4, 512, 1024]", mul_355: "f32[4, 512, 1024]", view_524: "f32[2048, 1024]", addmm_142: "f32[2048, 3072]", view_526: "f32[2048, 3072]", gt_72: "b8[4, 512, 1024]", mul_362: "f32[4, 512, 1024]", view_528: "f32[2048, 1024]", addmm_144: "f32[2048, 1024]", getitem_99: "f32[4, 512, 1]", rsqrt_49: "f32[4, 512, 1]", view_530: "f32[2048, 1024]", view_531: "f32[4, 512, 30522]", amax_24: "f32[2048, 1]", log: "f32[2048, 1]", convert_element_type: "f32[]", div_27: "f32[4, 512, 1]", div_28: "f32[4, 512, 1]", permute_287: "f32[64, 512, 512]", permute_288: "f32[64, 64, 512]", permute_289: "f32[64, 64, 512]", permute_290: "f32[64, 512, 64]", div_29: "f32[4, 512, 1]", div_30: "f32[4, 512, 1]", permute_320: "f32[64, 512, 512]", permute_321: "f32[64, 64, 512]", permute_322: "f32[64, 64, 512]", permute_323: "f32[64, 512, 64]", div_31: "f32[4, 512, 1]", div_32: "f32[4, 512, 1]", permute_353: "f32[64, 512, 512]", permute_354: "f32[64, 64, 512]", permute_355: "f32[64, 64, 512]", permute_356: "f32[64, 512, 64]", div_33: "f32[4, 512, 1]", div_34: "f32[4, 512, 1]", permute_386: "f32[64, 512, 512]", permute_387: "f32[64, 64, 512]", permute_388: "f32[64, 64, 512]", permute_389: "f32[64, 512, 64]", div_35: "f32[4, 512, 1]", div_36: "f32[4, 512, 1]", permute_419: "f32[64, 512, 512]", permute_420: "f32[64, 64, 512]", permute_421: "f32[64, 64, 512]", permute_422: "f32[64, 512, 64]", div_37: "f32[4, 512, 1]", div_38: "f32[4, 512, 1]", permute_452: "f32[64, 512, 512]", permute_453: "f32[64, 64, 512]", permute_454: "f32[64, 64, 512]", permute_455: "f32[64, 512, 64]", div_39: "f32[4, 512, 1]", div_40: "f32[4, 512, 1]", permute_485: "f32[64, 512, 512]", permute_486: "f32[64, 64, 512]", permute_487: "f32[64, 64, 512]", permute_488: "f32[64, 512, 64]", div_41: "f32[4, 512, 1]", div_42: "f32[4, 512, 1]", permute_518: "f32[64, 512, 512]", permute_519: "f32[64, 64, 512]", permute_520: "f32[64, 64, 512]", permute_521: "f32[64, 512, 64]", div_43: "f32[4, 512, 1]", div_44: "f32[4, 512, 1]", permute_551: "f32[64, 512, 512]", permute_552: "f32[64, 64, 512]", permute_553: "f32[64, 64, 512]", permute_554: "f32[64, 512, 64]", div_45: "f32[4, 512, 1]", div_46: "f32[4, 512, 1]", permute_584: "f32[64, 512, 512]", permute_585: "f32[64, 64, 512]", permute_586: "f32[64, 64, 512]", permute_587: "f32[64, 512, 64]", div_47: "f32[4, 512, 1]", div_48: "f32[4, 512, 1]", permute_617: "f32[64, 512, 512]", permute_618: "f32[64, 64, 512]", permute_619: "f32[64, 64, 512]", permute_620: "f32[64, 512, 64]", div_49: "f32[4, 512, 1]", div_50: "f32[4, 512, 1]", permute_650: "f32[64, 512, 512]", permute_651: "f32[64, 64, 512]", permute_652: "f32[64, 64, 512]", permute_653: "f32[64, 512, 64]", div_51: "f32[4, 512, 1]", div_52: "f32[4, 512, 1]", permute_683: "f32[64, 512, 512]", permute_684: "f32[64, 64, 512]", permute_685: "f32[64, 64, 512]", permute_686: "f32[64, 512, 64]", div_53: "f32[4, 512, 1]", div_54: "f32[4, 512, 1]", permute_716: "f32[64, 512, 512]", permute_717: "f32[64, 64, 512]", permute_718: "f32[64, 64, 512]", permute_719: "f32[64, 512, 64]", div_55: "f32[4, 512, 1]", div_56: "f32[4, 512, 1]", permute_749: "f32[64, 512, 512]", permute_750: "f32[64, 64, 512]", permute_751: "f32[64, 64, 512]", permute_752: "f32[64, 512, 64]", div_57: "f32[4, 512, 1]", div_58: "f32[4, 512, 1]", permute_782: "f32[64, 512, 512]", permute_783: "f32[64, 64, 512]", permute_784: "f32[64, 64, 512]", permute_785: "f32[64, 512, 64]", div_59: "f32[4, 512, 1]", div_60: "f32[4, 512, 1]", permute_815: "f32[64, 512, 512]", permute_816: "f32[64, 64, 512]", permute_817: "f32[64, 64, 512]", permute_818: "f32[64, 512, 64]", div_61: "f32[4, 512, 1]", div_62: "f32[4, 512, 1]", permute_848: "f32[64, 512, 512]", permute_849: "f32[64, 64, 512]", permute_850: "f32[64, 64, 512]", permute_851: "f32[64, 512, 64]", div_63: "f32[4, 512, 1]", div_64: "f32[4, 512, 1]", permute_881: "f32[64, 512, 512]", permute_882: "f32[64, 64, 512]", permute_883: "f32[64, 64, 512]", permute_884: "f32[64, 512, 64]", div_65: "f32[4, 512, 1]", div_66: "f32[4, 512, 1]", permute_914: "f32[64, 512, 512]", permute_915: "f32[64, 64, 512]", permute_916: "f32[64, 64, 512]", permute_917: "f32[64, 512, 64]", div_67: "f32[4, 512, 1]", div_68: "f32[4, 512, 1]", permute_947: "f32[64, 512, 512]", permute_948: "f32[64, 64, 512]", permute_949: "f32[64, 64, 512]", permute_950: "f32[64, 512, 64]", div_69: "f32[4, 512, 1]", div_70: "f32[4, 512, 1]", permute_980: "f32[64, 512, 512]", permute_981: "f32[64, 64, 512]", permute_982: "f32[64, 64, 512]", permute_983: "f32[64, 512, 64]", div_71: "f32[4, 512, 1]", div_72: "f32[4, 512, 1]", permute_1013: "f32[64, 512, 512]", permute_1014: "f32[64, 64, 512]", permute_1015: "f32[64, 64, 512]", permute_1016: "f32[64, 512, 64]", div_73: "f32[4, 512, 1]", div_74: "f32[4, 512, 1]", permute_1046: "f32[64, 512, 512]", permute_1047: "f32[64, 64, 512]", permute_1048: "f32[64, 64, 512]", permute_1049: "f32[64, 512, 64]", div_75: "f32[4, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[4, 512, 30522]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:979 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        div_25: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
        view_533: "i64[2048]" = torch.ops.aten.reshape.default(primals_398, [-1]);  primals_398 = None
        unsqueeze_4: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(view_533, 1);  view_533 = None
        ne_3: "b8[2048, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_4, -100)
        full_default_72: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_50: "i64[2048, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_4, full_default_72);  unsqueeze_4 = full_default_72 = None

        # No stacktrace found for following nodes
        iota_default: "i64[30522]" = torch.ops.prims.iota.default(30522, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 30522]" = torch.ops.aten.reshape.default(iota_default, [1, 30522]);  iota_default = None
        expand_default: "i64[2048, 30522]" = torch.ops.aten.expand.default(where_50, [2048, 30522]);  where_50 = None
        eq_tensor: "b8[2048, 30522]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:979 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        where_self: "f32[2048, 30522]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:979 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        where_51: "f32[2048, 1]" = torch.ops.aten.where.self(ne_3, div_25, full_default_1);  ne_3 = div_25 = None
        mul_369: "f32[2048, 30522]" = torch.ops.aten.mul.Tensor(where_self, where_51);  where_self = where_51 = None
        view_532: "f32[2048, 30522]" = torch.ops.aten.reshape.default(view_531, [-1, 30522]);  view_531 = None
        sub_74: "f32[2048, 30522]" = torch.ops.aten.sub.Tensor(view_532, amax_24);  view_532 = amax_24 = None
        sub_75: "f32[2048, 30522]" = torch.ops.aten.sub.Tensor(sub_74, log);  sub_74 = log = None
        exp_25: "f32[2048, 30522]" = torch.ops.aten.exp.default(sub_75);  sub_75 = None
        sum_28: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [1], True)
        mul_370: "f32[2048, 30522]" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
        sub_76: "f32[2048, 30522]" = torch.ops.aten.sub.Tensor(mul_369, mul_370);  mul_369 = mul_370 = None
        view_534: "f32[4, 512, 30522]" = torch.ops.aten.reshape.default(sub_76, [4, 512, 30522]);  sub_76 = None
        add_201: "f32[4, 512, 30522]" = torch.ops.aten.add.Tensor(tangents_2, view_534);  tangents_2 = view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:499 in forward, code: hidden_states = self.decoder(hidden_states)
        view_535: "f32[2048, 30522]" = torch.ops.aten.reshape.default(add_201, [2048, 30522]);  add_201 = None
        permute_265: "f32[1024, 30522]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_266: "f32[30522, 1024]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        mm: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_535, permute_266);  permute_266 = None
        permute_267: "f32[30522, 2048]" = torch.ops.aten.permute.default(view_535, [1, 0])
        mm_1: "f32[30522, 1024]" = torch.ops.aten.mm.default(permute_267, view_530);  permute_267 = view_530 = None
        sum_29: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
        view_536: "f32[30522]" = torch.ops.aten.reshape.default(sum_29, [30522]);  sum_29 = None
        view_537: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm, [4, 512, 1024]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:483 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_372: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(view_537, primals_395);  primals_395 = None
        mul_373: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_372, 1024)
        sum_30: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:481 in forward, code: hidden_states = self.dense(hidden_states)
        view_529: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(addmm_144, [4, 512, 1024]);  addmm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_364: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, 0.5)
        mul_365: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, 0.7071067811865476)
        erf_24: "f32[4, 512, 1024]" = torch.ops.aten.erf.default(mul_365);  mul_365 = None
        add_198: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_366: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_364, add_198);  mul_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:483 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_73: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_366, getitem_99);  mul_366 = getitem_99 = None
        mul_367: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_49);  sub_73 = None
        mul_374: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_372, mul_367);  mul_372 = None
        sum_31: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_374, [2], True);  mul_374 = None
        mul_375: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_367, sum_31);  sum_31 = None
        sub_78: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_373, sum_30);  mul_373 = sum_30 = None
        sub_79: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_78, mul_375);  sub_78 = mul_375 = None
        div_26: "f32[4, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 1024);  rsqrt_49 = None
        mul_376: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_26, sub_79);  div_26 = sub_79 = None
        mul_377: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(view_537, mul_367);  mul_367 = None
        sum_32: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 1]);  mul_377 = None
        sum_33: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_537, [0, 1]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_379: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_198, 0.5);  add_198 = None
        mul_380: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, view_529)
        mul_381: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_380, -0.5);  mul_380 = None
        exp_26: "f32[4, 512, 1024]" = torch.ops.aten.exp.default(mul_381);  mul_381 = None
        mul_382: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_383: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, mul_382);  view_529 = mul_382 = None
        add_203: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_379, mul_383);  mul_379 = mul_383 = None
        mul_384: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_376, add_203);  mul_376 = add_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:481 in forward, code: hidden_states = self.dense(hidden_states)
        view_538: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_384, [2048, 1024]);  mul_384 = None
        permute_264: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_393, [1, 0]);  primals_393 = None
        permute_270: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        mm_2: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_538, permute_270);  permute_270 = None
        permute_271: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_538, [1, 0])
        mm_3: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_271, view_528);  permute_271 = view_528 = None
        sum_34: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_538, [0], True);  view_538 = None
        view_539: "f32[1024]" = torch.ops.aten.reshape.default(sum_34, [1024]);  sum_34 = None
        view_540: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_2, [4, 512, 1024]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_386: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(view_540, primals_391);  primals_391 = None
        mul_387: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_386, 1024)
        sum_35: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_386, [2], True)
        mul_388: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_386, mul_362);  mul_386 = None
        sum_36: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
        mul_389: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_362, sum_36);  sum_36 = None
        sub_81: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_387, sum_35);  mul_387 = sum_35 = None
        sub_82: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_81, mul_389);  sub_81 = mul_389 = None
        mul_390: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_27, sub_82);  div_27 = sub_82 = None
        mul_391: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(view_540, mul_362);  mul_362 = None
        sum_37: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1]);  mul_391 = None
        sum_38: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_540, [0, 1]);  view_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_72, torch.float32);  gt_72 = None
        mul_392: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_393: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_390, mul_392);  mul_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_541: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_393, [2048, 1024]);  mul_393 = None
        permute_263: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_389, [1, 0]);  primals_389 = None
        permute_274: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        mm_4: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_541, permute_274);  permute_274 = None
        permute_275: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_541, [1, 0])
        mm_5: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_275, view_526);  permute_275 = view_526 = None
        sum_39: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_541, [0], True);  view_541 = None
        view_542: "f32[1024]" = torch.ops.aten.reshape.default(sum_39, [1024]);  sum_39 = None
        view_543: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_4, [4, 512, 3072]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_525: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_142, [4, 512, 3072]);  addmm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_358: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_525, 0.7071067811865476)
        erf_23: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_358);  mul_358 = None
        add_194: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_395: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_194, 0.5);  add_194 = None
        mul_396: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_525, view_525)
        mul_397: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_396, -0.5);  mul_396 = None
        exp_27: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_397);  mul_397 = None
        mul_398: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
        mul_399: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_525, mul_398);  view_525 = mul_398 = None
        add_205: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_395, mul_399);  mul_395 = mul_399 = None
        mul_400: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_543, add_205);  view_543 = add_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_544: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_400, [2048, 3072]);  mul_400 = None
        permute_262: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_387, [1, 0]);  primals_387 = None
        permute_278: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
        mm_6: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_544, permute_278);  permute_278 = None
        permute_279: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_544, [1, 0])
        mm_7: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_279, view_524);  permute_279 = view_524 = None
        sum_40: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_544, [0], True);  view_544 = None
        view_545: "f32[3072]" = torch.ops.aten.reshape.default(sum_40, [3072]);  sum_40 = None
        view_546: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_6, [4, 512, 1024]);  mm_6 = None
        add_206: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_390, view_546);  mul_390 = view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_402: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_206, primals_385);  primals_385 = None
        mul_403: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_402, 1024)
        sum_41: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True)
        mul_404: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_402, mul_355);  mul_402 = None
        sum_42: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True);  mul_404 = None
        mul_405: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_355, sum_42);  sum_42 = None
        sub_84: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_403, sum_41);  mul_403 = sum_41 = None
        sub_85: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_84, mul_405);  sub_84 = mul_405 = None
        mul_406: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_28, sub_85);  div_28 = sub_85 = None
        mul_407: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_206, mul_355);  mul_355 = None
        sum_43: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1]);  mul_407 = None
        sum_44: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_206, [0, 1]);  add_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_71, torch.float32);  gt_71 = None
        mul_408: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_409: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_406, mul_408);  mul_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_547: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_409, [2048, 1024]);  mul_409 = None
        permute_261: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_383, [1, 0]);  primals_383 = None
        permute_282: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
        mm_8: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_547, permute_282);  permute_282 = None
        permute_283: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_547, [1, 0])
        mm_9: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_283, view_522);  permute_283 = view_522 = None
        sum_45: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_547, [0], True);  view_547 = None
        view_548: "f32[1024]" = torch.ops.aten.reshape.default(sum_45, [1024]);  sum_45 = None
        view_549: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_8, [4, 512, 1024]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_550: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_549, [4, 512, 16, 64]);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_286: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_550, [0, 2, 1, 3]);  view_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_98: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
        view_551: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_98, [64, 512, 64]);  clone_98 = None
        bmm_48: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_287, view_551);  permute_287 = None
        bmm_49: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_551, permute_288);  view_551 = permute_288 = None
        view_552: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_48, [4, 16, 512, 64]);  bmm_48 = None
        view_553: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_49, [4, 16, 512, 512]);  bmm_49 = None
        convert_element_type_3: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_70, torch.float32);  gt_70 = None
        mul_410: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_411: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_553, mul_410);  view_553 = mul_410 = None
        mul_412: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_411, where_47);  mul_411 = None
        sum_46: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [-1], True)
        neg_1: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_47);  where_47 = None
        fma: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_1, sum_46, mul_412);  neg_1 = sum_46 = mul_412 = None
        view_554: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma, [64, 512, 512]);  fma = None
        bmm_50: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_289, view_554);  permute_289 = None
        bmm_51: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_554, permute_290);  view_554 = permute_290 = None
        view_555: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_50, [4, 16, 64, 512]);  bmm_50 = None
        view_556: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_51, [4, 16, 512, 64]);  bmm_51 = None
        mul_413: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_555, 0.3535533905932738);  view_555 = None
        permute_291: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_413, [0, 1, 3, 2]);  mul_413 = None
        mul_414: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_556, 0.3535533905932738);  view_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_292: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_552, [0, 2, 1, 3]);  view_552 = None
        clone_100: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
        view_557: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_100, [4, 512, 1024]);  clone_100 = None
        view_558: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_557, [2048, 1024]);  view_557 = None
        permute_257: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_381, [1, 0]);  primals_381 = None
        permute_293: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
        mm_10: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_558, permute_293);  permute_293 = None
        permute_294: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_558, [1, 0])
        mm_11: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_294, view_506);  permute_294 = None
        sum_47: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_558, [0], True);  view_558 = None
        view_559: "f32[1024]" = torch.ops.aten.reshape.default(sum_47, [1024]);  sum_47 = None
        view_560: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_10, [4, 512, 1024]);  mm_10 = None
        add_207: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_406, view_560);  mul_406 = view_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_297: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_291, [0, 2, 1, 3]);  permute_291 = None
        view_561: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_297, [4, 512, 1024]);  permute_297 = None
        clone_101: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_561, memory_format = torch.contiguous_format);  view_561 = None
        view_562: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_101, [2048, 1024]);  clone_101 = None
        permute_255: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_379, [1, 0]);  primals_379 = None
        permute_298: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
        mm_12: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_562, permute_298);  permute_298 = None
        permute_299: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_562, [1, 0])
        mm_13: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_299, view_506);  permute_299 = None
        sum_48: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
        view_563: "f32[1024]" = torch.ops.aten.reshape.default(sum_48, [1024]);  sum_48 = None
        view_564: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_12, [4, 512, 1024]);  mm_12 = None
        add_208: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_207, view_564);  add_207 = view_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_302: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_414, [0, 2, 1, 3]);  mul_414 = None
        clone_102: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_302, memory_format = torch.contiguous_format);  permute_302 = None
        view_565: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_102, [4, 512, 1024]);  clone_102 = None
        view_566: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_565, [2048, 1024]);  view_565 = None
        permute_253: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_377, [1, 0]);  primals_377 = None
        permute_303: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
        mm_14: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_566, permute_303);  permute_303 = None
        permute_304: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_566, [1, 0])
        mm_15: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_304, view_506);  permute_304 = view_506 = None
        sum_49: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_566, [0], True);  view_566 = None
        view_567: "f32[1024]" = torch.ops.aten.reshape.default(sum_49, [1024]);  sum_49 = None
        view_568: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_14, [4, 512, 1024]);  mm_14 = None
        add_209: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_208, view_568);  add_208 = view_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_416: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_209, primals_375);  primals_375 = None
        mul_417: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_416, 1024)
        sum_50: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [2], True)
        mul_418: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_416, mul_347);  mul_416 = None
        sum_51: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True);  mul_418 = None
        mul_419: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_347, sum_51);  sum_51 = None
        sub_87: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_417, sum_50);  mul_417 = sum_50 = None
        sub_88: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_87, mul_419);  sub_87 = mul_419 = None
        mul_420: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_29, sub_88);  div_29 = sub_88 = None
        mul_421: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_209, mul_347);  mul_347 = None
        sum_52: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 1]);  mul_421 = None
        sum_53: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_209, [0, 1]);  add_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_4: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_69, torch.float32);  gt_69 = None
        mul_422: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_423: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_420, mul_422);  mul_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_569: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_423, [2048, 1024]);  mul_423 = None
        permute_252: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_373, [1, 0]);  primals_373 = None
        permute_307: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
        mm_16: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_569, permute_307);  permute_307 = None
        permute_308: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_569, [1, 0])
        mm_17: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_308, view_504);  permute_308 = view_504 = None
        sum_54: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_569, [0], True);  view_569 = None
        view_570: "f32[1024]" = torch.ops.aten.reshape.default(sum_54, [1024]);  sum_54 = None
        view_571: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_16, [4, 512, 3072]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_503: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_136, [4, 512, 3072]);  addmm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_343: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_503, 0.7071067811865476)
        erf_22: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_343);  mul_343 = None
        add_186: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_425: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_186, 0.5);  add_186 = None
        mul_426: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_503, view_503)
        mul_427: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_426, -0.5);  mul_426 = None
        exp_28: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_427);  mul_427 = None
        mul_428: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
        mul_429: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_503, mul_428);  view_503 = mul_428 = None
        add_211: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_425, mul_429);  mul_425 = mul_429 = None
        mul_430: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_571, add_211);  view_571 = add_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_572: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_430, [2048, 3072]);  mul_430 = None
        permute_251: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_371, [1, 0]);  primals_371 = None
        permute_311: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
        mm_18: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_572, permute_311);  permute_311 = None
        permute_312: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_572, [1, 0])
        mm_19: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_312, view_502);  permute_312 = view_502 = None
        sum_55: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_572, [0], True);  view_572 = None
        view_573: "f32[3072]" = torch.ops.aten.reshape.default(sum_55, [3072]);  sum_55 = None
        view_574: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_18, [4, 512, 1024]);  mm_18 = None
        add_212: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_420, view_574);  mul_420 = view_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_432: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_212, primals_369);  primals_369 = None
        mul_433: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_432, 1024)
        sum_56: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [2], True)
        mul_434: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_432, mul_340);  mul_432 = None
        sum_57: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_434, [2], True);  mul_434 = None
        mul_435: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_340, sum_57);  sum_57 = None
        sub_90: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_433, sum_56);  mul_433 = sum_56 = None
        sub_91: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_90, mul_435);  sub_90 = mul_435 = None
        mul_436: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_30, sub_91);  div_30 = sub_91 = None
        mul_437: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_212, mul_340);  mul_340 = None
        sum_58: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 1]);  mul_437 = None
        sum_59: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_212, [0, 1]);  add_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_5: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_68, torch.float32);  gt_68 = None
        mul_438: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
        mul_439: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_436, mul_438);  mul_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_575: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_439, [2048, 1024]);  mul_439 = None
        permute_250: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_367, [1, 0]);  primals_367 = None
        permute_315: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
        mm_20: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_575, permute_315);  permute_315 = None
        permute_316: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_575, [1, 0])
        mm_21: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_316, view_500);  permute_316 = view_500 = None
        sum_60: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_575, [0], True);  view_575 = None
        view_576: "f32[1024]" = torch.ops.aten.reshape.default(sum_60, [1024]);  sum_60 = None
        view_577: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_20, [4, 512, 1024]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_578: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_577, [4, 512, 16, 64]);  view_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_319: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_105: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
        view_579: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_105, [64, 512, 64]);  clone_105 = None
        bmm_52: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_320, view_579);  permute_320 = None
        bmm_53: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_579, permute_321);  view_579 = permute_321 = None
        view_580: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_52, [4, 16, 512, 64]);  bmm_52 = None
        view_581: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_53, [4, 16, 512, 512]);  bmm_53 = None
        convert_element_type_6: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_67, torch.float32);  gt_67 = None
        mul_440: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
        mul_441: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_581, mul_440);  view_581 = mul_440 = None
        mul_442: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_441, where_45);  mul_441 = None
        sum_61: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [-1], True)
        neg_2: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_45);  where_45 = None
        fma_1: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_2, sum_61, mul_442);  neg_2 = sum_61 = mul_442 = None
        view_582: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_1, [64, 512, 512]);  fma_1 = None
        bmm_54: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_322, view_582);  permute_322 = None
        bmm_55: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_582, permute_323);  view_582 = permute_323 = None
        view_583: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_54, [4, 16, 64, 512]);  bmm_54 = None
        view_584: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_55, [4, 16, 512, 64]);  bmm_55 = None
        mul_443: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_583, 0.3535533905932738);  view_583 = None
        permute_324: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_443, [0, 1, 3, 2]);  mul_443 = None
        mul_444: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_584, 0.3535533905932738);  view_584 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_325: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_580, [0, 2, 1, 3]);  view_580 = None
        clone_107: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
        view_585: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_107, [4, 512, 1024]);  clone_107 = None
        view_586: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_585, [2048, 1024]);  view_585 = None
        permute_246: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_365, [1, 0]);  primals_365 = None
        permute_326: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
        mm_22: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_586, permute_326);  permute_326 = None
        permute_327: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_586, [1, 0])
        mm_23: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_327, view_484);  permute_327 = None
        sum_62: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_586, [0], True);  view_586 = None
        view_587: "f32[1024]" = torch.ops.aten.reshape.default(sum_62, [1024]);  sum_62 = None
        view_588: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_22, [4, 512, 1024]);  mm_22 = None
        add_213: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_436, view_588);  mul_436 = view_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_330: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_324, [0, 2, 1, 3]);  permute_324 = None
        view_589: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_330, [4, 512, 1024]);  permute_330 = None
        clone_108: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_589, memory_format = torch.contiguous_format);  view_589 = None
        view_590: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_108, [2048, 1024]);  clone_108 = None
        permute_244: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_363, [1, 0]);  primals_363 = None
        permute_331: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
        mm_24: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_590, permute_331);  permute_331 = None
        permute_332: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_590, [1, 0])
        mm_25: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_332, view_484);  permute_332 = None
        sum_63: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
        view_591: "f32[1024]" = torch.ops.aten.reshape.default(sum_63, [1024]);  sum_63 = None
        view_592: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_24, [4, 512, 1024]);  mm_24 = None
        add_214: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_213, view_592);  add_213 = view_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_335: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_444, [0, 2, 1, 3]);  mul_444 = None
        clone_109: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
        view_593: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_109, [4, 512, 1024]);  clone_109 = None
        view_594: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_593, [2048, 1024]);  view_593 = None
        permute_242: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_361, [1, 0]);  primals_361 = None
        permute_336: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        mm_26: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_594, permute_336);  permute_336 = None
        permute_337: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_594, [1, 0])
        mm_27: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_337, view_484);  permute_337 = view_484 = None
        sum_64: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_594, [0], True);  view_594 = None
        view_595: "f32[1024]" = torch.ops.aten.reshape.default(sum_64, [1024]);  sum_64 = None
        view_596: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_26, [4, 512, 1024]);  mm_26 = None
        add_215: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_214, view_596);  add_214 = view_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_446: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_215, primals_359);  primals_359 = None
        mul_447: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_446, 1024)
        sum_65: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_446, [2], True)
        mul_448: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_446, mul_332);  mul_446 = None
        sum_66: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_448, [2], True);  mul_448 = None
        mul_449: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_332, sum_66);  sum_66 = None
        sub_93: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_447, sum_65);  mul_447 = sum_65 = None
        sub_94: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_93, mul_449);  sub_93 = mul_449 = None
        mul_450: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_31, sub_94);  div_31 = sub_94 = None
        mul_451: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_215, mul_332);  mul_332 = None
        sum_67: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 1]);  mul_451 = None
        sum_68: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_215, [0, 1]);  add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_7: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_66, torch.float32);  gt_66 = None
        mul_452: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_453: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_450, mul_452);  mul_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_597: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_453, [2048, 1024]);  mul_453 = None
        permute_241: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_357, [1, 0]);  primals_357 = None
        permute_340: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        mm_28: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_597, permute_340);  permute_340 = None
        permute_341: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_597, [1, 0])
        mm_29: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_341, view_482);  permute_341 = view_482 = None
        sum_69: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_597, [0], True);  view_597 = None
        view_598: "f32[1024]" = torch.ops.aten.reshape.default(sum_69, [1024]);  sum_69 = None
        view_599: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_28, [4, 512, 3072]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_481: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_130, [4, 512, 3072]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_328: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_481, 0.7071067811865476)
        erf_21: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_328);  mul_328 = None
        add_178: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_455: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_178, 0.5);  add_178 = None
        mul_456: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_481, view_481)
        mul_457: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_456, -0.5);  mul_456 = None
        exp_29: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_457);  mul_457 = None
        mul_458: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
        mul_459: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_481, mul_458);  view_481 = mul_458 = None
        add_217: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_455, mul_459);  mul_455 = mul_459 = None
        mul_460: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_599, add_217);  view_599 = add_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_600: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_460, [2048, 3072]);  mul_460 = None
        permute_240: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_355, [1, 0]);  primals_355 = None
        permute_344: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
        mm_30: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_600, permute_344);  permute_344 = None
        permute_345: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_600, [1, 0])
        mm_31: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_345, view_480);  permute_345 = view_480 = None
        sum_70: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_600, [0], True);  view_600 = None
        view_601: "f32[3072]" = torch.ops.aten.reshape.default(sum_70, [3072]);  sum_70 = None
        view_602: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_30, [4, 512, 1024]);  mm_30 = None
        add_218: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_450, view_602);  mul_450 = view_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_462: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_218, primals_353);  primals_353 = None
        mul_463: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_462, 1024)
        sum_71: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_462, [2], True)
        mul_464: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_462, mul_325);  mul_462 = None
        sum_72: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True);  mul_464 = None
        mul_465: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_325, sum_72);  sum_72 = None
        sub_96: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_463, sum_71);  mul_463 = sum_71 = None
        sub_97: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_96, mul_465);  sub_96 = mul_465 = None
        mul_466: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_32, sub_97);  div_32 = sub_97 = None
        mul_467: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_218, mul_325);  mul_325 = None
        sum_73: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 1]);  mul_467 = None
        sum_74: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_218, [0, 1]);  add_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_8: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_65, torch.float32);  gt_65 = None
        mul_468: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_469: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_466, mul_468);  mul_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_603: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_469, [2048, 1024]);  mul_469 = None
        permute_239: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_351, [1, 0]);  primals_351 = None
        permute_348: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
        mm_32: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_603, permute_348);  permute_348 = None
        permute_349: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_603, [1, 0])
        mm_33: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_349, view_478);  permute_349 = view_478 = None
        sum_75: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_603, [0], True);  view_603 = None
        view_604: "f32[1024]" = torch.ops.aten.reshape.default(sum_75, [1024]);  sum_75 = None
        view_605: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_32, [4, 512, 1024]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_606: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_605, [4, 512, 16, 64]);  view_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_352: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_606, [0, 2, 1, 3]);  view_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_112: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None
        view_607: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_112, [64, 512, 64]);  clone_112 = None
        bmm_56: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_353, view_607);  permute_353 = None
        bmm_57: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_607, permute_354);  view_607 = permute_354 = None
        view_608: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_56, [4, 16, 512, 64]);  bmm_56 = None
        view_609: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_57, [4, 16, 512, 512]);  bmm_57 = None
        convert_element_type_9: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_64, torch.float32);  gt_64 = None
        mul_470: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
        mul_471: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_609, mul_470);  view_609 = mul_470 = None
        mul_472: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_471, where_43);  mul_471 = None
        sum_76: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [-1], True)
        neg_3: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_43);  where_43 = None
        fma_2: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_3, sum_76, mul_472);  neg_3 = sum_76 = mul_472 = None
        view_610: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_2, [64, 512, 512]);  fma_2 = None
        bmm_58: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_355, view_610);  permute_355 = None
        bmm_59: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_610, permute_356);  view_610 = permute_356 = None
        view_611: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_58, [4, 16, 64, 512]);  bmm_58 = None
        view_612: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_59, [4, 16, 512, 64]);  bmm_59 = None
        mul_473: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_611, 0.3535533905932738);  view_611 = None
        permute_357: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_473, [0, 1, 3, 2]);  mul_473 = None
        mul_474: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_612, 0.3535533905932738);  view_612 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_358: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_608, [0, 2, 1, 3]);  view_608 = None
        clone_114: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
        view_613: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_114, [4, 512, 1024]);  clone_114 = None
        view_614: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_613, [2048, 1024]);  view_613 = None
        permute_235: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_349, [1, 0]);  primals_349 = None
        permute_359: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
        mm_34: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_614, permute_359);  permute_359 = None
        permute_360: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_614, [1, 0])
        mm_35: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_360, view_462);  permute_360 = None
        sum_77: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_614, [0], True);  view_614 = None
        view_615: "f32[1024]" = torch.ops.aten.reshape.default(sum_77, [1024]);  sum_77 = None
        view_616: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_34, [4, 512, 1024]);  mm_34 = None
        add_219: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_466, view_616);  mul_466 = view_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_363: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_357, [0, 2, 1, 3]);  permute_357 = None
        view_617: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_363, [4, 512, 1024]);  permute_363 = None
        clone_115: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_617, memory_format = torch.contiguous_format);  view_617 = None
        view_618: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_115, [2048, 1024]);  clone_115 = None
        permute_233: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_347, [1, 0]);  primals_347 = None
        permute_364: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        mm_36: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_618, permute_364);  permute_364 = None
        permute_365: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_618, [1, 0])
        mm_37: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_365, view_462);  permute_365 = None
        sum_78: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_618, [0], True);  view_618 = None
        view_619: "f32[1024]" = torch.ops.aten.reshape.default(sum_78, [1024]);  sum_78 = None
        view_620: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_36, [4, 512, 1024]);  mm_36 = None
        add_220: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_219, view_620);  add_219 = view_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_368: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_474, [0, 2, 1, 3]);  mul_474 = None
        clone_116: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
        view_621: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_116, [4, 512, 1024]);  clone_116 = None
        view_622: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_621, [2048, 1024]);  view_621 = None
        permute_231: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_345, [1, 0]);  primals_345 = None
        permute_369: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
        mm_38: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_622, permute_369);  permute_369 = None
        permute_370: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_622, [1, 0])
        mm_39: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_370, view_462);  permute_370 = view_462 = None
        sum_79: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_622, [0], True);  view_622 = None
        view_623: "f32[1024]" = torch.ops.aten.reshape.default(sum_79, [1024]);  sum_79 = None
        view_624: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_38, [4, 512, 1024]);  mm_38 = None
        add_221: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_220, view_624);  add_220 = view_624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_476: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_221, primals_343);  primals_343 = None
        mul_477: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_476, 1024)
        sum_80: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True)
        mul_478: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_476, mul_317);  mul_476 = None
        sum_81: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_478, [2], True);  mul_478 = None
        mul_479: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_317, sum_81);  sum_81 = None
        sub_99: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_477, sum_80);  mul_477 = sum_80 = None
        sub_100: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_99, mul_479);  sub_99 = mul_479 = None
        mul_480: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_33, sub_100);  div_33 = sub_100 = None
        mul_481: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_221, mul_317);  mul_317 = None
        sum_82: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 1]);  mul_481 = None
        sum_83: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_221, [0, 1]);  add_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_10: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_63, torch.float32);  gt_63 = None
        mul_482: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_483: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_480, mul_482);  mul_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_625: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_483, [2048, 1024]);  mul_483 = None
        permute_230: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_341, [1, 0]);  primals_341 = None
        permute_373: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
        mm_40: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_625, permute_373);  permute_373 = None
        permute_374: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_625, [1, 0])
        mm_41: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_374, view_460);  permute_374 = view_460 = None
        sum_84: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_625, [0], True);  view_625 = None
        view_626: "f32[1024]" = torch.ops.aten.reshape.default(sum_84, [1024]);  sum_84 = None
        view_627: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_40, [4, 512, 3072]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_459: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_124, [4, 512, 3072]);  addmm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_313: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_459, 0.7071067811865476)
        erf_20: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_313);  mul_313 = None
        add_170: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_485: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_170, 0.5);  add_170 = None
        mul_486: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_459, view_459)
        mul_487: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_486, -0.5);  mul_486 = None
        exp_30: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_487);  mul_487 = None
        mul_488: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
        mul_489: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_459, mul_488);  view_459 = mul_488 = None
        add_223: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_485, mul_489);  mul_485 = mul_489 = None
        mul_490: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_627, add_223);  view_627 = add_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_628: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_490, [2048, 3072]);  mul_490 = None
        permute_229: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_339, [1, 0]);  primals_339 = None
        permute_377: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
        mm_42: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_628, permute_377);  permute_377 = None
        permute_378: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_628, [1, 0])
        mm_43: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_378, view_458);  permute_378 = view_458 = None
        sum_85: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_628, [0], True);  view_628 = None
        view_629: "f32[3072]" = torch.ops.aten.reshape.default(sum_85, [3072]);  sum_85 = None
        view_630: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_42, [4, 512, 1024]);  mm_42 = None
        add_224: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_480, view_630);  mul_480 = view_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_492: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_224, primals_337);  primals_337 = None
        mul_493: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_492, 1024)
        sum_86: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_492, [2], True)
        mul_494: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_492, mul_310);  mul_492 = None
        sum_87: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_494, [2], True);  mul_494 = None
        mul_495: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_310, sum_87);  sum_87 = None
        sub_102: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_493, sum_86);  mul_493 = sum_86 = None
        sub_103: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_102, mul_495);  sub_102 = mul_495 = None
        mul_496: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_34, sub_103);  div_34 = sub_103 = None
        mul_497: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_224, mul_310);  mul_310 = None
        sum_88: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_497, [0, 1]);  mul_497 = None
        sum_89: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_224, [0, 1]);  add_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_11: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_62, torch.float32);  gt_62 = None
        mul_498: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_499: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_496, mul_498);  mul_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_631: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_499, [2048, 1024]);  mul_499 = None
        permute_228: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_335, [1, 0]);  primals_335 = None
        permute_381: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
        mm_44: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_631, permute_381);  permute_381 = None
        permute_382: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_631, [1, 0])
        mm_45: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_382, view_456);  permute_382 = view_456 = None
        sum_90: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_631, [0], True);  view_631 = None
        view_632: "f32[1024]" = torch.ops.aten.reshape.default(sum_90, [1024]);  sum_90 = None
        view_633: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_44, [4, 512, 1024]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_634: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_633, [4, 512, 16, 64]);  view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_385: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_634, [0, 2, 1, 3]);  view_634 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_119: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
        view_635: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_119, [64, 512, 64]);  clone_119 = None
        bmm_60: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_386, view_635);  permute_386 = None
        bmm_61: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_635, permute_387);  view_635 = permute_387 = None
        view_636: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_60, [4, 16, 512, 64]);  bmm_60 = None
        view_637: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_61, [4, 16, 512, 512]);  bmm_61 = None
        convert_element_type_12: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_61, torch.float32);  gt_61 = None
        mul_500: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_501: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_637, mul_500);  view_637 = mul_500 = None
        mul_502: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_501, where_41);  mul_501 = None
        sum_91: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_502, [-1], True)
        neg_4: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_41);  where_41 = None
        fma_3: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_4, sum_91, mul_502);  neg_4 = sum_91 = mul_502 = None
        view_638: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_3, [64, 512, 512]);  fma_3 = None
        bmm_62: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_388, view_638);  permute_388 = None
        bmm_63: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_638, permute_389);  view_638 = permute_389 = None
        view_639: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_62, [4, 16, 64, 512]);  bmm_62 = None
        view_640: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_63, [4, 16, 512, 64]);  bmm_63 = None
        mul_503: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_639, 0.3535533905932738);  view_639 = None
        permute_390: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_503, [0, 1, 3, 2]);  mul_503 = None
        mul_504: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_640, 0.3535533905932738);  view_640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_391: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_636, [0, 2, 1, 3]);  view_636 = None
        clone_121: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
        view_641: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_121, [4, 512, 1024]);  clone_121 = None
        view_642: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_641, [2048, 1024]);  view_641 = None
        permute_224: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_333, [1, 0]);  primals_333 = None
        permute_392: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
        mm_46: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_642, permute_392);  permute_392 = None
        permute_393: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_642, [1, 0])
        mm_47: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_393, view_440);  permute_393 = None
        sum_92: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_642, [0], True);  view_642 = None
        view_643: "f32[1024]" = torch.ops.aten.reshape.default(sum_92, [1024]);  sum_92 = None
        view_644: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_46, [4, 512, 1024]);  mm_46 = None
        add_225: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_496, view_644);  mul_496 = view_644 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_396: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_390, [0, 2, 1, 3]);  permute_390 = None
        view_645: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_396, [4, 512, 1024]);  permute_396 = None
        clone_122: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_645, memory_format = torch.contiguous_format);  view_645 = None
        view_646: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_122, [2048, 1024]);  clone_122 = None
        permute_222: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_331, [1, 0]);  primals_331 = None
        permute_397: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        mm_48: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_646, permute_397);  permute_397 = None
        permute_398: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_646, [1, 0])
        mm_49: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_398, view_440);  permute_398 = None
        sum_93: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_646, [0], True);  view_646 = None
        view_647: "f32[1024]" = torch.ops.aten.reshape.default(sum_93, [1024]);  sum_93 = None
        view_648: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_48, [4, 512, 1024]);  mm_48 = None
        add_226: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_225, view_648);  add_225 = view_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_401: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_504, [0, 2, 1, 3]);  mul_504 = None
        clone_123: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_401, memory_format = torch.contiguous_format);  permute_401 = None
        view_649: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_123, [4, 512, 1024]);  clone_123 = None
        view_650: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_649, [2048, 1024]);  view_649 = None
        permute_220: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_329, [1, 0]);  primals_329 = None
        permute_402: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
        mm_50: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_650, permute_402);  permute_402 = None
        permute_403: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_650, [1, 0])
        mm_51: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_403, view_440);  permute_403 = view_440 = None
        sum_94: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_650, [0], True);  view_650 = None
        view_651: "f32[1024]" = torch.ops.aten.reshape.default(sum_94, [1024]);  sum_94 = None
        view_652: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_50, [4, 512, 1024]);  mm_50 = None
        add_227: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_226, view_652);  add_226 = view_652 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_506: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_227, primals_327);  primals_327 = None
        mul_507: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_506, 1024)
        sum_95: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_506, [2], True)
        mul_508: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_506, mul_302);  mul_506 = None
        sum_96: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [2], True);  mul_508 = None
        mul_509: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_302, sum_96);  sum_96 = None
        sub_105: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_507, sum_95);  mul_507 = sum_95 = None
        sub_106: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_105, mul_509);  sub_105 = mul_509 = None
        mul_510: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_35, sub_106);  div_35 = sub_106 = None
        mul_511: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_227, mul_302);  mul_302 = None
        sum_97: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 1]);  mul_511 = None
        sum_98: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_227, [0, 1]);  add_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_13: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_60, torch.float32);  gt_60 = None
        mul_512: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_513: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_510, mul_512);  mul_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_653: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_513, [2048, 1024]);  mul_513 = None
        permute_219: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_325, [1, 0]);  primals_325 = None
        permute_406: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
        mm_52: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_653, permute_406);  permute_406 = None
        permute_407: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_653, [1, 0])
        mm_53: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_407, view_438);  permute_407 = view_438 = None
        sum_99: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_653, [0], True);  view_653 = None
        view_654: "f32[1024]" = torch.ops.aten.reshape.default(sum_99, [1024]);  sum_99 = None
        view_655: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_52, [4, 512, 3072]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_437: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_118, [4, 512, 3072]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_298: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_437, 0.7071067811865476)
        erf_19: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_298);  mul_298 = None
        add_162: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_515: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_162, 0.5);  add_162 = None
        mul_516: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_437, view_437)
        mul_517: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_516, -0.5);  mul_516 = None
        exp_31: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_517);  mul_517 = None
        mul_518: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
        mul_519: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_437, mul_518);  view_437 = mul_518 = None
        add_229: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_515, mul_519);  mul_515 = mul_519 = None
        mul_520: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_655, add_229);  view_655 = add_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_656: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_520, [2048, 3072]);  mul_520 = None
        permute_218: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_323, [1, 0]);  primals_323 = None
        permute_410: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
        mm_54: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_656, permute_410);  permute_410 = None
        permute_411: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_656, [1, 0])
        mm_55: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_411, view_436);  permute_411 = view_436 = None
        sum_100: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_656, [0], True);  view_656 = None
        view_657: "f32[3072]" = torch.ops.aten.reshape.default(sum_100, [3072]);  sum_100 = None
        view_658: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_54, [4, 512, 1024]);  mm_54 = None
        add_230: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_510, view_658);  mul_510 = view_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_522: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_230, primals_321);  primals_321 = None
        mul_523: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_522, 1024)
        sum_101: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [2], True)
        mul_524: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_522, mul_295);  mul_522 = None
        sum_102: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_524, [2], True);  mul_524 = None
        mul_525: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_295, sum_102);  sum_102 = None
        sub_108: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_523, sum_101);  mul_523 = sum_101 = None
        sub_109: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_108, mul_525);  sub_108 = mul_525 = None
        mul_526: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_36, sub_109);  div_36 = sub_109 = None
        mul_527: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_230, mul_295);  mul_295 = None
        sum_103: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 1]);  mul_527 = None
        sum_104: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_14: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_59, torch.float32);  gt_59 = None
        mul_528: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_529: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_526, mul_528);  mul_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_659: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_529, [2048, 1024]);  mul_529 = None
        permute_217: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_319, [1, 0]);  primals_319 = None
        permute_414: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
        mm_56: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_659, permute_414);  permute_414 = None
        permute_415: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_659, [1, 0])
        mm_57: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_415, view_434);  permute_415 = view_434 = None
        sum_105: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_659, [0], True);  view_659 = None
        view_660: "f32[1024]" = torch.ops.aten.reshape.default(sum_105, [1024]);  sum_105 = None
        view_661: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_56, [4, 512, 1024]);  mm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_662: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_661, [4, 512, 16, 64]);  view_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_418: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_662, [0, 2, 1, 3]);  view_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_126: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
        view_663: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_126, [64, 512, 64]);  clone_126 = None
        bmm_64: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_419, view_663);  permute_419 = None
        bmm_65: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_663, permute_420);  view_663 = permute_420 = None
        view_664: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_64, [4, 16, 512, 64]);  bmm_64 = None
        view_665: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_65, [4, 16, 512, 512]);  bmm_65 = None
        convert_element_type_15: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_58, torch.float32);  gt_58 = None
        mul_530: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
        mul_531: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_665, mul_530);  view_665 = mul_530 = None
        mul_532: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_531, where_39);  mul_531 = None
        sum_106: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_532, [-1], True)
        neg_5: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_39);  where_39 = None
        fma_4: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_5, sum_106, mul_532);  neg_5 = sum_106 = mul_532 = None
        view_666: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_4, [64, 512, 512]);  fma_4 = None
        bmm_66: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_421, view_666);  permute_421 = None
        bmm_67: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_666, permute_422);  view_666 = permute_422 = None
        view_667: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_66, [4, 16, 64, 512]);  bmm_66 = None
        view_668: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_67, [4, 16, 512, 64]);  bmm_67 = None
        mul_533: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_667, 0.3535533905932738);  view_667 = None
        permute_423: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_533, [0, 1, 3, 2]);  mul_533 = None
        mul_534: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_668, 0.3535533905932738);  view_668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_424: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_664, [0, 2, 1, 3]);  view_664 = None
        clone_128: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_424, memory_format = torch.contiguous_format);  permute_424 = None
        view_669: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_128, [4, 512, 1024]);  clone_128 = None
        view_670: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_669, [2048, 1024]);  view_669 = None
        permute_213: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_317, [1, 0]);  primals_317 = None
        permute_425: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
        mm_58: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_670, permute_425);  permute_425 = None
        permute_426: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_670, [1, 0])
        mm_59: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_426, view_418);  permute_426 = None
        sum_107: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_670, [0], True);  view_670 = None
        view_671: "f32[1024]" = torch.ops.aten.reshape.default(sum_107, [1024]);  sum_107 = None
        view_672: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_58, [4, 512, 1024]);  mm_58 = None
        add_231: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_526, view_672);  mul_526 = view_672 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_429: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_423, [0, 2, 1, 3]);  permute_423 = None
        view_673: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_429, [4, 512, 1024]);  permute_429 = None
        clone_129: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_673, memory_format = torch.contiguous_format);  view_673 = None
        view_674: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_129, [2048, 1024]);  clone_129 = None
        permute_211: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_315, [1, 0]);  primals_315 = None
        permute_430: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        mm_60: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_674, permute_430);  permute_430 = None
        permute_431: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_674, [1, 0])
        mm_61: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_431, view_418);  permute_431 = None
        sum_108: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_674, [0], True);  view_674 = None
        view_675: "f32[1024]" = torch.ops.aten.reshape.default(sum_108, [1024]);  sum_108 = None
        view_676: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_60, [4, 512, 1024]);  mm_60 = None
        add_232: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_231, view_676);  add_231 = view_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_434: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_534, [0, 2, 1, 3]);  mul_534 = None
        clone_130: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_434, memory_format = torch.contiguous_format);  permute_434 = None
        view_677: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_130, [4, 512, 1024]);  clone_130 = None
        view_678: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_677, [2048, 1024]);  view_677 = None
        permute_209: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_313, [1, 0]);  primals_313 = None
        permute_435: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
        mm_62: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_678, permute_435);  permute_435 = None
        permute_436: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_678, [1, 0])
        mm_63: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_436, view_418);  permute_436 = view_418 = None
        sum_109: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_678, [0], True);  view_678 = None
        view_679: "f32[1024]" = torch.ops.aten.reshape.default(sum_109, [1024]);  sum_109 = None
        view_680: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_62, [4, 512, 1024]);  mm_62 = None
        add_233: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_232, view_680);  add_232 = view_680 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_536: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_233, primals_311);  primals_311 = None
        mul_537: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_536, 1024)
        sum_110: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_536, [2], True)
        mul_538: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_536, mul_287);  mul_536 = None
        sum_111: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_538, [2], True);  mul_538 = None
        mul_539: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_287, sum_111);  sum_111 = None
        sub_111: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_537, sum_110);  mul_537 = sum_110 = None
        sub_112: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_111, mul_539);  sub_111 = mul_539 = None
        mul_540: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_37, sub_112);  div_37 = sub_112 = None
        mul_541: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_233, mul_287);  mul_287 = None
        sum_112: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_541, [0, 1]);  mul_541 = None
        sum_113: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_16: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_57, torch.float32);  gt_57 = None
        mul_542: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_543: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_540, mul_542);  mul_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_681: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_543, [2048, 1024]);  mul_543 = None
        permute_208: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_309, [1, 0]);  primals_309 = None
        permute_439: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
        mm_64: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_681, permute_439);  permute_439 = None
        permute_440: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_681, [1, 0])
        mm_65: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_440, view_416);  permute_440 = view_416 = None
        sum_114: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_681, [0], True);  view_681 = None
        view_682: "f32[1024]" = torch.ops.aten.reshape.default(sum_114, [1024]);  sum_114 = None
        view_683: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_64, [4, 512, 3072]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_415: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_112, [4, 512, 3072]);  addmm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_283: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476)
        erf_18: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_283);  mul_283 = None
        add_154: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_545: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_154, 0.5);  add_154 = None
        mul_546: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_415, view_415)
        mul_547: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_546, -0.5);  mul_546 = None
        exp_32: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_547);  mul_547 = None
        mul_548: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
        mul_549: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_415, mul_548);  view_415 = mul_548 = None
        add_235: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_545, mul_549);  mul_545 = mul_549 = None
        mul_550: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_683, add_235);  view_683 = add_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_684: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_550, [2048, 3072]);  mul_550 = None
        permute_207: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_307, [1, 0]);  primals_307 = None
        permute_443: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        mm_66: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_684, permute_443);  permute_443 = None
        permute_444: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_684, [1, 0])
        mm_67: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_444, view_414);  permute_444 = view_414 = None
        sum_115: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_684, [0], True);  view_684 = None
        view_685: "f32[3072]" = torch.ops.aten.reshape.default(sum_115, [3072]);  sum_115 = None
        view_686: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_66, [4, 512, 1024]);  mm_66 = None
        add_236: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_540, view_686);  mul_540 = view_686 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_552: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_236, primals_305);  primals_305 = None
        mul_553: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_552, 1024)
        sum_116: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_552, [2], True)
        mul_554: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_552, mul_280);  mul_552 = None
        sum_117: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_554, [2], True);  mul_554 = None
        mul_555: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_280, sum_117);  sum_117 = None
        sub_114: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_553, sum_116);  mul_553 = sum_116 = None
        sub_115: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_114, mul_555);  sub_114 = mul_555 = None
        mul_556: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_38, sub_115);  div_38 = sub_115 = None
        mul_557: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_236, mul_280);  mul_280 = None
        sum_118: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_557, [0, 1]);  mul_557 = None
        sum_119: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_236, [0, 1]);  add_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_17: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_56, torch.float32);  gt_56 = None
        mul_558: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_559: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_556, mul_558);  mul_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_687: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_559, [2048, 1024]);  mul_559 = None
        permute_206: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_303, [1, 0]);  primals_303 = None
        permute_447: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
        mm_68: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_687, permute_447);  permute_447 = None
        permute_448: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_687, [1, 0])
        mm_69: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_448, view_412);  permute_448 = view_412 = None
        sum_120: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_687, [0], True);  view_687 = None
        view_688: "f32[1024]" = torch.ops.aten.reshape.default(sum_120, [1024]);  sum_120 = None
        view_689: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_68, [4, 512, 1024]);  mm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_690: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_689, [4, 512, 16, 64]);  view_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_451: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_690, [0, 2, 1, 3]);  view_690 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_133: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_451, memory_format = torch.contiguous_format);  permute_451 = None
        view_691: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_133, [64, 512, 64]);  clone_133 = None
        bmm_68: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_452, view_691);  permute_452 = None
        bmm_69: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_691, permute_453);  view_691 = permute_453 = None
        view_692: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_68, [4, 16, 512, 64]);  bmm_68 = None
        view_693: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_69, [4, 16, 512, 512]);  bmm_69 = None
        convert_element_type_18: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_55, torch.float32);  gt_55 = None
        mul_560: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
        mul_561: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_693, mul_560);  view_693 = mul_560 = None
        mul_562: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_561, where_37);  mul_561 = None
        sum_121: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_562, [-1], True)
        neg_6: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_37);  where_37 = None
        fma_5: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_6, sum_121, mul_562);  neg_6 = sum_121 = mul_562 = None
        view_694: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_5, [64, 512, 512]);  fma_5 = None
        bmm_70: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_454, view_694);  permute_454 = None
        bmm_71: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_694, permute_455);  view_694 = permute_455 = None
        view_695: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_70, [4, 16, 64, 512]);  bmm_70 = None
        view_696: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_71, [4, 16, 512, 64]);  bmm_71 = None
        mul_563: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_695, 0.3535533905932738);  view_695 = None
        permute_456: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_563, [0, 1, 3, 2]);  mul_563 = None
        mul_564: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_696, 0.3535533905932738);  view_696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_457: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_692, [0, 2, 1, 3]);  view_692 = None
        clone_135: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
        view_697: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_135, [4, 512, 1024]);  clone_135 = None
        view_698: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_697, [2048, 1024]);  view_697 = None
        permute_202: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_301, [1, 0]);  primals_301 = None
        permute_458: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
        mm_70: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_698, permute_458);  permute_458 = None
        permute_459: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_698, [1, 0])
        mm_71: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_459, view_396);  permute_459 = None
        sum_122: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_698, [0], True);  view_698 = None
        view_699: "f32[1024]" = torch.ops.aten.reshape.default(sum_122, [1024]);  sum_122 = None
        view_700: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_70, [4, 512, 1024]);  mm_70 = None
        add_237: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_556, view_700);  mul_556 = view_700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_462: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_456, [0, 2, 1, 3]);  permute_456 = None
        view_701: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_462, [4, 512, 1024]);  permute_462 = None
        clone_136: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_701, memory_format = torch.contiguous_format);  view_701 = None
        view_702: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_136, [2048, 1024]);  clone_136 = None
        permute_200: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_299, [1, 0]);  primals_299 = None
        permute_463: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
        mm_72: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_702, permute_463);  permute_463 = None
        permute_464: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_702, [1, 0])
        mm_73: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_464, view_396);  permute_464 = None
        sum_123: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_702, [0], True);  view_702 = None
        view_703: "f32[1024]" = torch.ops.aten.reshape.default(sum_123, [1024]);  sum_123 = None
        view_704: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_72, [4, 512, 1024]);  mm_72 = None
        add_238: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_237, view_704);  add_237 = view_704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_467: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_564, [0, 2, 1, 3]);  mul_564 = None
        clone_137: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_467, memory_format = torch.contiguous_format);  permute_467 = None
        view_705: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_137, [4, 512, 1024]);  clone_137 = None
        view_706: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_705, [2048, 1024]);  view_705 = None
        permute_198: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_297, [1, 0]);  primals_297 = None
        permute_468: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
        mm_74: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_706, permute_468);  permute_468 = None
        permute_469: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_706, [1, 0])
        mm_75: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_469, view_396);  permute_469 = view_396 = None
        sum_124: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_706, [0], True);  view_706 = None
        view_707: "f32[1024]" = torch.ops.aten.reshape.default(sum_124, [1024]);  sum_124 = None
        view_708: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_74, [4, 512, 1024]);  mm_74 = None
        add_239: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_238, view_708);  add_238 = view_708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_566: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_239, primals_295);  primals_295 = None
        mul_567: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_566, 1024)
        sum_125: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_566, [2], True)
        mul_568: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_566, mul_272);  mul_566 = None
        sum_126: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_568, [2], True);  mul_568 = None
        mul_569: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_272, sum_126);  sum_126 = None
        sub_117: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_567, sum_125);  mul_567 = sum_125 = None
        sub_118: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_117, mul_569);  sub_117 = mul_569 = None
        mul_570: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_39, sub_118);  div_39 = sub_118 = None
        mul_571: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_239, mul_272);  mul_272 = None
        sum_127: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_571, [0, 1]);  mul_571 = None
        sum_128: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_239, [0, 1]);  add_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_19: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_54, torch.float32);  gt_54 = None
        mul_572: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_573: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_570, mul_572);  mul_572 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_709: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_573, [2048, 1024]);  mul_573 = None
        permute_197: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_293, [1, 0]);  primals_293 = None
        permute_472: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        mm_76: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_709, permute_472);  permute_472 = None
        permute_473: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_709, [1, 0])
        mm_77: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_473, view_394);  permute_473 = view_394 = None
        sum_129: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_709, [0], True);  view_709 = None
        view_710: "f32[1024]" = torch.ops.aten.reshape.default(sum_129, [1024]);  sum_129 = None
        view_711: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_76, [4, 512, 3072]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_393: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_106, [4, 512, 3072]);  addmm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_268: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476)
        erf_17: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_268);  mul_268 = None
        add_146: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_575: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_146, 0.5);  add_146 = None
        mul_576: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_393, view_393)
        mul_577: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_576, -0.5);  mul_576 = None
        exp_33: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_577);  mul_577 = None
        mul_578: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
        mul_579: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_393, mul_578);  view_393 = mul_578 = None
        add_241: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_575, mul_579);  mul_575 = mul_579 = None
        mul_580: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_711, add_241);  view_711 = add_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_712: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_580, [2048, 3072]);  mul_580 = None
        permute_196: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
        permute_476: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
        mm_78: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_712, permute_476);  permute_476 = None
        permute_477: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_712, [1, 0])
        mm_79: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_477, view_392);  permute_477 = view_392 = None
        sum_130: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_712, [0], True);  view_712 = None
        view_713: "f32[3072]" = torch.ops.aten.reshape.default(sum_130, [3072]);  sum_130 = None
        view_714: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_78, [4, 512, 1024]);  mm_78 = None
        add_242: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_570, view_714);  mul_570 = view_714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_582: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_242, primals_289);  primals_289 = None
        mul_583: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_582, 1024)
        sum_131: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_582, [2], True)
        mul_584: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_582, mul_265);  mul_582 = None
        sum_132: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_584, [2], True);  mul_584 = None
        mul_585: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_265, sum_132);  sum_132 = None
        sub_120: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_583, sum_131);  mul_583 = sum_131 = None
        sub_121: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_120, mul_585);  sub_120 = mul_585 = None
        mul_586: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_40, sub_121);  div_40 = sub_121 = None
        mul_587: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_242, mul_265);  mul_265 = None
        sum_133: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_587, [0, 1]);  mul_587 = None
        sum_134: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_242, [0, 1]);  add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_20: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_53, torch.float32);  gt_53 = None
        mul_588: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_589: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_586, mul_588);  mul_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_715: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_589, [2048, 1024]);  mul_589 = None
        permute_195: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_287, [1, 0]);  primals_287 = None
        permute_480: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        mm_80: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_715, permute_480);  permute_480 = None
        permute_481: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_715, [1, 0])
        mm_81: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_481, view_390);  permute_481 = view_390 = None
        sum_135: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_715, [0], True);  view_715 = None
        view_716: "f32[1024]" = torch.ops.aten.reshape.default(sum_135, [1024]);  sum_135 = None
        view_717: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_80, [4, 512, 1024]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_718: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_717, [4, 512, 16, 64]);  view_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_484: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_718, [0, 2, 1, 3]);  view_718 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_140: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
        view_719: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_140, [64, 512, 64]);  clone_140 = None
        bmm_72: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_485, view_719);  permute_485 = None
        bmm_73: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_719, permute_486);  view_719 = permute_486 = None
        view_720: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_72, [4, 16, 512, 64]);  bmm_72 = None
        view_721: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_73, [4, 16, 512, 512]);  bmm_73 = None
        convert_element_type_21: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_52, torch.float32);  gt_52 = None
        mul_590: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
        mul_591: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_721, mul_590);  view_721 = mul_590 = None
        mul_592: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_591, where_35);  mul_591 = None
        sum_136: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_592, [-1], True)
        neg_7: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_35);  where_35 = None
        fma_6: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_7, sum_136, mul_592);  neg_7 = sum_136 = mul_592 = None
        view_722: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_6, [64, 512, 512]);  fma_6 = None
        bmm_74: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_487, view_722);  permute_487 = None
        bmm_75: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_722, permute_488);  view_722 = permute_488 = None
        view_723: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_74, [4, 16, 64, 512]);  bmm_74 = None
        view_724: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_75, [4, 16, 512, 64]);  bmm_75 = None
        mul_593: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_723, 0.3535533905932738);  view_723 = None
        permute_489: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_593, [0, 1, 3, 2]);  mul_593 = None
        mul_594: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_724, 0.3535533905932738);  view_724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_490: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_720, [0, 2, 1, 3]);  view_720 = None
        clone_142: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
        view_725: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_142, [4, 512, 1024]);  clone_142 = None
        view_726: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_725, [2048, 1024]);  view_725 = None
        permute_191: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
        permute_491: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
        mm_82: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_726, permute_491);  permute_491 = None
        permute_492: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_726, [1, 0])
        mm_83: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_492, view_374);  permute_492 = None
        sum_137: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_726, [0], True);  view_726 = None
        view_727: "f32[1024]" = torch.ops.aten.reshape.default(sum_137, [1024]);  sum_137 = None
        view_728: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_82, [4, 512, 1024]);  mm_82 = None
        add_243: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_586, view_728);  mul_586 = view_728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_495: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_489, [0, 2, 1, 3]);  permute_489 = None
        view_729: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_495, [4, 512, 1024]);  permute_495 = None
        clone_143: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_729, memory_format = torch.contiguous_format);  view_729 = None
        view_730: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_143, [2048, 1024]);  clone_143 = None
        permute_189: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_283, [1, 0]);  primals_283 = None
        permute_496: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
        mm_84: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_730, permute_496);  permute_496 = None
        permute_497: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_730, [1, 0])
        mm_85: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_497, view_374);  permute_497 = None
        sum_138: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_730, [0], True);  view_730 = None
        view_731: "f32[1024]" = torch.ops.aten.reshape.default(sum_138, [1024]);  sum_138 = None
        view_732: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_84, [4, 512, 1024]);  mm_84 = None
        add_244: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_243, view_732);  add_243 = view_732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_500: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_594, [0, 2, 1, 3]);  mul_594 = None
        clone_144: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_500, memory_format = torch.contiguous_format);  permute_500 = None
        view_733: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_144, [4, 512, 1024]);  clone_144 = None
        view_734: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_733, [2048, 1024]);  view_733 = None
        permute_187: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_281, [1, 0]);  primals_281 = None
        permute_501: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
        mm_86: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_734, permute_501);  permute_501 = None
        permute_502: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_734, [1, 0])
        mm_87: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_502, view_374);  permute_502 = view_374 = None
        sum_139: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_734, [0], True);  view_734 = None
        view_735: "f32[1024]" = torch.ops.aten.reshape.default(sum_139, [1024]);  sum_139 = None
        view_736: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_86, [4, 512, 1024]);  mm_86 = None
        add_245: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_244, view_736);  add_244 = view_736 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_596: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_245, primals_279);  primals_279 = None
        mul_597: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_596, 1024)
        sum_140: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_596, [2], True)
        mul_598: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_596, mul_257);  mul_596 = None
        sum_141: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_598, [2], True);  mul_598 = None
        mul_599: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_257, sum_141);  sum_141 = None
        sub_123: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_597, sum_140);  mul_597 = sum_140 = None
        sub_124: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_123, mul_599);  sub_123 = mul_599 = None
        mul_600: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_41, sub_124);  div_41 = sub_124 = None
        mul_601: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_245, mul_257);  mul_257 = None
        sum_142: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_601, [0, 1]);  mul_601 = None
        sum_143: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_245, [0, 1]);  add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_22: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_51, torch.float32);  gt_51 = None
        mul_602: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
        mul_603: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_600, mul_602);  mul_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_737: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_603, [2048, 1024]);  mul_603 = None
        permute_186: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_277, [1, 0]);  primals_277 = None
        permute_505: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        mm_88: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_737, permute_505);  permute_505 = None
        permute_506: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_737, [1, 0])
        mm_89: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_506, view_372);  permute_506 = view_372 = None
        sum_144: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_737, [0], True);  view_737 = None
        view_738: "f32[1024]" = torch.ops.aten.reshape.default(sum_144, [1024]);  sum_144 = None
        view_739: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_88, [4, 512, 3072]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_371: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_100, [4, 512, 3072]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_253: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476)
        erf_16: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_253);  mul_253 = None
        add_138: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_605: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_138, 0.5);  add_138 = None
        mul_606: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_371, view_371)
        mul_607: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_606, -0.5);  mul_606 = None
        exp_34: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_607);  mul_607 = None
        mul_608: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
        mul_609: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_371, mul_608);  view_371 = mul_608 = None
        add_247: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_605, mul_609);  mul_605 = mul_609 = None
        mul_610: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_739, add_247);  view_739 = add_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_740: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_610, [2048, 3072]);  mul_610 = None
        permute_185: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_275, [1, 0]);  primals_275 = None
        permute_509: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        mm_90: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_740, permute_509);  permute_509 = None
        permute_510: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_740, [1, 0])
        mm_91: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_510, view_370);  permute_510 = view_370 = None
        sum_145: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_740, [0], True);  view_740 = None
        view_741: "f32[3072]" = torch.ops.aten.reshape.default(sum_145, [3072]);  sum_145 = None
        view_742: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_90, [4, 512, 1024]);  mm_90 = None
        add_248: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_600, view_742);  mul_600 = view_742 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_612: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_248, primals_273);  primals_273 = None
        mul_613: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_612, 1024)
        sum_146: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_612, [2], True)
        mul_614: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_612, mul_250);  mul_612 = None
        sum_147: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_614, [2], True);  mul_614 = None
        mul_615: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_250, sum_147);  sum_147 = None
        sub_126: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_613, sum_146);  mul_613 = sum_146 = None
        sub_127: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_126, mul_615);  sub_126 = mul_615 = None
        mul_616: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_42, sub_127);  div_42 = sub_127 = None
        mul_617: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_248, mul_250);  mul_250 = None
        sum_148: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_617, [0, 1]);  mul_617 = None
        sum_149: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_248, [0, 1]);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_23: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_50, torch.float32);  gt_50 = None
        mul_618: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
        mul_619: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_616, mul_618);  mul_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_743: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_619, [2048, 1024]);  mul_619 = None
        permute_184: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
        permute_513: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
        mm_92: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_743, permute_513);  permute_513 = None
        permute_514: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_743, [1, 0])
        mm_93: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_514, view_368);  permute_514 = view_368 = None
        sum_150: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_743, [0], True);  view_743 = None
        view_744: "f32[1024]" = torch.ops.aten.reshape.default(sum_150, [1024]);  sum_150 = None
        view_745: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_92, [4, 512, 1024]);  mm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_746: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_745, [4, 512, 16, 64]);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_517: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_746, [0, 2, 1, 3]);  view_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_147: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format);  permute_517 = None
        view_747: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_147, [64, 512, 64]);  clone_147 = None
        bmm_76: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_518, view_747);  permute_518 = None
        bmm_77: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_747, permute_519);  view_747 = permute_519 = None
        view_748: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_76, [4, 16, 512, 64]);  bmm_76 = None
        view_749: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_77, [4, 16, 512, 512]);  bmm_77 = None
        convert_element_type_24: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_49, torch.float32);  gt_49 = None
        mul_620: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
        mul_621: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_749, mul_620);  view_749 = mul_620 = None
        mul_622: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_621, where_33);  mul_621 = None
        sum_151: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_622, [-1], True)
        neg_8: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_33);  where_33 = None
        fma_7: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_8, sum_151, mul_622);  neg_8 = sum_151 = mul_622 = None
        view_750: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_7, [64, 512, 512]);  fma_7 = None
        bmm_78: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_520, view_750);  permute_520 = None
        bmm_79: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_750, permute_521);  view_750 = permute_521 = None
        view_751: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_78, [4, 16, 64, 512]);  bmm_78 = None
        view_752: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_79, [4, 16, 512, 64]);  bmm_79 = None
        mul_623: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_751, 0.3535533905932738);  view_751 = None
        permute_522: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_623, [0, 1, 3, 2]);  mul_623 = None
        mul_624: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_752, 0.3535533905932738);  view_752 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_523: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_748, [0, 2, 1, 3]);  view_748 = None
        clone_149: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
        view_753: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_149, [4, 512, 1024]);  clone_149 = None
        view_754: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_753, [2048, 1024]);  view_753 = None
        permute_180: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_269, [1, 0]);  primals_269 = None
        permute_524: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
        mm_94: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_754, permute_524);  permute_524 = None
        permute_525: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_754, [1, 0])
        mm_95: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_525, view_352);  permute_525 = None
        sum_152: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_754, [0], True);  view_754 = None
        view_755: "f32[1024]" = torch.ops.aten.reshape.default(sum_152, [1024]);  sum_152 = None
        view_756: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_94, [4, 512, 1024]);  mm_94 = None
        add_249: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_616, view_756);  mul_616 = view_756 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_528: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_522, [0, 2, 1, 3]);  permute_522 = None
        view_757: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_528, [4, 512, 1024]);  permute_528 = None
        clone_150: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_757, memory_format = torch.contiguous_format);  view_757 = None
        view_758: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_150, [2048, 1024]);  clone_150 = None
        permute_178: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_267, [1, 0]);  primals_267 = None
        permute_529: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        mm_96: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_758, permute_529);  permute_529 = None
        permute_530: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_758, [1, 0])
        mm_97: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_530, view_352);  permute_530 = None
        sum_153: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_758, [0], True);  view_758 = None
        view_759: "f32[1024]" = torch.ops.aten.reshape.default(sum_153, [1024]);  sum_153 = None
        view_760: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_96, [4, 512, 1024]);  mm_96 = None
        add_250: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_249, view_760);  add_249 = view_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_533: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_624, [0, 2, 1, 3]);  mul_624 = None
        clone_151: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_533, memory_format = torch.contiguous_format);  permute_533 = None
        view_761: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_151, [4, 512, 1024]);  clone_151 = None
        view_762: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_761, [2048, 1024]);  view_761 = None
        permute_176: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
        permute_534: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        mm_98: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_762, permute_534);  permute_534 = None
        permute_535: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_762, [1, 0])
        mm_99: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_535, view_352);  permute_535 = view_352 = None
        sum_154: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_762, [0], True);  view_762 = None
        view_763: "f32[1024]" = torch.ops.aten.reshape.default(sum_154, [1024]);  sum_154 = None
        view_764: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_98, [4, 512, 1024]);  mm_98 = None
        add_251: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_250, view_764);  add_250 = view_764 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_626: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_251, primals_263);  primals_263 = None
        mul_627: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_626, 1024)
        sum_155: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_626, [2], True)
        mul_628: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_626, mul_242);  mul_626 = None
        sum_156: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_628, [2], True);  mul_628 = None
        mul_629: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_242, sum_156);  sum_156 = None
        sub_129: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_627, sum_155);  mul_627 = sum_155 = None
        sub_130: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_129, mul_629);  sub_129 = mul_629 = None
        mul_630: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_43, sub_130);  div_43 = sub_130 = None
        mul_631: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_251, mul_242);  mul_242 = None
        sum_157: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_631, [0, 1]);  mul_631 = None
        sum_158: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_251, [0, 1]);  add_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_25: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_48, torch.float32);  gt_48 = None
        mul_632: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_633: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_630, mul_632);  mul_632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_765: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_633, [2048, 1024]);  mul_633 = None
        permute_175: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
        permute_538: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
        mm_100: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_765, permute_538);  permute_538 = None
        permute_539: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_765, [1, 0])
        mm_101: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_539, view_350);  permute_539 = view_350 = None
        sum_159: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_765, [0], True);  view_765 = None
        view_766: "f32[1024]" = torch.ops.aten.reshape.default(sum_159, [1024]);  sum_159 = None
        view_767: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_100, [4, 512, 3072]);  mm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_349: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_94, [4, 512, 3072]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_238: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_349, 0.7071067811865476)
        erf_15: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_238);  mul_238 = None
        add_130: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_635: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_130, 0.5);  add_130 = None
        mul_636: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_349, view_349)
        mul_637: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_636, -0.5);  mul_636 = None
        exp_35: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_637);  mul_637 = None
        mul_638: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
        mul_639: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_349, mul_638);  view_349 = mul_638 = None
        add_253: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_635, mul_639);  mul_635 = mul_639 = None
        mul_640: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_767, add_253);  view_767 = add_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_768: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_640, [2048, 3072]);  mul_640 = None
        permute_174: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
        permute_542: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        mm_102: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_768, permute_542);  permute_542 = None
        permute_543: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_768, [1, 0])
        mm_103: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_543, view_348);  permute_543 = view_348 = None
        sum_160: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_768, [0], True);  view_768 = None
        view_769: "f32[3072]" = torch.ops.aten.reshape.default(sum_160, [3072]);  sum_160 = None
        view_770: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_102, [4, 512, 1024]);  mm_102 = None
        add_254: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_630, view_770);  mul_630 = view_770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_642: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_254, primals_257);  primals_257 = None
        mul_643: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_642, 1024)
        sum_161: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_642, [2], True)
        mul_644: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_642, mul_235);  mul_642 = None
        sum_162: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_644, [2], True);  mul_644 = None
        mul_645: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_235, sum_162);  sum_162 = None
        sub_132: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_643, sum_161);  mul_643 = sum_161 = None
        sub_133: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_132, mul_645);  sub_132 = mul_645 = None
        mul_646: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_44, sub_133);  div_44 = sub_133 = None
        mul_647: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_254, mul_235);  mul_235 = None
        sum_163: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_647, [0, 1]);  mul_647 = None
        sum_164: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_254, [0, 1]);  add_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_26: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_47, torch.float32);  gt_47 = None
        mul_648: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_649: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_646, mul_648);  mul_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_771: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_649, [2048, 1024]);  mul_649 = None
        permute_173: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
        permute_546: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        mm_104: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_771, permute_546);  permute_546 = None
        permute_547: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_771, [1, 0])
        mm_105: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_547, view_346);  permute_547 = view_346 = None
        sum_165: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_771, [0], True);  view_771 = None
        view_772: "f32[1024]" = torch.ops.aten.reshape.default(sum_165, [1024]);  sum_165 = None
        view_773: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_104, [4, 512, 1024]);  mm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_774: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_773, [4, 512, 16, 64]);  view_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_550: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_154: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_550, memory_format = torch.contiguous_format);  permute_550 = None
        view_775: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_154, [64, 512, 64]);  clone_154 = None
        bmm_80: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_551, view_775);  permute_551 = None
        bmm_81: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_775, permute_552);  view_775 = permute_552 = None
        view_776: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_80, [4, 16, 512, 64]);  bmm_80 = None
        view_777: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_81, [4, 16, 512, 512]);  bmm_81 = None
        convert_element_type_27: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_46, torch.float32);  gt_46 = None
        mul_650: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
        mul_651: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_777, mul_650);  view_777 = mul_650 = None
        mul_652: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_651, where_31);  mul_651 = None
        sum_166: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_652, [-1], True)
        neg_9: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_31);  where_31 = None
        fma_8: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_9, sum_166, mul_652);  neg_9 = sum_166 = mul_652 = None
        view_778: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_8, [64, 512, 512]);  fma_8 = None
        bmm_82: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_553, view_778);  permute_553 = None
        bmm_83: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_778, permute_554);  view_778 = permute_554 = None
        view_779: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_82, [4, 16, 64, 512]);  bmm_82 = None
        view_780: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_83, [4, 16, 512, 64]);  bmm_83 = None
        mul_653: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_779, 0.3535533905932738);  view_779 = None
        permute_555: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_653, [0, 1, 3, 2]);  mul_653 = None
        mul_654: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_780, 0.3535533905932738);  view_780 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_556: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_776, [0, 2, 1, 3]);  view_776 = None
        clone_156: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_556, memory_format = torch.contiguous_format);  permute_556 = None
        view_781: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_156, [4, 512, 1024]);  clone_156 = None
        view_782: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_781, [2048, 1024]);  view_781 = None
        permute_169: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_253, [1, 0]);  primals_253 = None
        permute_557: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
        mm_106: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_782, permute_557);  permute_557 = None
        permute_558: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_782, [1, 0])
        mm_107: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_558, view_330);  permute_558 = None
        sum_167: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_782, [0], True);  view_782 = None
        view_783: "f32[1024]" = torch.ops.aten.reshape.default(sum_167, [1024]);  sum_167 = None
        view_784: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_106, [4, 512, 1024]);  mm_106 = None
        add_255: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_646, view_784);  mul_646 = view_784 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_561: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_555, [0, 2, 1, 3]);  permute_555 = None
        view_785: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_561, [4, 512, 1024]);  permute_561 = None
        clone_157: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_785, memory_format = torch.contiguous_format);  view_785 = None
        view_786: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_157, [2048, 1024]);  clone_157 = None
        permute_167: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_251, [1, 0]);  primals_251 = None
        permute_562: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        mm_108: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_786, permute_562);  permute_562 = None
        permute_563: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_786, [1, 0])
        mm_109: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_563, view_330);  permute_563 = None
        sum_168: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_786, [0], True);  view_786 = None
        view_787: "f32[1024]" = torch.ops.aten.reshape.default(sum_168, [1024]);  sum_168 = None
        view_788: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_108, [4, 512, 1024]);  mm_108 = None
        add_256: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_255, view_788);  add_255 = view_788 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_566: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_654, [0, 2, 1, 3]);  mul_654 = None
        clone_158: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_566, memory_format = torch.contiguous_format);  permute_566 = None
        view_789: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_158, [4, 512, 1024]);  clone_158 = None
        view_790: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_789, [2048, 1024]);  view_789 = None
        permute_165: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
        permute_567: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        mm_110: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_790, permute_567);  permute_567 = None
        permute_568: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_790, [1, 0])
        mm_111: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_568, view_330);  permute_568 = view_330 = None
        sum_169: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_790, [0], True);  view_790 = None
        view_791: "f32[1024]" = torch.ops.aten.reshape.default(sum_169, [1024]);  sum_169 = None
        view_792: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_110, [4, 512, 1024]);  mm_110 = None
        add_257: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_256, view_792);  add_256 = view_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_656: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_257, primals_247);  primals_247 = None
        mul_657: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_656, 1024)
        sum_170: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_656, [2], True)
        mul_658: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_656, mul_227);  mul_656 = None
        sum_171: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_658, [2], True);  mul_658 = None
        mul_659: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_227, sum_171);  sum_171 = None
        sub_135: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_657, sum_170);  mul_657 = sum_170 = None
        sub_136: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_135, mul_659);  sub_135 = mul_659 = None
        mul_660: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_45, sub_136);  div_45 = sub_136 = None
        mul_661: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_257, mul_227);  mul_227 = None
        sum_172: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 1]);  mul_661 = None
        sum_173: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_257, [0, 1]);  add_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_28: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_45, torch.float32);  gt_45 = None
        mul_662: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
        mul_663: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_660, mul_662);  mul_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_793: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_663, [2048, 1024]);  mul_663 = None
        permute_164: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
        permute_571: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
        mm_112: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_793, permute_571);  permute_571 = None
        permute_572: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_793, [1, 0])
        mm_113: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_572, view_328);  permute_572 = view_328 = None
        sum_174: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_793, [0], True);  view_793 = None
        view_794: "f32[1024]" = torch.ops.aten.reshape.default(sum_174, [1024]);  sum_174 = None
        view_795: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_112, [4, 512, 3072]);  mm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_327: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_88, [4, 512, 3072]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_223: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476)
        erf_14: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_223);  mul_223 = None
        add_122: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_665: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_122, 0.5);  add_122 = None
        mul_666: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_327, view_327)
        mul_667: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_666, -0.5);  mul_666 = None
        exp_36: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_667);  mul_667 = None
        mul_668: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
        mul_669: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_327, mul_668);  view_327 = mul_668 = None
        add_259: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_665, mul_669);  mul_665 = mul_669 = None
        mul_670: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_795, add_259);  view_795 = add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_796: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_670, [2048, 3072]);  mul_670 = None
        permute_163: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_243, [1, 0]);  primals_243 = None
        permute_575: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        mm_114: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_796, permute_575);  permute_575 = None
        permute_576: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_796, [1, 0])
        mm_115: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_576, view_326);  permute_576 = view_326 = None
        sum_175: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True);  view_796 = None
        view_797: "f32[3072]" = torch.ops.aten.reshape.default(sum_175, [3072]);  sum_175 = None
        view_798: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_114, [4, 512, 1024]);  mm_114 = None
        add_260: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_660, view_798);  mul_660 = view_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_672: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_260, primals_241);  primals_241 = None
        mul_673: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_672, 1024)
        sum_176: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_672, [2], True)
        mul_674: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_672, mul_220);  mul_672 = None
        sum_177: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_674, [2], True);  mul_674 = None
        mul_675: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_220, sum_177);  sum_177 = None
        sub_138: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_673, sum_176);  mul_673 = sum_176 = None
        sub_139: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_138, mul_675);  sub_138 = mul_675 = None
        mul_676: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_46, sub_139);  div_46 = sub_139 = None
        mul_677: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_260, mul_220);  mul_220 = None
        sum_178: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_677, [0, 1]);  mul_677 = None
        sum_179: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_260, [0, 1]);  add_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_29: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_44, torch.float32);  gt_44 = None
        mul_678: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
        mul_679: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_676, mul_678);  mul_678 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_799: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_679, [2048, 1024]);  mul_679 = None
        permute_162: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_239, [1, 0]);  primals_239 = None
        permute_579: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
        mm_116: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_799, permute_579);  permute_579 = None
        permute_580: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_799, [1, 0])
        mm_117: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_580, view_324);  permute_580 = view_324 = None
        sum_180: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_799, [0], True);  view_799 = None
        view_800: "f32[1024]" = torch.ops.aten.reshape.default(sum_180, [1024]);  sum_180 = None
        view_801: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_116, [4, 512, 1024]);  mm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_802: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_801, [4, 512, 16, 64]);  view_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_583: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_802, [0, 2, 1, 3]);  view_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_161: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_583, memory_format = torch.contiguous_format);  permute_583 = None
        view_803: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_161, [64, 512, 64]);  clone_161 = None
        bmm_84: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_584, view_803);  permute_584 = None
        bmm_85: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_803, permute_585);  view_803 = permute_585 = None
        view_804: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_84, [4, 16, 512, 64]);  bmm_84 = None
        view_805: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_85, [4, 16, 512, 512]);  bmm_85 = None
        convert_element_type_30: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_43, torch.float32);  gt_43 = None
        mul_680: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
        mul_681: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_805, mul_680);  view_805 = mul_680 = None
        mul_682: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_681, where_29);  mul_681 = None
        sum_181: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_682, [-1], True)
        neg_10: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_29);  where_29 = None
        fma_9: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_10, sum_181, mul_682);  neg_10 = sum_181 = mul_682 = None
        view_806: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_9, [64, 512, 512]);  fma_9 = None
        bmm_86: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_586, view_806);  permute_586 = None
        bmm_87: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_806, permute_587);  view_806 = permute_587 = None
        view_807: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_86, [4, 16, 64, 512]);  bmm_86 = None
        view_808: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_87, [4, 16, 512, 64]);  bmm_87 = None
        mul_683: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_807, 0.3535533905932738);  view_807 = None
        permute_588: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_683, [0, 1, 3, 2]);  mul_683 = None
        mul_684: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_808, 0.3535533905932738);  view_808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_589: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_804, [0, 2, 1, 3]);  view_804 = None
        clone_163: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_589, memory_format = torch.contiguous_format);  permute_589 = None
        view_809: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_163, [4, 512, 1024]);  clone_163 = None
        view_810: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_809, [2048, 1024]);  view_809 = None
        permute_158: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
        permute_590: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
        mm_118: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_810, permute_590);  permute_590 = None
        permute_591: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_810, [1, 0])
        mm_119: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_591, view_308);  permute_591 = None
        sum_182: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_810, [0], True);  view_810 = None
        view_811: "f32[1024]" = torch.ops.aten.reshape.default(sum_182, [1024]);  sum_182 = None
        view_812: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_118, [4, 512, 1024]);  mm_118 = None
        add_261: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_676, view_812);  mul_676 = view_812 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_594: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_588, [0, 2, 1, 3]);  permute_588 = None
        view_813: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_594, [4, 512, 1024]);  permute_594 = None
        clone_164: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_813, memory_format = torch.contiguous_format);  view_813 = None
        view_814: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_164, [2048, 1024]);  clone_164 = None
        permute_156: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
        permute_595: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        mm_120: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_814, permute_595);  permute_595 = None
        permute_596: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_814, [1, 0])
        mm_121: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_596, view_308);  permute_596 = None
        sum_183: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_814, [0], True);  view_814 = None
        view_815: "f32[1024]" = torch.ops.aten.reshape.default(sum_183, [1024]);  sum_183 = None
        view_816: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_120, [4, 512, 1024]);  mm_120 = None
        add_262: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_261, view_816);  add_261 = view_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_599: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_684, [0, 2, 1, 3]);  mul_684 = None
        clone_165: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_599, memory_format = torch.contiguous_format);  permute_599 = None
        view_817: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_165, [4, 512, 1024]);  clone_165 = None
        view_818: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_817, [2048, 1024]);  view_817 = None
        permute_154: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_233, [1, 0]);  primals_233 = None
        permute_600: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        mm_122: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_818, permute_600);  permute_600 = None
        permute_601: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_818, [1, 0])
        mm_123: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_601, view_308);  permute_601 = view_308 = None
        sum_184: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_818, [0], True);  view_818 = None
        view_819: "f32[1024]" = torch.ops.aten.reshape.default(sum_184, [1024]);  sum_184 = None
        view_820: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_122, [4, 512, 1024]);  mm_122 = None
        add_263: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_262, view_820);  add_262 = view_820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_686: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_263, primals_231);  primals_231 = None
        mul_687: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_686, 1024)
        sum_185: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_686, [2], True)
        mul_688: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_686, mul_212);  mul_686 = None
        sum_186: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_688, [2], True);  mul_688 = None
        mul_689: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_212, sum_186);  sum_186 = None
        sub_141: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_687, sum_185);  mul_687 = sum_185 = None
        sub_142: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_141, mul_689);  sub_141 = mul_689 = None
        mul_690: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_47, sub_142);  div_47 = sub_142 = None
        mul_691: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_263, mul_212);  mul_212 = None
        sum_187: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 1]);  mul_691 = None
        sum_188: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_31: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_42, torch.float32);  gt_42 = None
        mul_692: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
        mul_693: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_690, mul_692);  mul_692 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_821: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_693, [2048, 1024]);  mul_693 = None
        permute_153: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
        permute_604: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        mm_124: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_821, permute_604);  permute_604 = None
        permute_605: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_821, [1, 0])
        mm_125: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_605, view_306);  permute_605 = view_306 = None
        sum_189: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_821, [0], True);  view_821 = None
        view_822: "f32[1024]" = torch.ops.aten.reshape.default(sum_189, [1024]);  sum_189 = None
        view_823: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_124, [4, 512, 3072]);  mm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_305: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_82, [4, 512, 3072]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_208: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_305, 0.7071067811865476)
        erf_13: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_208);  mul_208 = None
        add_114: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_695: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_114, 0.5);  add_114 = None
        mul_696: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_305, view_305)
        mul_697: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_696, -0.5);  mul_696 = None
        exp_37: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_697);  mul_697 = None
        mul_698: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
        mul_699: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_305, mul_698);  view_305 = mul_698 = None
        add_265: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_695, mul_699);  mul_695 = mul_699 = None
        mul_700: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_823, add_265);  view_823 = add_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_824: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_700, [2048, 3072]);  mul_700 = None
        permute_152: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
        permute_608: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        mm_126: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_824, permute_608);  permute_608 = None
        permute_609: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_824, [1, 0])
        mm_127: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_609, view_304);  permute_609 = view_304 = None
        sum_190: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_824, [0], True);  view_824 = None
        view_825: "f32[3072]" = torch.ops.aten.reshape.default(sum_190, [3072]);  sum_190 = None
        view_826: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_126, [4, 512, 1024]);  mm_126 = None
        add_266: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_690, view_826);  mul_690 = view_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_702: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_266, primals_225);  primals_225 = None
        mul_703: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_702, 1024)
        sum_191: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_702, [2], True)
        mul_704: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_702, mul_205);  mul_702 = None
        sum_192: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_704, [2], True);  mul_704 = None
        mul_705: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_205, sum_192);  sum_192 = None
        sub_144: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_703, sum_191);  mul_703 = sum_191 = None
        sub_145: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_144, mul_705);  sub_144 = mul_705 = None
        mul_706: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_48, sub_145);  div_48 = sub_145 = None
        mul_707: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_266, mul_205);  mul_205 = None
        sum_193: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 1]);  mul_707 = None
        sum_194: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_32: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_41, torch.float32);  gt_41 = None
        mul_708: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
        mul_709: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_706, mul_708);  mul_708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_827: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_709, [2048, 1024]);  mul_709 = None
        permute_151: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_223, [1, 0]);  primals_223 = None
        permute_612: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        mm_128: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_827, permute_612);  permute_612 = None
        permute_613: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_827, [1, 0])
        mm_129: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_613, view_302);  permute_613 = view_302 = None
        sum_195: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_827, [0], True);  view_827 = None
        view_828: "f32[1024]" = torch.ops.aten.reshape.default(sum_195, [1024]);  sum_195 = None
        view_829: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_128, [4, 512, 1024]);  mm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_830: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_829, [4, 512, 16, 64]);  view_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_616: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_830, [0, 2, 1, 3]);  view_830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_168: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_616, memory_format = torch.contiguous_format);  permute_616 = None
        view_831: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_168, [64, 512, 64]);  clone_168 = None
        bmm_88: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_617, view_831);  permute_617 = None
        bmm_89: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_831, permute_618);  view_831 = permute_618 = None
        view_832: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_88, [4, 16, 512, 64]);  bmm_88 = None
        view_833: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_89, [4, 16, 512, 512]);  bmm_89 = None
        convert_element_type_33: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_40, torch.float32);  gt_40 = None
        mul_710: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
        mul_711: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_833, mul_710);  view_833 = mul_710 = None
        mul_712: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_711, where_27);  mul_711 = None
        sum_196: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_712, [-1], True)
        neg_11: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_27);  where_27 = None
        fma_10: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_11, sum_196, mul_712);  neg_11 = sum_196 = mul_712 = None
        view_834: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_10, [64, 512, 512]);  fma_10 = None
        bmm_90: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_619, view_834);  permute_619 = None
        bmm_91: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_834, permute_620);  view_834 = permute_620 = None
        view_835: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_90, [4, 16, 64, 512]);  bmm_90 = None
        view_836: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_91, [4, 16, 512, 64]);  bmm_91 = None
        mul_713: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_835, 0.3535533905932738);  view_835 = None
        permute_621: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_713, [0, 1, 3, 2]);  mul_713 = None
        mul_714: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_836, 0.3535533905932738);  view_836 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_622: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_832, [0, 2, 1, 3]);  view_832 = None
        clone_170: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_622, memory_format = torch.contiguous_format);  permute_622 = None
        view_837: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_170, [4, 512, 1024]);  clone_170 = None
        view_838: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_837, [2048, 1024]);  view_837 = None
        permute_147: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
        permute_623: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
        mm_130: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_838, permute_623);  permute_623 = None
        permute_624: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_838, [1, 0])
        mm_131: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_624, view_286);  permute_624 = None
        sum_197: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_838, [0], True);  view_838 = None
        view_839: "f32[1024]" = torch.ops.aten.reshape.default(sum_197, [1024]);  sum_197 = None
        view_840: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_130, [4, 512, 1024]);  mm_130 = None
        add_267: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_706, view_840);  mul_706 = view_840 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_627: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_621, [0, 2, 1, 3]);  permute_621 = None
        view_841: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_627, [4, 512, 1024]);  permute_627 = None
        clone_171: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_841, memory_format = torch.contiguous_format);  view_841 = None
        view_842: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_171, [2048, 1024]);  clone_171 = None
        permute_145: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
        permute_628: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        mm_132: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_842, permute_628);  permute_628 = None
        permute_629: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_842, [1, 0])
        mm_133: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_629, view_286);  permute_629 = None
        sum_198: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_842, [0], True);  view_842 = None
        view_843: "f32[1024]" = torch.ops.aten.reshape.default(sum_198, [1024]);  sum_198 = None
        view_844: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_132, [4, 512, 1024]);  mm_132 = None
        add_268: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_267, view_844);  add_267 = view_844 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_632: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_714, [0, 2, 1, 3]);  mul_714 = None
        clone_172: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_632, memory_format = torch.contiguous_format);  permute_632 = None
        view_845: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_172, [4, 512, 1024]);  clone_172 = None
        view_846: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_845, [2048, 1024]);  view_845 = None
        permute_143: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_217, [1, 0]);  primals_217 = None
        permute_633: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        mm_134: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_846, permute_633);  permute_633 = None
        permute_634: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_846, [1, 0])
        mm_135: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_634, view_286);  permute_634 = view_286 = None
        sum_199: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_846, [0], True);  view_846 = None
        view_847: "f32[1024]" = torch.ops.aten.reshape.default(sum_199, [1024]);  sum_199 = None
        view_848: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_134, [4, 512, 1024]);  mm_134 = None
        add_269: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_268, view_848);  add_268 = view_848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_716: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_269, primals_215);  primals_215 = None
        mul_717: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_716, 1024)
        sum_200: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_716, [2], True)
        mul_718: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_716, mul_197);  mul_716 = None
        sum_201: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_718, [2], True);  mul_718 = None
        mul_719: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_197, sum_201);  sum_201 = None
        sub_147: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_717, sum_200);  mul_717 = sum_200 = None
        sub_148: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_147, mul_719);  sub_147 = mul_719 = None
        mul_720: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_49, sub_148);  div_49 = sub_148 = None
        mul_721: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_269, mul_197);  mul_197 = None
        sum_202: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_721, [0, 1]);  mul_721 = None
        sum_203: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_269, [0, 1]);  add_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_34: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_39, torch.float32);  gt_39 = None
        mul_722: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
        mul_723: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_720, mul_722);  mul_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_849: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_723, [2048, 1024]);  mul_723 = None
        permute_142: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
        permute_637: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        mm_136: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_849, permute_637);  permute_637 = None
        permute_638: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_849, [1, 0])
        mm_137: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_638, view_284);  permute_638 = view_284 = None
        sum_204: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_849, [0], True);  view_849 = None
        view_850: "f32[1024]" = torch.ops.aten.reshape.default(sum_204, [1024]);  sum_204 = None
        view_851: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_136, [4, 512, 3072]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_283: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_76, [4, 512, 3072]);  addmm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_193: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_283, 0.7071067811865476)
        erf_12: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_193);  mul_193 = None
        add_106: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_725: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
        mul_726: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_283, view_283)
        mul_727: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_726, -0.5);  mul_726 = None
        exp_38: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_727);  mul_727 = None
        mul_728: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
        mul_729: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_283, mul_728);  view_283 = mul_728 = None
        add_271: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_725, mul_729);  mul_725 = mul_729 = None
        mul_730: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_851, add_271);  view_851 = add_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_852: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_730, [2048, 3072]);  mul_730 = None
        permute_141: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
        permute_641: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        mm_138: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_852, permute_641);  permute_641 = None
        permute_642: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_852, [1, 0])
        mm_139: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_642, view_282);  permute_642 = view_282 = None
        sum_205: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_852, [0], True);  view_852 = None
        view_853: "f32[3072]" = torch.ops.aten.reshape.default(sum_205, [3072]);  sum_205 = None
        view_854: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_138, [4, 512, 1024]);  mm_138 = None
        add_272: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_720, view_854);  mul_720 = view_854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_732: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_272, primals_209);  primals_209 = None
        mul_733: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_732, 1024)
        sum_206: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_732, [2], True)
        mul_734: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_732, mul_190);  mul_732 = None
        sum_207: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_734, [2], True);  mul_734 = None
        mul_735: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_190, sum_207);  sum_207 = None
        sub_150: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_733, sum_206);  mul_733 = sum_206 = None
        sub_151: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_150, mul_735);  sub_150 = mul_735 = None
        mul_736: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_50, sub_151);  div_50 = sub_151 = None
        mul_737: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_272, mul_190);  mul_190 = None
        sum_208: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_737, [0, 1]);  mul_737 = None
        sum_209: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_272, [0, 1]);  add_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_35: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_38, torch.float32);  gt_38 = None
        mul_738: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
        mul_739: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_736, mul_738);  mul_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_855: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_739, [2048, 1024]);  mul_739 = None
        permute_140: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
        permute_645: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        mm_140: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_855, permute_645);  permute_645 = None
        permute_646: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_855, [1, 0])
        mm_141: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_646, view_280);  permute_646 = view_280 = None
        sum_210: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_855, [0], True);  view_855 = None
        view_856: "f32[1024]" = torch.ops.aten.reshape.default(sum_210, [1024]);  sum_210 = None
        view_857: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_140, [4, 512, 1024]);  mm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_858: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_857, [4, 512, 16, 64]);  view_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_649: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_858, [0, 2, 1, 3]);  view_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_175: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_649, memory_format = torch.contiguous_format);  permute_649 = None
        view_859: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_175, [64, 512, 64]);  clone_175 = None
        bmm_92: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_650, view_859);  permute_650 = None
        bmm_93: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_859, permute_651);  view_859 = permute_651 = None
        view_860: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_92, [4, 16, 512, 64]);  bmm_92 = None
        view_861: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_93, [4, 16, 512, 512]);  bmm_93 = None
        convert_element_type_36: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_37, torch.float32);  gt_37 = None
        mul_740: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
        mul_741: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_861, mul_740);  view_861 = mul_740 = None
        mul_742: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_741, where_25);  mul_741 = None
        sum_211: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_742, [-1], True)
        neg_12: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_25);  where_25 = None
        fma_11: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_12, sum_211, mul_742);  neg_12 = sum_211 = mul_742 = None
        view_862: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_11, [64, 512, 512]);  fma_11 = None
        bmm_94: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_652, view_862);  permute_652 = None
        bmm_95: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_862, permute_653);  view_862 = permute_653 = None
        view_863: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_94, [4, 16, 64, 512]);  bmm_94 = None
        view_864: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_95, [4, 16, 512, 64]);  bmm_95 = None
        mul_743: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_863, 0.3535533905932738);  view_863 = None
        permute_654: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_743, [0, 1, 3, 2]);  mul_743 = None
        mul_744: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_864, 0.3535533905932738);  view_864 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_655: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_860, [0, 2, 1, 3]);  view_860 = None
        clone_177: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_655, memory_format = torch.contiguous_format);  permute_655 = None
        view_865: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_177, [4, 512, 1024]);  clone_177 = None
        view_866: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_865, [2048, 1024]);  view_865 = None
        permute_136: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
        permute_656: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        mm_142: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_866, permute_656);  permute_656 = None
        permute_657: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_866, [1, 0])
        mm_143: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_657, view_264);  permute_657 = None
        sum_212: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_866, [0], True);  view_866 = None
        view_867: "f32[1024]" = torch.ops.aten.reshape.default(sum_212, [1024]);  sum_212 = None
        view_868: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_142, [4, 512, 1024]);  mm_142 = None
        add_273: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_736, view_868);  mul_736 = view_868 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_660: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_654, [0, 2, 1, 3]);  permute_654 = None
        view_869: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_660, [4, 512, 1024]);  permute_660 = None
        clone_178: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_869, memory_format = torch.contiguous_format);  view_869 = None
        view_870: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_178, [2048, 1024]);  clone_178 = None
        permute_134: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
        permute_661: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        mm_144: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_870, permute_661);  permute_661 = None
        permute_662: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_870, [1, 0])
        mm_145: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_662, view_264);  permute_662 = None
        sum_213: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_870, [0], True);  view_870 = None
        view_871: "f32[1024]" = torch.ops.aten.reshape.default(sum_213, [1024]);  sum_213 = None
        view_872: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_144, [4, 512, 1024]);  mm_144 = None
        add_274: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_273, view_872);  add_273 = view_872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_665: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_744, [0, 2, 1, 3]);  mul_744 = None
        clone_179: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_665, memory_format = torch.contiguous_format);  permute_665 = None
        view_873: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_179, [4, 512, 1024]);  clone_179 = None
        view_874: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_873, [2048, 1024]);  view_873 = None
        permute_132: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
        permute_666: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        mm_146: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_874, permute_666);  permute_666 = None
        permute_667: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_874, [1, 0])
        mm_147: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_667, view_264);  permute_667 = view_264 = None
        sum_214: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_874, [0], True);  view_874 = None
        view_875: "f32[1024]" = torch.ops.aten.reshape.default(sum_214, [1024]);  sum_214 = None
        view_876: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_146, [4, 512, 1024]);  mm_146 = None
        add_275: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_274, view_876);  add_274 = view_876 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_746: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_275, primals_199);  primals_199 = None
        mul_747: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_746, 1024)
        sum_215: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_746, [2], True)
        mul_748: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_746, mul_182);  mul_746 = None
        sum_216: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_748, [2], True);  mul_748 = None
        mul_749: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_182, sum_216);  sum_216 = None
        sub_153: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_747, sum_215);  mul_747 = sum_215 = None
        sub_154: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_153, mul_749);  sub_153 = mul_749 = None
        mul_750: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_51, sub_154);  div_51 = sub_154 = None
        mul_751: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_275, mul_182);  mul_182 = None
        sum_217: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_751, [0, 1]);  mul_751 = None
        sum_218: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_275, [0, 1]);  add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_37: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_36, torch.float32);  gt_36 = None
        mul_752: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
        mul_753: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_750, mul_752);  mul_752 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_877: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_753, [2048, 1024]);  mul_753 = None
        permute_131: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_197, [1, 0]);  primals_197 = None
        permute_670: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        mm_148: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_877, permute_670);  permute_670 = None
        permute_671: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_877, [1, 0])
        mm_149: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_671, view_262);  permute_671 = view_262 = None
        sum_219: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_877, [0], True);  view_877 = None
        view_878: "f32[1024]" = torch.ops.aten.reshape.default(sum_219, [1024]);  sum_219 = None
        view_879: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_148, [4, 512, 3072]);  mm_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_261: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_70, [4, 512, 3072]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_178: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
        erf_11: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_178);  mul_178 = None
        add_98: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_755: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_98, 0.5);  add_98 = None
        mul_756: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, view_261)
        mul_757: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_756, -0.5);  mul_756 = None
        exp_39: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_757);  mul_757 = None
        mul_758: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
        mul_759: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, mul_758);  view_261 = mul_758 = None
        add_277: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_755, mul_759);  mul_755 = mul_759 = None
        mul_760: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_879, add_277);  view_879 = add_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_880: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_760, [2048, 3072]);  mul_760 = None
        permute_130: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
        permute_674: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        mm_150: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_880, permute_674);  permute_674 = None
        permute_675: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_880, [1, 0])
        mm_151: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_675, view_260);  permute_675 = view_260 = None
        sum_220: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_880, [0], True);  view_880 = None
        view_881: "f32[3072]" = torch.ops.aten.reshape.default(sum_220, [3072]);  sum_220 = None
        view_882: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_150, [4, 512, 1024]);  mm_150 = None
        add_278: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_750, view_882);  mul_750 = view_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_762: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_278, primals_193);  primals_193 = None
        mul_763: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_762, 1024)
        sum_221: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_762, [2], True)
        mul_764: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_762, mul_175);  mul_762 = None
        sum_222: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_764, [2], True);  mul_764 = None
        mul_765: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_175, sum_222);  sum_222 = None
        sub_156: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_763, sum_221);  mul_763 = sum_221 = None
        sub_157: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_156, mul_765);  sub_156 = mul_765 = None
        mul_766: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_52, sub_157);  div_52 = sub_157 = None
        mul_767: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_278, mul_175);  mul_175 = None
        sum_223: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_767, [0, 1]);  mul_767 = None
        sum_224: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 1]);  add_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_38: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_35, torch.float32);  gt_35 = None
        mul_768: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
        mul_769: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_766, mul_768);  mul_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_883: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_769, [2048, 1024]);  mul_769 = None
        permute_129: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
        permute_678: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        mm_152: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_883, permute_678);  permute_678 = None
        permute_679: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_883, [1, 0])
        mm_153: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_679, view_258);  permute_679 = view_258 = None
        sum_225: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_883, [0], True);  view_883 = None
        view_884: "f32[1024]" = torch.ops.aten.reshape.default(sum_225, [1024]);  sum_225 = None
        view_885: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_152, [4, 512, 1024]);  mm_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_886: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_885, [4, 512, 16, 64]);  view_885 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_682: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_886, [0, 2, 1, 3]);  view_886 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_182: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_682, memory_format = torch.contiguous_format);  permute_682 = None
        view_887: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_182, [64, 512, 64]);  clone_182 = None
        bmm_96: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_683, view_887);  permute_683 = None
        bmm_97: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_887, permute_684);  view_887 = permute_684 = None
        view_888: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_96, [4, 16, 512, 64]);  bmm_96 = None
        view_889: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_97, [4, 16, 512, 512]);  bmm_97 = None
        convert_element_type_39: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_34, torch.float32);  gt_34 = None
        mul_770: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.1111111111111112);  convert_element_type_39 = None
        mul_771: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_889, mul_770);  view_889 = mul_770 = None
        mul_772: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_771, where_23);  mul_771 = None
        sum_226: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_772, [-1], True)
        neg_13: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_23);  where_23 = None
        fma_12: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_13, sum_226, mul_772);  neg_13 = sum_226 = mul_772 = None
        view_890: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_12, [64, 512, 512]);  fma_12 = None
        bmm_98: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_685, view_890);  permute_685 = None
        bmm_99: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_890, permute_686);  view_890 = permute_686 = None
        view_891: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_98, [4, 16, 64, 512]);  bmm_98 = None
        view_892: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_99, [4, 16, 512, 64]);  bmm_99 = None
        mul_773: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_891, 0.3535533905932738);  view_891 = None
        permute_687: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_773, [0, 1, 3, 2]);  mul_773 = None
        mul_774: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_892, 0.3535533905932738);  view_892 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_688: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_888, [0, 2, 1, 3]);  view_888 = None
        clone_184: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_688, memory_format = torch.contiguous_format);  permute_688 = None
        view_893: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_184, [4, 512, 1024]);  clone_184 = None
        view_894: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_893, [2048, 1024]);  view_893 = None
        permute_125: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
        permute_689: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        mm_154: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_894, permute_689);  permute_689 = None
        permute_690: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_894, [1, 0])
        mm_155: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_690, view_242);  permute_690 = None
        sum_227: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_894, [0], True);  view_894 = None
        view_895: "f32[1024]" = torch.ops.aten.reshape.default(sum_227, [1024]);  sum_227 = None
        view_896: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_154, [4, 512, 1024]);  mm_154 = None
        add_279: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_766, view_896);  mul_766 = view_896 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_693: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_687, [0, 2, 1, 3]);  permute_687 = None
        view_897: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_693, [4, 512, 1024]);  permute_693 = None
        clone_185: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_897, memory_format = torch.contiguous_format);  view_897 = None
        view_898: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_185, [2048, 1024]);  clone_185 = None
        permute_123: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
        permute_694: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        mm_156: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_898, permute_694);  permute_694 = None
        permute_695: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_898, [1, 0])
        mm_157: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_695, view_242);  permute_695 = None
        sum_228: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_898, [0], True);  view_898 = None
        view_899: "f32[1024]" = torch.ops.aten.reshape.default(sum_228, [1024]);  sum_228 = None
        view_900: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_156, [4, 512, 1024]);  mm_156 = None
        add_280: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_279, view_900);  add_279 = view_900 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_698: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_774, [0, 2, 1, 3]);  mul_774 = None
        clone_186: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_698, memory_format = torch.contiguous_format);  permute_698 = None
        view_901: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_186, [4, 512, 1024]);  clone_186 = None
        view_902: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_901, [2048, 1024]);  view_901 = None
        permute_121: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
        permute_699: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_158: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_902, permute_699);  permute_699 = None
        permute_700: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_902, [1, 0])
        mm_159: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_700, view_242);  permute_700 = view_242 = None
        sum_229: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_902, [0], True);  view_902 = None
        view_903: "f32[1024]" = torch.ops.aten.reshape.default(sum_229, [1024]);  sum_229 = None
        view_904: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_158, [4, 512, 1024]);  mm_158 = None
        add_281: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_280, view_904);  add_280 = view_904 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_776: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_281, primals_183);  primals_183 = None
        mul_777: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_776, 1024)
        sum_230: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_776, [2], True)
        mul_778: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_776, mul_167);  mul_776 = None
        sum_231: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_778, [2], True);  mul_778 = None
        mul_779: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_167, sum_231);  sum_231 = None
        sub_159: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_777, sum_230);  mul_777 = sum_230 = None
        sub_160: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_159, mul_779);  sub_159 = mul_779 = None
        mul_780: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_53, sub_160);  div_53 = sub_160 = None
        mul_781: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_281, mul_167);  mul_167 = None
        sum_232: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_781, [0, 1]);  mul_781 = None
        sum_233: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_281, [0, 1]);  add_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_40: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_33, torch.float32);  gt_33 = None
        mul_782: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
        mul_783: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_780, mul_782);  mul_782 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_905: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_783, [2048, 1024]);  mul_783 = None
        permute_120: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
        permute_703: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        mm_160: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_905, permute_703);  permute_703 = None
        permute_704: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_905, [1, 0])
        mm_161: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_704, view_240);  permute_704 = view_240 = None
        sum_234: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_905, [0], True);  view_905 = None
        view_906: "f32[1024]" = torch.ops.aten.reshape.default(sum_234, [1024]);  sum_234 = None
        view_907: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_160, [4, 512, 3072]);  mm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_239: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_64, [4, 512, 3072]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_163: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
        erf_10: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
        add_90: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_785: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_90, 0.5);  add_90 = None
        mul_786: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, view_239)
        mul_787: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_786, -0.5);  mul_786 = None
        exp_40: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_787);  mul_787 = None
        mul_788: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
        mul_789: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, mul_788);  view_239 = mul_788 = None
        add_283: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_785, mul_789);  mul_785 = mul_789 = None
        mul_790: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_907, add_283);  view_907 = add_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_908: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_790, [2048, 3072]);  mul_790 = None
        permute_119: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
        permute_707: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        mm_162: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_908, permute_707);  permute_707 = None
        permute_708: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_908, [1, 0])
        mm_163: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_708, view_238);  permute_708 = view_238 = None
        sum_235: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_908, [0], True);  view_908 = None
        view_909: "f32[3072]" = torch.ops.aten.reshape.default(sum_235, [3072]);  sum_235 = None
        view_910: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_162, [4, 512, 1024]);  mm_162 = None
        add_284: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_780, view_910);  mul_780 = view_910 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_792: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_284, primals_177);  primals_177 = None
        mul_793: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_792, 1024)
        sum_236: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_792, [2], True)
        mul_794: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_792, mul_160);  mul_792 = None
        sum_237: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_794, [2], True);  mul_794 = None
        mul_795: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_160, sum_237);  sum_237 = None
        sub_162: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_793, sum_236);  mul_793 = sum_236 = None
        sub_163: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_162, mul_795);  sub_162 = mul_795 = None
        mul_796: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_54, sub_163);  div_54 = sub_163 = None
        mul_797: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_284, mul_160);  mul_160 = None
        sum_238: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_797, [0, 1]);  mul_797 = None
        sum_239: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_284, [0, 1]);  add_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_41: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_32, torch.float32);  gt_32 = None
        mul_798: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
        mul_799: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_796, mul_798);  mul_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_911: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_799, [2048, 1024]);  mul_799 = None
        permute_118: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
        permute_711: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        mm_164: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_911, permute_711);  permute_711 = None
        permute_712: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_911, [1, 0])
        mm_165: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_712, view_236);  permute_712 = view_236 = None
        sum_240: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_911, [0], True);  view_911 = None
        view_912: "f32[1024]" = torch.ops.aten.reshape.default(sum_240, [1024]);  sum_240 = None
        view_913: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_164, [4, 512, 1024]);  mm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_914: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_913, [4, 512, 16, 64]);  view_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_715: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_914, [0, 2, 1, 3]);  view_914 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_189: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_715, memory_format = torch.contiguous_format);  permute_715 = None
        view_915: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_189, [64, 512, 64]);  clone_189 = None
        bmm_100: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_716, view_915);  permute_716 = None
        bmm_101: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_915, permute_717);  view_915 = permute_717 = None
        view_916: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_100, [4, 16, 512, 64]);  bmm_100 = None
        view_917: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_101, [4, 16, 512, 512]);  bmm_101 = None
        convert_element_type_42: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_31, torch.float32);  gt_31 = None
        mul_800: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 1.1111111111111112);  convert_element_type_42 = None
        mul_801: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_917, mul_800);  view_917 = mul_800 = None
        mul_802: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_801, where_21);  mul_801 = None
        sum_241: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_802, [-1], True)
        neg_14: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_21);  where_21 = None
        fma_13: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_14, sum_241, mul_802);  neg_14 = sum_241 = mul_802 = None
        view_918: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_13, [64, 512, 512]);  fma_13 = None
        bmm_102: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_718, view_918);  permute_718 = None
        bmm_103: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_918, permute_719);  view_918 = permute_719 = None
        view_919: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_102, [4, 16, 64, 512]);  bmm_102 = None
        view_920: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_103, [4, 16, 512, 64]);  bmm_103 = None
        mul_803: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_919, 0.3535533905932738);  view_919 = None
        permute_720: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_803, [0, 1, 3, 2]);  mul_803 = None
        mul_804: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_920, 0.3535533905932738);  view_920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_721: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_916, [0, 2, 1, 3]);  view_916 = None
        clone_191: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_721, memory_format = torch.contiguous_format);  permute_721 = None
        view_921: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_191, [4, 512, 1024]);  clone_191 = None
        view_922: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_921, [2048, 1024]);  view_921 = None
        permute_114: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
        permute_722: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
        mm_166: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_922, permute_722);  permute_722 = None
        permute_723: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_922, [1, 0])
        mm_167: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_723, view_220);  permute_723 = None
        sum_242: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_922, [0], True);  view_922 = None
        view_923: "f32[1024]" = torch.ops.aten.reshape.default(sum_242, [1024]);  sum_242 = None
        view_924: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_166, [4, 512, 1024]);  mm_166 = None
        add_285: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_796, view_924);  mul_796 = view_924 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_726: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_720, [0, 2, 1, 3]);  permute_720 = None
        view_925: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_726, [4, 512, 1024]);  permute_726 = None
        clone_192: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_925, memory_format = torch.contiguous_format);  view_925 = None
        view_926: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_192, [2048, 1024]);  clone_192 = None
        permute_112: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
        permute_727: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        mm_168: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_926, permute_727);  permute_727 = None
        permute_728: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_926, [1, 0])
        mm_169: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_728, view_220);  permute_728 = None
        sum_243: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_926, [0], True);  view_926 = None
        view_927: "f32[1024]" = torch.ops.aten.reshape.default(sum_243, [1024]);  sum_243 = None
        view_928: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_168, [4, 512, 1024]);  mm_168 = None
        add_286: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_285, view_928);  add_285 = view_928 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_731: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_804, [0, 2, 1, 3]);  mul_804 = None
        clone_193: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_731, memory_format = torch.contiguous_format);  permute_731 = None
        view_929: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_193, [4, 512, 1024]);  clone_193 = None
        view_930: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_929, [2048, 1024]);  view_929 = None
        permute_110: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
        permute_732: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        mm_170: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_930, permute_732);  permute_732 = None
        permute_733: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_930, [1, 0])
        mm_171: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_733, view_220);  permute_733 = view_220 = None
        sum_244: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_930, [0], True);  view_930 = None
        view_931: "f32[1024]" = torch.ops.aten.reshape.default(sum_244, [1024]);  sum_244 = None
        view_932: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_170, [4, 512, 1024]);  mm_170 = None
        add_287: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_286, view_932);  add_286 = view_932 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_806: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_287, primals_167);  primals_167 = None
        mul_807: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_806, 1024)
        sum_245: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_806, [2], True)
        mul_808: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_806, mul_152);  mul_806 = None
        sum_246: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_808, [2], True);  mul_808 = None
        mul_809: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_152, sum_246);  sum_246 = None
        sub_165: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_807, sum_245);  mul_807 = sum_245 = None
        sub_166: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_165, mul_809);  sub_165 = mul_809 = None
        mul_810: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_55, sub_166);  div_55 = sub_166 = None
        mul_811: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_287, mul_152);  mul_152 = None
        sum_247: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_811, [0, 1]);  mul_811 = None
        sum_248: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_287, [0, 1]);  add_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_43: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_30, torch.float32);  gt_30 = None
        mul_812: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.1111111111111112);  convert_element_type_43 = None
        mul_813: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_810, mul_812);  mul_812 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_933: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_813, [2048, 1024]);  mul_813 = None
        permute_109: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
        permute_736: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        mm_172: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_933, permute_736);  permute_736 = None
        permute_737: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_933, [1, 0])
        mm_173: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_737, view_218);  permute_737 = view_218 = None
        sum_249: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_933, [0], True);  view_933 = None
        view_934: "f32[1024]" = torch.ops.aten.reshape.default(sum_249, [1024]);  sum_249 = None
        view_935: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_172, [4, 512, 3072]);  mm_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_217: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_58, [4, 512, 3072]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_148: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
        erf_9: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
        add_82: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_815: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_82, 0.5);  add_82 = None
        mul_816: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, view_217)
        mul_817: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_816, -0.5);  mul_816 = None
        exp_41: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_817);  mul_817 = None
        mul_818: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
        mul_819: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, mul_818);  view_217 = mul_818 = None
        add_289: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_815, mul_819);  mul_815 = mul_819 = None
        mul_820: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_935, add_289);  view_935 = add_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_936: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_820, [2048, 3072]);  mul_820 = None
        permute_108: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
        permute_740: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        mm_174: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_936, permute_740);  permute_740 = None
        permute_741: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_936, [1, 0])
        mm_175: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_741, view_216);  permute_741 = view_216 = None
        sum_250: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_936, [0], True);  view_936 = None
        view_937: "f32[3072]" = torch.ops.aten.reshape.default(sum_250, [3072]);  sum_250 = None
        view_938: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_174, [4, 512, 1024]);  mm_174 = None
        add_290: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_810, view_938);  mul_810 = view_938 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_822: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_290, primals_161);  primals_161 = None
        mul_823: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_822, 1024)
        sum_251: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_822, [2], True)
        mul_824: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_822, mul_145);  mul_822 = None
        sum_252: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_824, [2], True);  mul_824 = None
        mul_825: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_145, sum_252);  sum_252 = None
        sub_168: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_823, sum_251);  mul_823 = sum_251 = None
        sub_169: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_168, mul_825);  sub_168 = mul_825 = None
        mul_826: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_56, sub_169);  div_56 = sub_169 = None
        mul_827: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_290, mul_145);  mul_145 = None
        sum_253: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_827, [0, 1]);  mul_827 = None
        sum_254: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_290, [0, 1]);  add_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_44: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_29, torch.float32);  gt_29 = None
        mul_828: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1.1111111111111112);  convert_element_type_44 = None
        mul_829: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_826, mul_828);  mul_828 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_939: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_829, [2048, 1024]);  mul_829 = None
        permute_107: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
        permute_744: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        mm_176: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_939, permute_744);  permute_744 = None
        permute_745: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_939, [1, 0])
        mm_177: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_745, view_214);  permute_745 = view_214 = None
        sum_255: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_939, [0], True);  view_939 = None
        view_940: "f32[1024]" = torch.ops.aten.reshape.default(sum_255, [1024]);  sum_255 = None
        view_941: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_176, [4, 512, 1024]);  mm_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_942: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_941, [4, 512, 16, 64]);  view_941 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_748: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_942, [0, 2, 1, 3]);  view_942 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_196: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_748, memory_format = torch.contiguous_format);  permute_748 = None
        view_943: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_196, [64, 512, 64]);  clone_196 = None
        bmm_104: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_749, view_943);  permute_749 = None
        bmm_105: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_943, permute_750);  view_943 = permute_750 = None
        view_944: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_104, [4, 16, 512, 64]);  bmm_104 = None
        view_945: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_105, [4, 16, 512, 512]);  bmm_105 = None
        convert_element_type_45: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_28, torch.float32);  gt_28 = None
        mul_830: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 1.1111111111111112);  convert_element_type_45 = None
        mul_831: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_945, mul_830);  view_945 = mul_830 = None
        mul_832: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_831, where_19);  mul_831 = None
        sum_256: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_832, [-1], True)
        neg_15: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_19);  where_19 = None
        fma_14: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_15, sum_256, mul_832);  neg_15 = sum_256 = mul_832 = None
        view_946: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_14, [64, 512, 512]);  fma_14 = None
        bmm_106: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_751, view_946);  permute_751 = None
        bmm_107: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_946, permute_752);  view_946 = permute_752 = None
        view_947: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_106, [4, 16, 64, 512]);  bmm_106 = None
        view_948: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_107, [4, 16, 512, 64]);  bmm_107 = None
        mul_833: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_947, 0.3535533905932738);  view_947 = None
        permute_753: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_833, [0, 1, 3, 2]);  mul_833 = None
        mul_834: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_948, 0.3535533905932738);  view_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_754: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_944, [0, 2, 1, 3]);  view_944 = None
        clone_198: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_754, memory_format = torch.contiguous_format);  permute_754 = None
        view_949: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_198, [4, 512, 1024]);  clone_198 = None
        view_950: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_949, [2048, 1024]);  view_949 = None
        permute_103: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
        permute_755: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        mm_178: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_950, permute_755);  permute_755 = None
        permute_756: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_950, [1, 0])
        mm_179: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_756, view_198);  permute_756 = None
        sum_257: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_950, [0], True);  view_950 = None
        view_951: "f32[1024]" = torch.ops.aten.reshape.default(sum_257, [1024]);  sum_257 = None
        view_952: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_178, [4, 512, 1024]);  mm_178 = None
        add_291: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_826, view_952);  mul_826 = view_952 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_759: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_753, [0, 2, 1, 3]);  permute_753 = None
        view_953: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_759, [4, 512, 1024]);  permute_759 = None
        clone_199: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_953, memory_format = torch.contiguous_format);  view_953 = None
        view_954: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_199, [2048, 1024]);  clone_199 = None
        permute_101: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
        permute_760: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_180: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_954, permute_760);  permute_760 = None
        permute_761: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_954, [1, 0])
        mm_181: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_761, view_198);  permute_761 = None
        sum_258: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_954, [0], True);  view_954 = None
        view_955: "f32[1024]" = torch.ops.aten.reshape.default(sum_258, [1024]);  sum_258 = None
        view_956: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_180, [4, 512, 1024]);  mm_180 = None
        add_292: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_291, view_956);  add_291 = view_956 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_764: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_834, [0, 2, 1, 3]);  mul_834 = None
        clone_200: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_764, memory_format = torch.contiguous_format);  permute_764 = None
        view_957: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_200, [4, 512, 1024]);  clone_200 = None
        view_958: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_957, [2048, 1024]);  view_957 = None
        permute_99: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
        permute_765: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_182: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_958, permute_765);  permute_765 = None
        permute_766: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_958, [1, 0])
        mm_183: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_766, view_198);  permute_766 = view_198 = None
        sum_259: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_958, [0], True);  view_958 = None
        view_959: "f32[1024]" = torch.ops.aten.reshape.default(sum_259, [1024]);  sum_259 = None
        view_960: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_182, [4, 512, 1024]);  mm_182 = None
        add_293: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_292, view_960);  add_292 = view_960 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_836: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_293, primals_151);  primals_151 = None
        mul_837: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_836, 1024)
        sum_260: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_836, [2], True)
        mul_838: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_836, mul_137);  mul_836 = None
        sum_261: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_838, [2], True);  mul_838 = None
        mul_839: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_137, sum_261);  sum_261 = None
        sub_171: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_837, sum_260);  mul_837 = sum_260 = None
        sub_172: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_171, mul_839);  sub_171 = mul_839 = None
        mul_840: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_57, sub_172);  div_57 = sub_172 = None
        mul_841: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_293, mul_137);  mul_137 = None
        sum_262: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_841, [0, 1]);  mul_841 = None
        sum_263: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_293, [0, 1]);  add_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_46: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_842: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 1.1111111111111112);  convert_element_type_46 = None
        mul_843: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_840, mul_842);  mul_842 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_961: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_843, [2048, 1024]);  mul_843 = None
        permute_98: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
        permute_769: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        mm_184: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_961, permute_769);  permute_769 = None
        permute_770: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_961, [1, 0])
        mm_185: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_770, view_196);  permute_770 = view_196 = None
        sum_264: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_961, [0], True);  view_961 = None
        view_962: "f32[1024]" = torch.ops.aten.reshape.default(sum_264, [1024]);  sum_264 = None
        view_963: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_184, [4, 512, 3072]);  mm_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_195: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_52, [4, 512, 3072]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_133: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
        erf_8: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_133);  mul_133 = None
        add_74: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_845: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_74, 0.5);  add_74 = None
        mul_846: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, view_195)
        mul_847: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_846, -0.5);  mul_846 = None
        exp_42: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_847);  mul_847 = None
        mul_848: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
        mul_849: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, mul_848);  view_195 = mul_848 = None
        add_295: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_845, mul_849);  mul_845 = mul_849 = None
        mul_850: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_963, add_295);  view_963 = add_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_964: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_850, [2048, 3072]);  mul_850 = None
        permute_97: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
        permute_773: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_186: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_964, permute_773);  permute_773 = None
        permute_774: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_964, [1, 0])
        mm_187: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_774, view_194);  permute_774 = view_194 = None
        sum_265: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_964, [0], True);  view_964 = None
        view_965: "f32[3072]" = torch.ops.aten.reshape.default(sum_265, [3072]);  sum_265 = None
        view_966: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_186, [4, 512, 1024]);  mm_186 = None
        add_296: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_840, view_966);  mul_840 = view_966 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_852: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_296, primals_145);  primals_145 = None
        mul_853: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_852, 1024)
        sum_266: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_852, [2], True)
        mul_854: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_852, mul_130);  mul_852 = None
        sum_267: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_854, [2], True);  mul_854 = None
        mul_855: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_130, sum_267);  sum_267 = None
        sub_174: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_853, sum_266);  mul_853 = sum_266 = None
        sub_175: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_174, mul_855);  sub_174 = mul_855 = None
        mul_856: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_58, sub_175);  div_58 = sub_175 = None
        mul_857: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_296, mul_130);  mul_130 = None
        sum_268: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_857, [0, 1]);  mul_857 = None
        sum_269: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_296, [0, 1]);  add_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_47: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_858: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
        mul_859: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_856, mul_858);  mul_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_967: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_859, [2048, 1024]);  mul_859 = None
        permute_96: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
        permute_777: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        mm_188: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_967, permute_777);  permute_777 = None
        permute_778: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_967, [1, 0])
        mm_189: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_778, view_192);  permute_778 = view_192 = None
        sum_270: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_967, [0], True);  view_967 = None
        view_968: "f32[1024]" = torch.ops.aten.reshape.default(sum_270, [1024]);  sum_270 = None
        view_969: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_188, [4, 512, 1024]);  mm_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_970: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_969, [4, 512, 16, 64]);  view_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_781: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_970, [0, 2, 1, 3]);  view_970 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_203: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_781, memory_format = torch.contiguous_format);  permute_781 = None
        view_971: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_203, [64, 512, 64]);  clone_203 = None
        bmm_108: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_782, view_971);  permute_782 = None
        bmm_109: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_971, permute_783);  view_971 = permute_783 = None
        view_972: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_108, [4, 16, 512, 64]);  bmm_108 = None
        view_973: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_109, [4, 16, 512, 512]);  bmm_109 = None
        convert_element_type_48: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_860: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 1.1111111111111112);  convert_element_type_48 = None
        mul_861: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_973, mul_860);  view_973 = mul_860 = None
        mul_862: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_861, where_17);  mul_861 = None
        sum_271: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_862, [-1], True)
        neg_16: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_17);  where_17 = None
        fma_15: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_16, sum_271, mul_862);  neg_16 = sum_271 = mul_862 = None
        view_974: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_15, [64, 512, 512]);  fma_15 = None
        bmm_110: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_784, view_974);  permute_784 = None
        bmm_111: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_974, permute_785);  view_974 = permute_785 = None
        view_975: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_110, [4, 16, 64, 512]);  bmm_110 = None
        view_976: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_111, [4, 16, 512, 64]);  bmm_111 = None
        mul_863: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_975, 0.3535533905932738);  view_975 = None
        permute_786: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_863, [0, 1, 3, 2]);  mul_863 = None
        mul_864: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_976, 0.3535533905932738);  view_976 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_787: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_972, [0, 2, 1, 3]);  view_972 = None
        clone_205: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_787, memory_format = torch.contiguous_format);  permute_787 = None
        view_977: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_205, [4, 512, 1024]);  clone_205 = None
        view_978: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_977, [2048, 1024]);  view_977 = None
        permute_92: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        permute_788: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        mm_190: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_978, permute_788);  permute_788 = None
        permute_789: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_978, [1, 0])
        mm_191: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_789, view_176);  permute_789 = None
        sum_272: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_978, [0], True);  view_978 = None
        view_979: "f32[1024]" = torch.ops.aten.reshape.default(sum_272, [1024]);  sum_272 = None
        view_980: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_190, [4, 512, 1024]);  mm_190 = None
        add_297: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_856, view_980);  mul_856 = view_980 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_792: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_786, [0, 2, 1, 3]);  permute_786 = None
        view_981: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_792, [4, 512, 1024]);  permute_792 = None
        clone_206: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_981, memory_format = torch.contiguous_format);  view_981 = None
        view_982: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_206, [2048, 1024]);  clone_206 = None
        permute_90: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        permute_793: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        mm_192: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_982, permute_793);  permute_793 = None
        permute_794: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_982, [1, 0])
        mm_193: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_794, view_176);  permute_794 = None
        sum_273: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_982, [0], True);  view_982 = None
        view_983: "f32[1024]" = torch.ops.aten.reshape.default(sum_273, [1024]);  sum_273 = None
        view_984: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_192, [4, 512, 1024]);  mm_192 = None
        add_298: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_297, view_984);  add_297 = view_984 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_797: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_864, [0, 2, 1, 3]);  mul_864 = None
        clone_207: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_797, memory_format = torch.contiguous_format);  permute_797 = None
        view_985: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_207, [4, 512, 1024]);  clone_207 = None
        view_986: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_985, [2048, 1024]);  view_985 = None
        permute_88: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
        permute_798: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_194: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_986, permute_798);  permute_798 = None
        permute_799: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_986, [1, 0])
        mm_195: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_799, view_176);  permute_799 = view_176 = None
        sum_274: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_986, [0], True);  view_986 = None
        view_987: "f32[1024]" = torch.ops.aten.reshape.default(sum_274, [1024]);  sum_274 = None
        view_988: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_194, [4, 512, 1024]);  mm_194 = None
        add_299: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_298, view_988);  add_298 = view_988 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_866: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_299, primals_135);  primals_135 = None
        mul_867: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_866, 1024)
        sum_275: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_866, [2], True)
        mul_868: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_866, mul_122);  mul_866 = None
        sum_276: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_868, [2], True);  mul_868 = None
        mul_869: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_122, sum_276);  sum_276 = None
        sub_177: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_867, sum_275);  mul_867 = sum_275 = None
        sub_178: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_177, mul_869);  sub_177 = mul_869 = None
        mul_870: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_59, sub_178);  div_59 = sub_178 = None
        mul_871: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_299, mul_122);  mul_122 = None
        sum_277: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_871, [0, 1]);  mul_871 = None
        sum_278: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_299, [0, 1]);  add_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_49: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_872: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 1.1111111111111112);  convert_element_type_49 = None
        mul_873: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_870, mul_872);  mul_872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_989: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_873, [2048, 1024]);  mul_873 = None
        permute_87: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        permute_802: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_196: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_989, permute_802);  permute_802 = None
        permute_803: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_989, [1, 0])
        mm_197: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_803, view_174);  permute_803 = view_174 = None
        sum_279: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_989, [0], True);  view_989 = None
        view_990: "f32[1024]" = torch.ops.aten.reshape.default(sum_279, [1024]);  sum_279 = None
        view_991: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_196, [4, 512, 3072]);  mm_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_173: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_46, [4, 512, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_118: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
        erf_7: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
        add_66: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_875: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_66, 0.5);  add_66 = None
        mul_876: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, view_173)
        mul_877: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_876, -0.5);  mul_876 = None
        exp_43: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_877);  mul_877 = None
        mul_878: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
        mul_879: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, mul_878);  view_173 = mul_878 = None
        add_301: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_875, mul_879);  mul_875 = mul_879 = None
        mul_880: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_991, add_301);  view_991 = add_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_992: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_880, [2048, 3072]);  mul_880 = None
        permute_86: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
        permute_806: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_198: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_992, permute_806);  permute_806 = None
        permute_807: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_992, [1, 0])
        mm_199: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_807, view_172);  permute_807 = view_172 = None
        sum_280: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_992, [0], True);  view_992 = None
        view_993: "f32[3072]" = torch.ops.aten.reshape.default(sum_280, [3072]);  sum_280 = None
        view_994: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_198, [4, 512, 1024]);  mm_198 = None
        add_302: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_870, view_994);  mul_870 = view_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_882: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_302, primals_129);  primals_129 = None
        mul_883: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_882, 1024)
        sum_281: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_882, [2], True)
        mul_884: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_882, mul_115);  mul_882 = None
        sum_282: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_884, [2], True);  mul_884 = None
        mul_885: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_115, sum_282);  sum_282 = None
        sub_180: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_883, sum_281);  mul_883 = sum_281 = None
        sub_181: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_180, mul_885);  sub_180 = mul_885 = None
        mul_886: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_60, sub_181);  div_60 = sub_181 = None
        mul_887: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_302, mul_115);  mul_115 = None
        sum_283: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_887, [0, 1]);  mul_887 = None
        sum_284: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_302, [0, 1]);  add_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_50: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_888: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 1.1111111111111112);  convert_element_type_50 = None
        mul_889: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_886, mul_888);  mul_888 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_995: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_889, [2048, 1024]);  mul_889 = None
        permute_85: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        permute_810: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        mm_200: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_995, permute_810);  permute_810 = None
        permute_811: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_995, [1, 0])
        mm_201: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_811, view_170);  permute_811 = view_170 = None
        sum_285: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_995, [0], True);  view_995 = None
        view_996: "f32[1024]" = torch.ops.aten.reshape.default(sum_285, [1024]);  sum_285 = None
        view_997: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_200, [4, 512, 1024]);  mm_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_998: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_997, [4, 512, 16, 64]);  view_997 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_814: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_998, [0, 2, 1, 3]);  view_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_210: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_814, memory_format = torch.contiguous_format);  permute_814 = None
        view_999: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_210, [64, 512, 64]);  clone_210 = None
        bmm_112: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_815, view_999);  permute_815 = None
        bmm_113: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_999, permute_816);  view_999 = permute_816 = None
        view_1000: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_112, [4, 16, 512, 64]);  bmm_112 = None
        view_1001: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_113, [4, 16, 512, 512]);  bmm_113 = None
        convert_element_type_51: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_890: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_51, 1.1111111111111112);  convert_element_type_51 = None
        mul_891: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_1001, mul_890);  view_1001 = mul_890 = None
        mul_892: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_891, where_15);  mul_891 = None
        sum_286: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_892, [-1], True)
        neg_17: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_15);  where_15 = None
        fma_16: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_17, sum_286, mul_892);  neg_17 = sum_286 = mul_892 = None
        view_1002: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_16, [64, 512, 512]);  fma_16 = None
        bmm_114: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_817, view_1002);  permute_817 = None
        bmm_115: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_1002, permute_818);  view_1002 = permute_818 = None
        view_1003: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_114, [4, 16, 64, 512]);  bmm_114 = None
        view_1004: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_115, [4, 16, 512, 64]);  bmm_115 = None
        mul_893: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_1003, 0.3535533905932738);  view_1003 = None
        permute_819: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_893, [0, 1, 3, 2]);  mul_893 = None
        mul_894: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_1004, 0.3535533905932738);  view_1004 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_820: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_1000, [0, 2, 1, 3]);  view_1000 = None
        clone_212: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_820, memory_format = torch.contiguous_format);  permute_820 = None
        view_1005: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_212, [4, 512, 1024]);  clone_212 = None
        view_1006: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_1005, [2048, 1024]);  view_1005 = None
        permute_81: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
        permute_821: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        mm_202: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1006, permute_821);  permute_821 = None
        permute_822: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1006, [1, 0])
        mm_203: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_822, view_154);  permute_822 = None
        sum_287: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1006, [0], True);  view_1006 = None
        view_1007: "f32[1024]" = torch.ops.aten.reshape.default(sum_287, [1024]);  sum_287 = None
        view_1008: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_202, [4, 512, 1024]);  mm_202 = None
        add_303: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_886, view_1008);  mul_886 = view_1008 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_825: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_819, [0, 2, 1, 3]);  permute_819 = None
        view_1009: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_825, [4, 512, 1024]);  permute_825 = None
        clone_213: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_1009, memory_format = torch.contiguous_format);  view_1009 = None
        view_1010: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_213, [2048, 1024]);  clone_213 = None
        permute_79: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
        permute_826: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        mm_204: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1010, permute_826);  permute_826 = None
        permute_827: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1010, [1, 0])
        mm_205: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_827, view_154);  permute_827 = None
        sum_288: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1010, [0], True);  view_1010 = None
        view_1011: "f32[1024]" = torch.ops.aten.reshape.default(sum_288, [1024]);  sum_288 = None
        view_1012: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_204, [4, 512, 1024]);  mm_204 = None
        add_304: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_303, view_1012);  add_303 = view_1012 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_830: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_894, [0, 2, 1, 3]);  mul_894 = None
        clone_214: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_830, memory_format = torch.contiguous_format);  permute_830 = None
        view_1013: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_214, [4, 512, 1024]);  clone_214 = None
        view_1014: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_1013, [2048, 1024]);  view_1013 = None
        permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
        permute_831: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_206: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1014, permute_831);  permute_831 = None
        permute_832: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1014, [1, 0])
        mm_207: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_832, view_154);  permute_832 = view_154 = None
        sum_289: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True);  view_1014 = None
        view_1015: "f32[1024]" = torch.ops.aten.reshape.default(sum_289, [1024]);  sum_289 = None
        view_1016: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_206, [4, 512, 1024]);  mm_206 = None
        add_305: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_304, view_1016);  add_304 = view_1016 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_896: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_305, primals_119);  primals_119 = None
        mul_897: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_896, 1024)
        sum_290: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_896, [2], True)
        mul_898: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_896, mul_107);  mul_896 = None
        sum_291: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_898, [2], True);  mul_898 = None
        mul_899: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_107, sum_291);  sum_291 = None
        sub_183: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_897, sum_290);  mul_897 = sum_290 = None
        sub_184: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_183, mul_899);  sub_183 = mul_899 = None
        mul_900: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_61, sub_184);  div_61 = sub_184 = None
        mul_901: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_305, mul_107);  mul_107 = None
        sum_292: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_901, [0, 1]);  mul_901 = None
        sum_293: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_305, [0, 1]);  add_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_52: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_902: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
        mul_903: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_900, mul_902);  mul_902 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_1017: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_903, [2048, 1024]);  mul_903 = None
        permute_76: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
        permute_835: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_208: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_1017, permute_835);  permute_835 = None
        permute_836: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1017, [1, 0])
        mm_209: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_836, view_152);  permute_836 = view_152 = None
        sum_294: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1017, [0], True);  view_1017 = None
        view_1018: "f32[1024]" = torch.ops.aten.reshape.default(sum_294, [1024]);  sum_294 = None
        view_1019: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_208, [4, 512, 3072]);  mm_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_151: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_40, [4, 512, 3072]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_103: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
        erf_6: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_58: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_905: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_58, 0.5);  add_58 = None
        mul_906: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, view_151)
        mul_907: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_906, -0.5);  mul_906 = None
        exp_44: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_907);  mul_907 = None
        mul_908: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
        mul_909: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, mul_908);  view_151 = mul_908 = None
        add_307: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_905, mul_909);  mul_905 = mul_909 = None
        mul_910: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_1019, add_307);  view_1019 = add_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_1020: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_910, [2048, 3072]);  mul_910 = None
        permute_75: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        permute_839: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_210: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1020, permute_839);  permute_839 = None
        permute_840: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_1020, [1, 0])
        mm_211: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_840, view_150);  permute_840 = view_150 = None
        sum_295: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1020, [0], True);  view_1020 = None
        view_1021: "f32[3072]" = torch.ops.aten.reshape.default(sum_295, [3072]);  sum_295 = None
        view_1022: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_210, [4, 512, 1024]);  mm_210 = None
        add_308: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_900, view_1022);  mul_900 = view_1022 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_912: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_308, primals_113);  primals_113 = None
        mul_913: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_912, 1024)
        sum_296: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_912, [2], True)
        mul_914: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_912, mul_100);  mul_912 = None
        sum_297: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_914, [2], True);  mul_914 = None
        mul_915: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_100, sum_297);  sum_297 = None
        sub_186: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_913, sum_296);  mul_913 = sum_296 = None
        sub_187: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_186, mul_915);  sub_186 = mul_915 = None
        mul_916: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_62, sub_187);  div_62 = sub_187 = None
        mul_917: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_308, mul_100);  mul_100 = None
        sum_298: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_917, [0, 1]);  mul_917 = None
        sum_299: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_308, [0, 1]);  add_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_53: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_918: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.1111111111111112);  convert_element_type_53 = None
        mul_919: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_916, mul_918);  mul_918 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_1023: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_919, [2048, 1024]);  mul_919 = None
        permute_74: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
        permute_843: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        mm_212: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1023, permute_843);  permute_843 = None
        permute_844: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1023, [1, 0])
        mm_213: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_844, view_148);  permute_844 = view_148 = None
        sum_300: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1023, [0], True);  view_1023 = None
        view_1024: "f32[1024]" = torch.ops.aten.reshape.default(sum_300, [1024]);  sum_300 = None
        view_1025: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_212, [4, 512, 1024]);  mm_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_1026: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1025, [4, 512, 16, 64]);  view_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_847: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_1026, [0, 2, 1, 3]);  view_1026 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_217: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_847, memory_format = torch.contiguous_format);  permute_847 = None
        view_1027: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_217, [64, 512, 64]);  clone_217 = None
        bmm_116: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_848, view_1027);  permute_848 = None
        bmm_117: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_1027, permute_849);  view_1027 = permute_849 = None
        view_1028: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_116, [4, 16, 512, 64]);  bmm_116 = None
        view_1029: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_117, [4, 16, 512, 512]);  bmm_117 = None
        convert_element_type_54: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_19, torch.float32);  gt_19 = None
        mul_920: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1.1111111111111112);  convert_element_type_54 = None
        mul_921: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_1029, mul_920);  view_1029 = mul_920 = None
        mul_922: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_921, where_13);  mul_921 = None
        sum_301: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_922, [-1], True)
        neg_18: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_13);  where_13 = None
        fma_17: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_18, sum_301, mul_922);  neg_18 = sum_301 = mul_922 = None
        view_1030: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_17, [64, 512, 512]);  fma_17 = None
        bmm_118: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_850, view_1030);  permute_850 = None
        bmm_119: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_1030, permute_851);  view_1030 = permute_851 = None
        view_1031: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_118, [4, 16, 64, 512]);  bmm_118 = None
        view_1032: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_119, [4, 16, 512, 64]);  bmm_119 = None
        mul_923: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_1031, 0.3535533905932738);  view_1031 = None
        permute_852: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_923, [0, 1, 3, 2]);  mul_923 = None
        mul_924: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_1032, 0.3535533905932738);  view_1032 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_853: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_1028, [0, 2, 1, 3]);  view_1028 = None
        clone_219: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_853, memory_format = torch.contiguous_format);  permute_853 = None
        view_1033: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_219, [4, 512, 1024]);  clone_219 = None
        view_1034: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_1033, [2048, 1024]);  view_1033 = None
        permute_70: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        permute_854: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        mm_214: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1034, permute_854);  permute_854 = None
        permute_855: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1034, [1, 0])
        mm_215: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_855, view_132);  permute_855 = None
        sum_302: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1034, [0], True);  view_1034 = None
        view_1035: "f32[1024]" = torch.ops.aten.reshape.default(sum_302, [1024]);  sum_302 = None
        view_1036: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_214, [4, 512, 1024]);  mm_214 = None
        add_309: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_916, view_1036);  mul_916 = view_1036 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_858: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_852, [0, 2, 1, 3]);  permute_852 = None
        view_1037: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_858, [4, 512, 1024]);  permute_858 = None
        clone_220: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_1037, memory_format = torch.contiguous_format);  view_1037 = None
        view_1038: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_220, [2048, 1024]);  clone_220 = None
        permute_68: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
        permute_859: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        mm_216: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1038, permute_859);  permute_859 = None
        permute_860: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1038, [1, 0])
        mm_217: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_860, view_132);  permute_860 = None
        sum_303: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1038, [0], True);  view_1038 = None
        view_1039: "f32[1024]" = torch.ops.aten.reshape.default(sum_303, [1024]);  sum_303 = None
        view_1040: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_216, [4, 512, 1024]);  mm_216 = None
        add_310: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_309, view_1040);  add_309 = view_1040 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_863: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_924, [0, 2, 1, 3]);  mul_924 = None
        clone_221: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_863, memory_format = torch.contiguous_format);  permute_863 = None
        view_1041: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_221, [4, 512, 1024]);  clone_221 = None
        view_1042: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_1041, [2048, 1024]);  view_1041 = None
        permute_66: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        permute_864: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_218: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1042, permute_864);  permute_864 = None
        permute_865: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1042, [1, 0])
        mm_219: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_865, view_132);  permute_865 = view_132 = None
        sum_304: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1042, [0], True);  view_1042 = None
        view_1043: "f32[1024]" = torch.ops.aten.reshape.default(sum_304, [1024]);  sum_304 = None
        view_1044: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_218, [4, 512, 1024]);  mm_218 = None
        add_311: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_310, view_1044);  add_310 = view_1044 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_926: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_311, primals_103);  primals_103 = None
        mul_927: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_926, 1024)
        sum_305: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_926, [2], True)
        mul_928: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_926, mul_92);  mul_926 = None
        sum_306: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_928, [2], True);  mul_928 = None
        mul_929: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_92, sum_306);  sum_306 = None
        sub_189: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_927, sum_305);  mul_927 = sum_305 = None
        sub_190: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_189, mul_929);  sub_189 = mul_929 = None
        mul_930: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_63, sub_190);  div_63 = sub_190 = None
        mul_931: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_311, mul_92);  mul_92 = None
        sum_307: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_931, [0, 1]);  mul_931 = None
        sum_308: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_311, [0, 1]);  add_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_55: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_932: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.1111111111111112);  convert_element_type_55 = None
        mul_933: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_930, mul_932);  mul_932 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_1045: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_933, [2048, 1024]);  mul_933 = None
        permute_65: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        permute_868: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_220: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_1045, permute_868);  permute_868 = None
        permute_869: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1045, [1, 0])
        mm_221: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_869, view_130);  permute_869 = view_130 = None
        sum_309: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1045, [0], True);  view_1045 = None
        view_1046: "f32[1024]" = torch.ops.aten.reshape.default(sum_309, [1024]);  sum_309 = None
        view_1047: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_220, [4, 512, 3072]);  mm_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_129: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [4, 512, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_88: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
        erf_5: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
        add_50: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_935: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_50, 0.5);  add_50 = None
        mul_936: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, view_129)
        mul_937: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_936, -0.5);  mul_936 = None
        exp_45: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_937);  mul_937 = None
        mul_938: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
        mul_939: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, mul_938);  view_129 = mul_938 = None
        add_313: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_935, mul_939);  mul_935 = mul_939 = None
        mul_940: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_1047, add_313);  view_1047 = add_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_1048: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_940, [2048, 3072]);  mul_940 = None
        permute_64: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        permute_872: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_222: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1048, permute_872);  permute_872 = None
        permute_873: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_1048, [1, 0])
        mm_223: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_873, view_128);  permute_873 = view_128 = None
        sum_310: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1048, [0], True);  view_1048 = None
        view_1049: "f32[3072]" = torch.ops.aten.reshape.default(sum_310, [3072]);  sum_310 = None
        view_1050: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_222, [4, 512, 1024]);  mm_222 = None
        add_314: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_930, view_1050);  mul_930 = view_1050 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_942: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_314, primals_97);  primals_97 = None
        mul_943: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_942, 1024)
        sum_311: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_942, [2], True)
        mul_944: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_942, mul_85);  mul_942 = None
        sum_312: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_944, [2], True);  mul_944 = None
        mul_945: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_85, sum_312);  sum_312 = None
        sub_192: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_943, sum_311);  mul_943 = sum_311 = None
        sub_193: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_192, mul_945);  sub_192 = mul_945 = None
        mul_946: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_64, sub_193);  div_64 = sub_193 = None
        mul_947: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_314, mul_85);  mul_85 = None
        sum_313: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_947, [0, 1]);  mul_947 = None
        sum_314: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_314, [0, 1]);  add_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_56: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_948: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 1.1111111111111112);  convert_element_type_56 = None
        mul_949: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_946, mul_948);  mul_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_1051: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_949, [2048, 1024]);  mul_949 = None
        permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
        permute_876: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_224: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1051, permute_876);  permute_876 = None
        permute_877: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1051, [1, 0])
        mm_225: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_877, view_126);  permute_877 = view_126 = None
        sum_315: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1051, [0], True);  view_1051 = None
        view_1052: "f32[1024]" = torch.ops.aten.reshape.default(sum_315, [1024]);  sum_315 = None
        view_1053: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_224, [4, 512, 1024]);  mm_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_1054: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1053, [4, 512, 16, 64]);  view_1053 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_880: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_1054, [0, 2, 1, 3]);  view_1054 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_224: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_880, memory_format = torch.contiguous_format);  permute_880 = None
        view_1055: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_224, [64, 512, 64]);  clone_224 = None
        bmm_120: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_881, view_1055);  permute_881 = None
        bmm_121: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_1055, permute_882);  view_1055 = permute_882 = None
        view_1056: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_120, [4, 16, 512, 64]);  bmm_120 = None
        view_1057: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_121, [4, 16, 512, 512]);  bmm_121 = None
        convert_element_type_57: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_950: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_57, 1.1111111111111112);  convert_element_type_57 = None
        mul_951: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_1057, mul_950);  view_1057 = mul_950 = None
        mul_952: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_951, where_11);  mul_951 = None
        sum_316: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_952, [-1], True)
        neg_19: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_11);  where_11 = None
        fma_18: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_19, sum_316, mul_952);  neg_19 = sum_316 = mul_952 = None
        view_1058: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_18, [64, 512, 512]);  fma_18 = None
        bmm_122: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_883, view_1058);  permute_883 = None
        bmm_123: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_1058, permute_884);  view_1058 = permute_884 = None
        view_1059: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_122, [4, 16, 64, 512]);  bmm_122 = None
        view_1060: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_123, [4, 16, 512, 64]);  bmm_123 = None
        mul_953: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_1059, 0.3535533905932738);  view_1059 = None
        permute_885: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_953, [0, 1, 3, 2]);  mul_953 = None
        mul_954: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_1060, 0.3535533905932738);  view_1060 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_886: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_1056, [0, 2, 1, 3]);  view_1056 = None
        clone_226: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_886, memory_format = torch.contiguous_format);  permute_886 = None
        view_1061: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_226, [4, 512, 1024]);  clone_226 = None
        view_1062: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_1061, [2048, 1024]);  view_1061 = None
        permute_59: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        permute_887: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        mm_226: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1062, permute_887);  permute_887 = None
        permute_888: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1062, [1, 0])
        mm_227: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_888, view_110);  permute_888 = None
        sum_317: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1062, [0], True);  view_1062 = None
        view_1063: "f32[1024]" = torch.ops.aten.reshape.default(sum_317, [1024]);  sum_317 = None
        view_1064: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_226, [4, 512, 1024]);  mm_226 = None
        add_315: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_946, view_1064);  mul_946 = view_1064 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_891: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_885, [0, 2, 1, 3]);  permute_885 = None
        view_1065: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_891, [4, 512, 1024]);  permute_891 = None
        clone_227: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_1065, memory_format = torch.contiguous_format);  view_1065 = None
        view_1066: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_227, [2048, 1024]);  clone_227 = None
        permute_57: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        permute_892: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_228: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1066, permute_892);  permute_892 = None
        permute_893: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1066, [1, 0])
        mm_229: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_893, view_110);  permute_893 = None
        sum_318: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1066, [0], True);  view_1066 = None
        view_1067: "f32[1024]" = torch.ops.aten.reshape.default(sum_318, [1024]);  sum_318 = None
        view_1068: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_228, [4, 512, 1024]);  mm_228 = None
        add_316: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_315, view_1068);  add_315 = view_1068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_896: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_954, [0, 2, 1, 3]);  mul_954 = None
        clone_228: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_896, memory_format = torch.contiguous_format);  permute_896 = None
        view_1069: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_228, [4, 512, 1024]);  clone_228 = None
        view_1070: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_1069, [2048, 1024]);  view_1069 = None
        permute_55: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
        permute_897: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_230: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1070, permute_897);  permute_897 = None
        permute_898: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1070, [1, 0])
        mm_231: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_898, view_110);  permute_898 = view_110 = None
        sum_319: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1070, [0], True);  view_1070 = None
        view_1071: "f32[1024]" = torch.ops.aten.reshape.default(sum_319, [1024]);  sum_319 = None
        view_1072: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_230, [4, 512, 1024]);  mm_230 = None
        add_317: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_316, view_1072);  add_316 = view_1072 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_956: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_317, primals_87);  primals_87 = None
        mul_957: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_956, 1024)
        sum_320: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_956, [2], True)
        mul_958: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_956, mul_77);  mul_956 = None
        sum_321: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_958, [2], True);  mul_958 = None
        mul_959: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_77, sum_321);  sum_321 = None
        sub_195: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_957, sum_320);  mul_957 = sum_320 = None
        sub_196: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_195, mul_959);  sub_195 = mul_959 = None
        mul_960: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_65, sub_196);  div_65 = sub_196 = None
        mul_961: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_317, mul_77);  mul_77 = None
        sum_322: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_961, [0, 1]);  mul_961 = None
        sum_323: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_317, [0, 1]);  add_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_58: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_962: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 1.1111111111111112);  convert_element_type_58 = None
        mul_963: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_960, mul_962);  mul_962 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_1073: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_963, [2048, 1024]);  mul_963 = None
        permute_54: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        permute_901: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_232: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_1073, permute_901);  permute_901 = None
        permute_902: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1073, [1, 0])
        mm_233: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_902, view_108);  permute_902 = view_108 = None
        sum_324: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1073, [0], True);  view_1073 = None
        view_1074: "f32[1024]" = torch.ops.aten.reshape.default(sum_324, [1024]);  sum_324 = None
        view_1075: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_232, [4, 512, 3072]);  mm_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_107: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [4, 512, 3072]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_73: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
        erf_4: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
        add_42: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_965: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_42, 0.5);  add_42 = None
        mul_966: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, view_107)
        mul_967: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_966, -0.5);  mul_966 = None
        exp_46: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_967);  mul_967 = None
        mul_968: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
        mul_969: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_968);  view_107 = mul_968 = None
        add_319: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_965, mul_969);  mul_965 = mul_969 = None
        mul_970: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_1075, add_319);  view_1075 = add_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_1076: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_970, [2048, 3072]);  mul_970 = None
        permute_53: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
        permute_905: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_234: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1076, permute_905);  permute_905 = None
        permute_906: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_1076, [1, 0])
        mm_235: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_906, view_106);  permute_906 = view_106 = None
        sum_325: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1076, [0], True);  view_1076 = None
        view_1077: "f32[3072]" = torch.ops.aten.reshape.default(sum_325, [3072]);  sum_325 = None
        view_1078: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_234, [4, 512, 1024]);  mm_234 = None
        add_320: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_960, view_1078);  mul_960 = view_1078 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_972: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_320, primals_81);  primals_81 = None
        mul_973: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_972, 1024)
        sum_326: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_972, [2], True)
        mul_974: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_972, mul_70);  mul_972 = None
        sum_327: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_974, [2], True);  mul_974 = None
        mul_975: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_70, sum_327);  sum_327 = None
        sub_198: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_973, sum_326);  mul_973 = sum_326 = None
        sub_199: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_198, mul_975);  sub_198 = mul_975 = None
        mul_976: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_66, sub_199);  div_66 = sub_199 = None
        mul_977: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_320, mul_70);  mul_70 = None
        sum_328: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_977, [0, 1]);  mul_977 = None
        sum_329: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_320, [0, 1]);  add_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_59: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_978: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 1.1111111111111112);  convert_element_type_59 = None
        mul_979: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_976, mul_978);  mul_978 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_1079: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_979, [2048, 1024]);  mul_979 = None
        permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        permute_909: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_236: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1079, permute_909);  permute_909 = None
        permute_910: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1079, [1, 0])
        mm_237: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_910, view_104);  permute_910 = view_104 = None
        sum_330: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1079, [0], True);  view_1079 = None
        view_1080: "f32[1024]" = torch.ops.aten.reshape.default(sum_330, [1024]);  sum_330 = None
        view_1081: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_236, [4, 512, 1024]);  mm_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_1082: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1081, [4, 512, 16, 64]);  view_1081 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_913: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_1082, [0, 2, 1, 3]);  view_1082 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_231: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_913, memory_format = torch.contiguous_format);  permute_913 = None
        view_1083: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_231, [64, 512, 64]);  clone_231 = None
        bmm_124: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_914, view_1083);  permute_914 = None
        bmm_125: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_1083, permute_915);  view_1083 = permute_915 = None
        view_1084: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_124, [4, 16, 512, 64]);  bmm_124 = None
        view_1085: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_125, [4, 16, 512, 512]);  bmm_125 = None
        convert_element_type_60: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_13, torch.float32);  gt_13 = None
        mul_980: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 1.1111111111111112);  convert_element_type_60 = None
        mul_981: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_1085, mul_980);  view_1085 = mul_980 = None
        mul_982: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_981, where_9);  mul_981 = None
        sum_331: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_982, [-1], True)
        neg_20: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_9);  where_9 = None
        fma_19: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_20, sum_331, mul_982);  neg_20 = sum_331 = mul_982 = None
        view_1086: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_19, [64, 512, 512]);  fma_19 = None
        bmm_126: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_916, view_1086);  permute_916 = None
        bmm_127: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_1086, permute_917);  view_1086 = permute_917 = None
        view_1087: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_126, [4, 16, 64, 512]);  bmm_126 = None
        view_1088: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_127, [4, 16, 512, 64]);  bmm_127 = None
        mul_983: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_1087, 0.3535533905932738);  view_1087 = None
        permute_918: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_983, [0, 1, 3, 2]);  mul_983 = None
        mul_984: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_1088, 0.3535533905932738);  view_1088 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_919: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_1084, [0, 2, 1, 3]);  view_1084 = None
        clone_233: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_919, memory_format = torch.contiguous_format);  permute_919 = None
        view_1089: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_233, [4, 512, 1024]);  clone_233 = None
        view_1090: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_1089, [2048, 1024]);  view_1089 = None
        permute_48: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        permute_920: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_238: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1090, permute_920);  permute_920 = None
        permute_921: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1090, [1, 0])
        mm_239: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_921, view_88);  permute_921 = None
        sum_332: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1090, [0], True);  view_1090 = None
        view_1091: "f32[1024]" = torch.ops.aten.reshape.default(sum_332, [1024]);  sum_332 = None
        view_1092: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_238, [4, 512, 1024]);  mm_238 = None
        add_321: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_976, view_1092);  mul_976 = view_1092 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_924: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_918, [0, 2, 1, 3]);  permute_918 = None
        view_1093: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_924, [4, 512, 1024]);  permute_924 = None
        clone_234: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_1093, memory_format = torch.contiguous_format);  view_1093 = None
        view_1094: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_234, [2048, 1024]);  clone_234 = None
        permute_46: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        permute_925: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_240: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1094, permute_925);  permute_925 = None
        permute_926: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1094, [1, 0])
        mm_241: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_926, view_88);  permute_926 = None
        sum_333: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1094, [0], True);  view_1094 = None
        view_1095: "f32[1024]" = torch.ops.aten.reshape.default(sum_333, [1024]);  sum_333 = None
        view_1096: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_240, [4, 512, 1024]);  mm_240 = None
        add_322: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_321, view_1096);  add_321 = view_1096 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_929: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_984, [0, 2, 1, 3]);  mul_984 = None
        clone_235: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_929, memory_format = torch.contiguous_format);  permute_929 = None
        view_1097: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_235, [4, 512, 1024]);  clone_235 = None
        view_1098: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_1097, [2048, 1024]);  view_1097 = None
        permute_44: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        permute_930: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_242: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1098, permute_930);  permute_930 = None
        permute_931: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1098, [1, 0])
        mm_243: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_931, view_88);  permute_931 = view_88 = None
        sum_334: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1098, [0], True);  view_1098 = None
        view_1099: "f32[1024]" = torch.ops.aten.reshape.default(sum_334, [1024]);  sum_334 = None
        view_1100: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_242, [4, 512, 1024]);  mm_242 = None
        add_323: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_322, view_1100);  add_322 = view_1100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_986: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_323, primals_71);  primals_71 = None
        mul_987: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_986, 1024)
        sum_335: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_986, [2], True)
        mul_988: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_986, mul_62);  mul_986 = None
        sum_336: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_988, [2], True);  mul_988 = None
        mul_989: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_62, sum_336);  sum_336 = None
        sub_201: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_987, sum_335);  mul_987 = sum_335 = None
        sub_202: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_201, mul_989);  sub_201 = mul_989 = None
        mul_990: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_67, sub_202);  div_67 = sub_202 = None
        mul_991: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_323, mul_62);  mul_62 = None
        sum_337: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_991, [0, 1]);  mul_991 = None
        sum_338: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_323, [0, 1]);  add_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_61: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_992: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 1.1111111111111112);  convert_element_type_61 = None
        mul_993: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_990, mul_992);  mul_992 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_1101: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_993, [2048, 1024]);  mul_993 = None
        permute_43: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        permute_934: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_244: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_1101, permute_934);  permute_934 = None
        permute_935: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1101, [1, 0])
        mm_245: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_935, view_86);  permute_935 = view_86 = None
        sum_339: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1101, [0], True);  view_1101 = None
        view_1102: "f32[1024]" = torch.ops.aten.reshape.default(sum_339, [1024]);  sum_339 = None
        view_1103: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_244, [4, 512, 3072]);  mm_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_85: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [4, 512, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_58: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
        erf_3: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
        add_34: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_995: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
        mul_996: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, view_85)
        mul_997: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_996, -0.5);  mul_996 = None
        exp_47: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_997);  mul_997 = None
        mul_998: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
        mul_999: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, mul_998);  view_85 = mul_998 = None
        add_325: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_995, mul_999);  mul_995 = mul_999 = None
        mul_1000: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_1103, add_325);  view_1103 = add_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_1104: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_1000, [2048, 3072]);  mul_1000 = None
        permute_42: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        permute_938: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_246: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1104, permute_938);  permute_938 = None
        permute_939: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_1104, [1, 0])
        mm_247: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_939, view_84);  permute_939 = view_84 = None
        sum_340: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1104, [0], True);  view_1104 = None
        view_1105: "f32[3072]" = torch.ops.aten.reshape.default(sum_340, [3072]);  sum_340 = None
        view_1106: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_246, [4, 512, 1024]);  mm_246 = None
        add_326: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_990, view_1106);  mul_990 = view_1106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_1002: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_326, primals_65);  primals_65 = None
        mul_1003: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1002, 1024)
        sum_341: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1002, [2], True)
        mul_1004: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1002, mul_55);  mul_1002 = None
        sum_342: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1004, [2], True);  mul_1004 = None
        mul_1005: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_55, sum_342);  sum_342 = None
        sub_204: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_1003, sum_341);  mul_1003 = sum_341 = None
        sub_205: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_204, mul_1005);  sub_204 = mul_1005 = None
        mul_1006: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_68, sub_205);  div_68 = sub_205 = None
        mul_1007: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_326, mul_55);  mul_55 = None
        sum_343: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1007, [0, 1]);  mul_1007 = None
        sum_344: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_326, [0, 1]);  add_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_62: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_1008: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
        mul_1009: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1006, mul_1008);  mul_1008 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_1107: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_1009, [2048, 1024]);  mul_1009 = None
        permute_41: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        permute_942: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_248: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1107, permute_942);  permute_942 = None
        permute_943: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1107, [1, 0])
        mm_249: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_943, view_82);  permute_943 = view_82 = None
        sum_345: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1107, [0], True);  view_1107 = None
        view_1108: "f32[1024]" = torch.ops.aten.reshape.default(sum_345, [1024]);  sum_345 = None
        view_1109: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_248, [4, 512, 1024]);  mm_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_1110: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1109, [4, 512, 16, 64]);  view_1109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_946: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_1110, [0, 2, 1, 3]);  view_1110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_238: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_946, memory_format = torch.contiguous_format);  permute_946 = None
        view_1111: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_238, [64, 512, 64]);  clone_238 = None
        bmm_128: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_947, view_1111);  permute_947 = None
        bmm_129: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_1111, permute_948);  view_1111 = permute_948 = None
        view_1112: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_128, [4, 16, 512, 64]);  bmm_128 = None
        view_1113: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_129, [4, 16, 512, 512]);  bmm_129 = None
        convert_element_type_63: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_1010: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.1111111111111112);  convert_element_type_63 = None
        mul_1011: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_1113, mul_1010);  view_1113 = mul_1010 = None
        mul_1012: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_1011, where_7);  mul_1011 = None
        sum_346: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1012, [-1], True)
        neg_21: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_7);  where_7 = None
        fma_20: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_21, sum_346, mul_1012);  neg_21 = sum_346 = mul_1012 = None
        view_1114: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_20, [64, 512, 512]);  fma_20 = None
        bmm_130: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_949, view_1114);  permute_949 = None
        bmm_131: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_1114, permute_950);  view_1114 = permute_950 = None
        view_1115: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_130, [4, 16, 64, 512]);  bmm_130 = None
        view_1116: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_131, [4, 16, 512, 64]);  bmm_131 = None
        mul_1013: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_1115, 0.3535533905932738);  view_1115 = None
        permute_951: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_1013, [0, 1, 3, 2]);  mul_1013 = None
        mul_1014: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_1116, 0.3535533905932738);  view_1116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_952: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_1112, [0, 2, 1, 3]);  view_1112 = None
        clone_240: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_952, memory_format = torch.contiguous_format);  permute_952 = None
        view_1117: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_240, [4, 512, 1024]);  clone_240 = None
        view_1118: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_1117, [2048, 1024]);  view_1117 = None
        permute_37: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        permute_953: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        mm_250: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1118, permute_953);  permute_953 = None
        permute_954: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1118, [1, 0])
        mm_251: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_954, view_66);  permute_954 = None
        sum_347: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1118, [0], True);  view_1118 = None
        view_1119: "f32[1024]" = torch.ops.aten.reshape.default(sum_347, [1024]);  sum_347 = None
        view_1120: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_250, [4, 512, 1024]);  mm_250 = None
        add_327: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_1006, view_1120);  mul_1006 = view_1120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_957: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_951, [0, 2, 1, 3]);  permute_951 = None
        view_1121: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_957, [4, 512, 1024]);  permute_957 = None
        clone_241: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_1121, memory_format = torch.contiguous_format);  view_1121 = None
        view_1122: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_241, [2048, 1024]);  clone_241 = None
        permute_35: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        permute_958: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_252: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1122, permute_958);  permute_958 = None
        permute_959: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1122, [1, 0])
        mm_253: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_959, view_66);  permute_959 = None
        sum_348: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1122, [0], True);  view_1122 = None
        view_1123: "f32[1024]" = torch.ops.aten.reshape.default(sum_348, [1024]);  sum_348 = None
        view_1124: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_252, [4, 512, 1024]);  mm_252 = None
        add_328: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_327, view_1124);  add_327 = view_1124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_962: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_1014, [0, 2, 1, 3]);  mul_1014 = None
        clone_242: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_962, memory_format = torch.contiguous_format);  permute_962 = None
        view_1125: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_242, [4, 512, 1024]);  clone_242 = None
        view_1126: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_1125, [2048, 1024]);  view_1125 = None
        permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        permute_963: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_254: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1126, permute_963);  permute_963 = None
        permute_964: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1126, [1, 0])
        mm_255: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_964, view_66);  permute_964 = view_66 = None
        sum_349: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1126, [0], True);  view_1126 = None
        view_1127: "f32[1024]" = torch.ops.aten.reshape.default(sum_349, [1024]);  sum_349 = None
        view_1128: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_254, [4, 512, 1024]);  mm_254 = None
        add_329: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_328, view_1128);  add_328 = view_1128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_1016: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_329, primals_55);  primals_55 = None
        mul_1017: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1016, 1024)
        sum_350: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1016, [2], True)
        mul_1018: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1016, mul_47);  mul_1016 = None
        sum_351: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1018, [2], True);  mul_1018 = None
        mul_1019: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_47, sum_351);  sum_351 = None
        sub_207: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_1017, sum_350);  mul_1017 = sum_350 = None
        sub_208: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_207, mul_1019);  sub_207 = mul_1019 = None
        mul_1020: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_69, sub_208);  div_69 = sub_208 = None
        mul_1021: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_329, mul_47);  mul_47 = None
        sum_352: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1021, [0, 1]);  mul_1021 = None
        sum_353: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_329, [0, 1]);  add_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_64: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_1022: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1.1111111111111112);  convert_element_type_64 = None
        mul_1023: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1020, mul_1022);  mul_1022 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_1129: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_1023, [2048, 1024]);  mul_1023 = None
        permute_32: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
        permute_967: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_256: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_1129, permute_967);  permute_967 = None
        permute_968: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1129, [1, 0])
        mm_257: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_968, view_64);  permute_968 = view_64 = None
        sum_354: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1129, [0], True);  view_1129 = None
        view_1130: "f32[1024]" = torch.ops.aten.reshape.default(sum_354, [1024]);  sum_354 = None
        view_1131: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_256, [4, 512, 3072]);  mm_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_63: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [4, 512, 3072]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_43: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
        erf_2: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_26: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_1025: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_26, 0.5);  add_26 = None
        mul_1026: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, view_63)
        mul_1027: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_1026, -0.5);  mul_1026 = None
        exp_48: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_1027);  mul_1027 = None
        mul_1028: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
        mul_1029: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, mul_1028);  view_63 = mul_1028 = None
        add_331: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_1025, mul_1029);  mul_1025 = mul_1029 = None
        mul_1030: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_1131, add_331);  view_1131 = add_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_1132: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_1030, [2048, 3072]);  mul_1030 = None
        permute_31: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        permute_971: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_258: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1132, permute_971);  permute_971 = None
        permute_972: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_1132, [1, 0])
        mm_259: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_972, view_62);  permute_972 = view_62 = None
        sum_355: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1132, [0], True);  view_1132 = None
        view_1133: "f32[3072]" = torch.ops.aten.reshape.default(sum_355, [3072]);  sum_355 = None
        view_1134: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_258, [4, 512, 1024]);  mm_258 = None
        add_332: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_1020, view_1134);  mul_1020 = view_1134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_1032: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_332, primals_49);  primals_49 = None
        mul_1033: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1032, 1024)
        sum_356: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1032, [2], True)
        mul_1034: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1032, mul_40);  mul_1032 = None
        sum_357: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1034, [2], True);  mul_1034 = None
        mul_1035: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_40, sum_357);  sum_357 = None
        sub_210: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_1033, sum_356);  mul_1033 = sum_356 = None
        sub_211: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_210, mul_1035);  sub_210 = mul_1035 = None
        mul_1036: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_70, sub_211);  div_70 = sub_211 = None
        mul_1037: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_332, mul_40);  mul_40 = None
        sum_358: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1037, [0, 1]);  mul_1037 = None
        sum_359: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_332, [0, 1]);  add_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_65: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_1038: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
        mul_1039: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1036, mul_1038);  mul_1038 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_1135: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_1039, [2048, 1024]);  mul_1039 = None
        permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
        permute_975: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_260: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1135, permute_975);  permute_975 = None
        permute_976: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1135, [1, 0])
        mm_261: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_976, view_60);  permute_976 = view_60 = None
        sum_360: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1135, [0], True);  view_1135 = None
        view_1136: "f32[1024]" = torch.ops.aten.reshape.default(sum_360, [1024]);  sum_360 = None
        view_1137: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_260, [4, 512, 1024]);  mm_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_1138: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1137, [4, 512, 16, 64]);  view_1137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_979: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_1138, [0, 2, 1, 3]);  view_1138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_245: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_979, memory_format = torch.contiguous_format);  permute_979 = None
        view_1139: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_245, [64, 512, 64]);  clone_245 = None
        bmm_132: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_980, view_1139);  permute_980 = None
        bmm_133: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_1139, permute_981);  view_1139 = permute_981 = None
        view_1140: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_132, [4, 16, 512, 64]);  bmm_132 = None
        view_1141: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_133, [4, 16, 512, 512]);  bmm_133 = None
        convert_element_type_66: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_1040: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
        mul_1041: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_1141, mul_1040);  view_1141 = mul_1040 = None
        mul_1042: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_1041, where_5);  mul_1041 = None
        sum_361: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1042, [-1], True)
        neg_22: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_5);  where_5 = None
        fma_21: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_22, sum_361, mul_1042);  neg_22 = sum_361 = mul_1042 = None
        view_1142: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_21, [64, 512, 512]);  fma_21 = None
        bmm_134: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_982, view_1142);  permute_982 = None
        bmm_135: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_1142, permute_983);  view_1142 = permute_983 = None
        view_1143: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_134, [4, 16, 64, 512]);  bmm_134 = None
        view_1144: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_135, [4, 16, 512, 64]);  bmm_135 = None
        mul_1043: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_1143, 0.3535533905932738);  view_1143 = None
        permute_984: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_1043, [0, 1, 3, 2]);  mul_1043 = None
        mul_1044: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_1144, 0.3535533905932738);  view_1144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_985: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_1140, [0, 2, 1, 3]);  view_1140 = None
        clone_247: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_985, memory_format = torch.contiguous_format);  permute_985 = None
        view_1145: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_247, [4, 512, 1024]);  clone_247 = None
        view_1146: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_1145, [2048, 1024]);  view_1145 = None
        permute_26: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        permute_986: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        mm_262: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1146, permute_986);  permute_986 = None
        permute_987: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1146, [1, 0])
        mm_263: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_987, view_44);  permute_987 = None
        sum_362: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1146, [0], True);  view_1146 = None
        view_1147: "f32[1024]" = torch.ops.aten.reshape.default(sum_362, [1024]);  sum_362 = None
        view_1148: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_262, [4, 512, 1024]);  mm_262 = None
        add_333: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_1036, view_1148);  mul_1036 = view_1148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_990: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_984, [0, 2, 1, 3]);  permute_984 = None
        view_1149: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_990, [4, 512, 1024]);  permute_990 = None
        clone_248: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_1149, memory_format = torch.contiguous_format);  view_1149 = None
        view_1150: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_248, [2048, 1024]);  clone_248 = None
        permute_24: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_991: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_264: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1150, permute_991);  permute_991 = None
        permute_992: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1150, [1, 0])
        mm_265: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_992, view_44);  permute_992 = None
        sum_363: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1150, [0], True);  view_1150 = None
        view_1151: "f32[1024]" = torch.ops.aten.reshape.default(sum_363, [1024]);  sum_363 = None
        view_1152: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_264, [4, 512, 1024]);  mm_264 = None
        add_334: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_333, view_1152);  add_333 = view_1152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_995: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_1044, [0, 2, 1, 3]);  mul_1044 = None
        clone_249: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_995, memory_format = torch.contiguous_format);  permute_995 = None
        view_1153: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_249, [4, 512, 1024]);  clone_249 = None
        view_1154: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_1153, [2048, 1024]);  view_1153 = None
        permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
        permute_996: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_266: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1154, permute_996);  permute_996 = None
        permute_997: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1154, [1, 0])
        mm_267: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_997, view_44);  permute_997 = view_44 = None
        sum_364: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1154, [0], True);  view_1154 = None
        view_1155: "f32[1024]" = torch.ops.aten.reshape.default(sum_364, [1024]);  sum_364 = None
        view_1156: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_266, [4, 512, 1024]);  mm_266 = None
        add_335: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_334, view_1156);  add_334 = view_1156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_1046: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_335, primals_39);  primals_39 = None
        mul_1047: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1046, 1024)
        sum_365: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1046, [2], True)
        mul_1048: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1046, mul_32);  mul_1046 = None
        sum_366: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1048, [2], True);  mul_1048 = None
        mul_1049: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_32, sum_366);  sum_366 = None
        sub_213: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_1047, sum_365);  mul_1047 = sum_365 = None
        sub_214: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_213, mul_1049);  sub_213 = mul_1049 = None
        mul_1050: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_71, sub_214);  div_71 = sub_214 = None
        mul_1051: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_335, mul_32);  mul_32 = None
        sum_367: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1051, [0, 1]);  mul_1051 = None
        sum_368: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_335, [0, 1]);  add_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_67: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_1052: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.1111111111111112);  convert_element_type_67 = None
        mul_1053: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1050, mul_1052);  mul_1052 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_1157: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_1053, [2048, 1024]);  mul_1053 = None
        permute_21: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        permute_1000: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_268: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_1157, permute_1000);  permute_1000 = None
        permute_1001: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1157, [1, 0])
        mm_269: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_1001, view_42);  permute_1001 = view_42 = None
        sum_369: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1157, [0], True);  view_1157 = None
        view_1158: "f32[1024]" = torch.ops.aten.reshape.default(sum_369, [1024]);  sum_369 = None
        view_1159: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_268, [4, 512, 3072]);  mm_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_41: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [4, 512, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_28: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
        erf_1: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_18: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_1055: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_18, 0.5);  add_18 = None
        mul_1056: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, view_41)
        mul_1057: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_1056, -0.5);  mul_1056 = None
        exp_49: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_1057);  mul_1057 = None
        mul_1058: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_49, 0.3989422804014327);  exp_49 = None
        mul_1059: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, mul_1058);  view_41 = mul_1058 = None
        add_337: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_1055, mul_1059);  mul_1055 = mul_1059 = None
        mul_1060: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_1159, add_337);  view_1159 = add_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_1160: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_1060, [2048, 3072]);  mul_1060 = None
        permute_20: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
        permute_1004: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_270: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1160, permute_1004);  permute_1004 = None
        permute_1005: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_1160, [1, 0])
        mm_271: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_1005, view_40);  permute_1005 = view_40 = None
        sum_370: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1160, [0], True);  view_1160 = None
        view_1161: "f32[3072]" = torch.ops.aten.reshape.default(sum_370, [3072]);  sum_370 = None
        view_1162: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_270, [4, 512, 1024]);  mm_270 = None
        add_338: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_1050, view_1162);  mul_1050 = view_1162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_1062: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_338, primals_33);  primals_33 = None
        mul_1063: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1062, 1024)
        sum_371: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1062, [2], True)
        mul_1064: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1062, mul_25);  mul_1062 = None
        sum_372: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1064, [2], True);  mul_1064 = None
        mul_1065: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_25, sum_372);  sum_372 = None
        sub_216: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_1063, sum_371);  mul_1063 = sum_371 = None
        sub_217: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_216, mul_1065);  sub_216 = mul_1065 = None
        mul_1066: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_72, sub_217);  div_72 = sub_217 = None
        mul_1067: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_338, mul_25);  mul_25 = None
        sum_373: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1067, [0, 1]);  mul_1067 = None
        sum_374: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_338, [0, 1]);  add_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_68: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_1068: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
        mul_1069: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1066, mul_1068);  mul_1068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_1163: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_1069, [2048, 1024]);  mul_1069 = None
        permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        permute_1008: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_272: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1163, permute_1008);  permute_1008 = None
        permute_1009: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1163, [1, 0])
        mm_273: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1009, view_38);  permute_1009 = view_38 = None
        sum_375: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1163, [0], True);  view_1163 = None
        view_1164: "f32[1024]" = torch.ops.aten.reshape.default(sum_375, [1024]);  sum_375 = None
        view_1165: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_272, [4, 512, 1024]);  mm_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_1166: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1165, [4, 512, 16, 64]);  view_1165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1012: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_1166, [0, 2, 1, 3]);  view_1166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_252: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_1012, memory_format = torch.contiguous_format);  permute_1012 = None
        view_1167: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_252, [64, 512, 64]);  clone_252 = None
        bmm_136: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_1013, view_1167);  permute_1013 = None
        bmm_137: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_1167, permute_1014);  view_1167 = permute_1014 = None
        view_1168: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_136, [4, 16, 512, 64]);  bmm_136 = None
        view_1169: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_137, [4, 16, 512, 512]);  bmm_137 = None
        convert_element_type_69: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_1070: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_69, 1.1111111111111112);  convert_element_type_69 = None
        mul_1071: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_1169, mul_1070);  view_1169 = mul_1070 = None
        mul_1072: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_1071, where_3);  mul_1071 = None
        sum_376: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1072, [-1], True)
        neg_23: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_3);  where_3 = None
        fma_22: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_23, sum_376, mul_1072);  neg_23 = sum_376 = mul_1072 = None
        view_1170: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_22, [64, 512, 512]);  fma_22 = None
        bmm_138: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_1015, view_1170);  permute_1015 = None
        bmm_139: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_1170, permute_1016);  view_1170 = permute_1016 = None
        view_1171: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_138, [4, 16, 64, 512]);  bmm_138 = None
        view_1172: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_139, [4, 16, 512, 64]);  bmm_139 = None
        mul_1073: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_1171, 0.3535533905932738);  view_1171 = None
        permute_1017: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_1073, [0, 1, 3, 2]);  mul_1073 = None
        mul_1074: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_1172, 0.3535533905932738);  view_1172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_1018: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_1168, [0, 2, 1, 3]);  view_1168 = None
        clone_254: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1018, memory_format = torch.contiguous_format);  permute_1018 = None
        view_1173: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_254, [4, 512, 1024]);  clone_254 = None
        view_1174: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_1173, [2048, 1024]);  view_1173 = None
        permute_15: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_1019: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        mm_274: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1174, permute_1019);  permute_1019 = None
        permute_1020: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1174, [1, 0])
        mm_275: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1020, view_22);  permute_1020 = None
        sum_377: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1174, [0], True);  view_1174 = None
        view_1175: "f32[1024]" = torch.ops.aten.reshape.default(sum_377, [1024]);  sum_377 = None
        view_1176: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_274, [4, 512, 1024]);  mm_274 = None
        add_339: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_1066, view_1176);  mul_1066 = view_1176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_1023: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_1017, [0, 2, 1, 3]);  permute_1017 = None
        view_1177: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_1023, [4, 512, 1024]);  permute_1023 = None
        clone_255: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_1177, memory_format = torch.contiguous_format);  view_1177 = None
        view_1178: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_255, [2048, 1024]);  clone_255 = None
        permute_13: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_1024: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_276: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1178, permute_1024);  permute_1024 = None
        permute_1025: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1178, [1, 0])
        mm_277: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1025, view_22);  permute_1025 = None
        sum_378: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1178, [0], True);  view_1178 = None
        view_1179: "f32[1024]" = torch.ops.aten.reshape.default(sum_378, [1024]);  sum_378 = None
        view_1180: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_276, [4, 512, 1024]);  mm_276 = None
        add_340: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_339, view_1180);  add_339 = view_1180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_1028: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_1074, [0, 2, 1, 3]);  mul_1074 = None
        clone_256: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1028, memory_format = torch.contiguous_format);  permute_1028 = None
        view_1181: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_256, [4, 512, 1024]);  clone_256 = None
        view_1182: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_1181, [2048, 1024]);  view_1181 = None
        permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_1029: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_278: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1182, permute_1029);  permute_1029 = None
        permute_1030: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1182, [1, 0])
        mm_279: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1030, view_22);  permute_1030 = view_22 = None
        sum_379: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1182, [0], True);  view_1182 = None
        view_1183: "f32[1024]" = torch.ops.aten.reshape.default(sum_379, [1024]);  sum_379 = None
        view_1184: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_278, [4, 512, 1024]);  mm_278 = None
        add_341: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_340, view_1184);  add_340 = view_1184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_1076: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_341, primals_23);  primals_23 = None
        mul_1077: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1076, 1024)
        sum_380: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1076, [2], True)
        mul_1078: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1076, mul_17);  mul_1076 = None
        sum_381: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1078, [2], True);  mul_1078 = None
        mul_1079: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_17, sum_381);  sum_381 = None
        sub_219: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_1077, sum_380);  mul_1077 = sum_380 = None
        sub_220: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_219, mul_1079);  sub_219 = mul_1079 = None
        mul_1080: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_73, sub_220);  div_73 = sub_220 = None
        mul_1081: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_341, mul_17);  mul_17 = None
        sum_382: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1081, [0, 1]);  mul_1081 = None
        sum_383: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_341, [0, 1]);  add_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_70: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_1082: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 1.1111111111111112);  convert_element_type_70 = None
        mul_1083: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1080, mul_1082);  mul_1082 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_1185: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_1083, [2048, 1024]);  mul_1083 = None
        permute_10: "f32[3072, 1024]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_1033: "f32[1024, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_280: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_1185, permute_1033);  permute_1033 = None
        permute_1034: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1185, [1, 0])
        mm_281: "f32[1024, 3072]" = torch.ops.aten.mm.default(permute_1034, view_20);  permute_1034 = view_20 = None
        sum_384: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1185, [0], True);  view_1185 = None
        view_1186: "f32[1024]" = torch.ops.aten.reshape.default(sum_384, [1024]);  sum_384 = None
        view_1187: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_280, [4, 512, 3072]);  mm_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_19: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [4, 512, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_13: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
        erf: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_10: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_1085: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_10, 0.5);  add_10 = None
        mul_1086: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, view_19)
        mul_1087: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_1086, -0.5);  mul_1086 = None
        exp_50: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_1087);  mul_1087 = None
        mul_1088: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_50, 0.3989422804014327);  exp_50 = None
        mul_1089: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, mul_1088);  view_19 = mul_1088 = None
        add_343: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_1085, mul_1089);  mul_1085 = mul_1089 = None
        mul_1090: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_1187, add_343);  view_1187 = add_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_1188: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_1090, [2048, 3072]);  mul_1090 = None
        permute_9: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_1037: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_282: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1188, permute_1037);  permute_1037 = None
        permute_1038: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_1188, [1, 0])
        mm_283: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_1038, view_18);  permute_1038 = view_18 = None
        sum_385: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1188, [0], True);  view_1188 = None
        view_1189: "f32[3072]" = torch.ops.aten.reshape.default(sum_385, [3072]);  sum_385 = None
        view_1190: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_282, [4, 512, 1024]);  mm_282 = None
        add_344: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_1080, view_1190);  mul_1080 = view_1190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_1092: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_344, primals_17);  primals_17 = None
        mul_1093: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1092, 1024)
        sum_386: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1092, [2], True)
        mul_1094: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1092, mul_10);  mul_1092 = None
        sum_387: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1094, [2], True);  mul_1094 = None
        mul_1095: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_10, sum_387);  sum_387 = None
        sub_222: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_1093, sum_386);  mul_1093 = sum_386 = None
        sub_223: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_222, mul_1095);  sub_222 = mul_1095 = None
        mul_1096: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_74, sub_223);  div_74 = sub_223 = None
        mul_1097: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_344, mul_10);  mul_10 = None
        sum_388: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1097, [0, 1]);  mul_1097 = None
        sum_389: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_344, [0, 1]);  add_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_71: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_1098: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
        mul_1099: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1096, mul_1098);  mul_1098 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_1191: "f32[2048, 1024]" = torch.ops.aten.reshape.default(mul_1099, [2048, 1024]);  mul_1099 = None
        permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_1041: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_284: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1191, permute_1041);  permute_1041 = None
        permute_1042: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1191, [1, 0])
        mm_285: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1042, view_16);  permute_1042 = view_16 = None
        sum_390: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1191, [0], True);  view_1191 = None
        view_1192: "f32[1024]" = torch.ops.aten.reshape.default(sum_390, [1024]);  sum_390 = None
        view_1193: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_284, [4, 512, 1024]);  mm_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_1194: "f32[4, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1193, [4, 512, 16, 64]);  view_1193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1045: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(view_1194, [0, 2, 1, 3]);  view_1194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_259: "f32[4, 16, 512, 64]" = torch.ops.aten.clone.default(permute_1045, memory_format = torch.contiguous_format);  permute_1045 = None
        view_1195: "f32[64, 512, 64]" = torch.ops.aten.reshape.default(clone_259, [64, 512, 64]);  clone_259 = None
        bmm_140: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(permute_1046, view_1195);  permute_1046 = None
        bmm_141: "f32[64, 512, 512]" = torch.ops.aten.bmm.default(view_1195, permute_1047);  view_1195 = permute_1047 = None
        view_1196: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_140, [4, 16, 512, 64]);  bmm_140 = None
        view_1197: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm_141, [4, 16, 512, 512]);  bmm_141 = None
        convert_element_type_72: "f32[4, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_1100: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 1.1111111111111112);  convert_element_type_72 = None
        mul_1101: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(view_1197, mul_1100);  view_1197 = mul_1100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_2: "b8[4, 1, 512, 512]" = torch.ops.aten.expand.default(ge, [4, -1, 512, 512]);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[4, 1, 512, 512]" = torch.ops.aten.where.self(expand_2, full_default_1, full_default);  expand_2 = full_default = None
        view_11: "f32[4, 16, 512, 512]" = torch.ops.aten.reshape.default(bmm, [4, 16, 512, 512]);  bmm = None
        add_6: "f32[4, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_11, where);  view_11 = where = None
        sub_1: "f32[4, 16, 512, 512]" = torch.ops.aten.sub.Tensor(add_6, amax);  add_6 = amax = None
        exp: "f32[4, 16, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div: "f32[4, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        full_default_2: "f32[4, 16, 512, 512]" = torch.ops.aten.full.default([4, 16, 512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[4, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div);  logical_not_1 = full_default_2 = div = None
        mul_1102: "f32[4, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_1101, where_1);  mul_1101 = None
        sum_391: "f32[4, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1102, [-1], True)
        neg_24: "f32[4, 16, 512, 512]" = torch.ops.aten.neg.default(where_1);  where_1 = None
        fma_23: "f32[4, 16, 512, 512]" = torch.ops.prims.fma.default(neg_24, sum_391, mul_1102);  neg_24 = sum_391 = mul_1102 = None
        view_1198: "f32[64, 512, 512]" = torch.ops.aten.reshape.default(fma_23, [64, 512, 512]);  fma_23 = None
        bmm_142: "f32[64, 64, 512]" = torch.ops.aten.bmm.default(permute_1048, view_1198);  permute_1048 = None
        bmm_143: "f32[64, 512, 64]" = torch.ops.aten.bmm.default(view_1198, permute_1049);  view_1198 = permute_1049 = None
        view_1199: "f32[4, 16, 64, 512]" = torch.ops.aten.reshape.default(bmm_142, [4, 16, 64, 512]);  bmm_142 = None
        view_1200: "f32[4, 16, 512, 64]" = torch.ops.aten.reshape.default(bmm_143, [4, 16, 512, 64]);  bmm_143 = None
        mul_1103: "f32[4, 16, 64, 512]" = torch.ops.aten.mul.Scalar(view_1199, 0.3535533905932738);  view_1199 = None
        permute_1050: "f32[4, 16, 512, 64]" = torch.ops.aten.permute.default(mul_1103, [0, 1, 3, 2]);  mul_1103 = None
        mul_1104: "f32[4, 16, 512, 64]" = torch.ops.aten.mul.Scalar(view_1200, 0.3535533905932738);  view_1200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_1051: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(view_1196, [0, 2, 1, 3]);  view_1196 = None
        clone_261: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1051, memory_format = torch.contiguous_format);  permute_1051 = None
        view_1201: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_261, [4, 512, 1024]);  clone_261 = None
        view_1202: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_1201, [2048, 1024]);  view_1201 = None
        permute_4: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_1052: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_286: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1202, permute_1052);  permute_1052 = None
        permute_1053: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1202, [1, 0])
        mm_287: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1053, view);  permute_1053 = None
        sum_392: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1202, [0], True);  view_1202 = None
        view_1203: "f32[1024]" = torch.ops.aten.reshape.default(sum_392, [1024]);  sum_392 = None
        view_1204: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_286, [4, 512, 1024]);  mm_286 = None
        add_345: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_1096, view_1204);  mul_1096 = view_1204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_1056: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(permute_1050, [0, 2, 1, 3]);  permute_1050 = None
        view_1205: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(permute_1056, [4, 512, 1024]);  permute_1056 = None
        clone_262: "f32[4, 512, 1024]" = torch.ops.aten.clone.default(view_1205, memory_format = torch.contiguous_format);  view_1205 = None
        view_1206: "f32[2048, 1024]" = torch.ops.aten.reshape.default(clone_262, [2048, 1024]);  clone_262 = None
        permute_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_1057: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_288: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1206, permute_1057);  permute_1057 = None
        permute_1058: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1206, [1, 0])
        mm_289: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1058, view);  permute_1058 = None
        sum_393: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1206, [0], True);  view_1206 = None
        view_1207: "f32[1024]" = torch.ops.aten.reshape.default(sum_393, [1024]);  sum_393 = None
        view_1208: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_288, [4, 512, 1024]);  mm_288 = None
        add_346: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_345, view_1208);  add_345 = view_1208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_1061: "f32[4, 512, 16, 64]" = torch.ops.aten.permute.default(mul_1104, [0, 2, 1, 3]);  mul_1104 = None
        clone_263: "f32[4, 512, 16, 64]" = torch.ops.aten.clone.default(permute_1061, memory_format = torch.contiguous_format);  permute_1061 = None
        view_1209: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(clone_263, [4, 512, 1024]);  clone_263 = None
        view_1210: "f32[2048, 1024]" = torch.ops.aten.reshape.default(view_1209, [2048, 1024]);  view_1209 = None
        permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_1062: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_290: "f32[2048, 1024]" = torch.ops.aten.mm.default(view_1210, permute_1062);  permute_1062 = None
        permute_1063: "f32[1024, 2048]" = torch.ops.aten.permute.default(view_1210, [1, 0])
        mm_291: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1063, view);  permute_1063 = view = None
        sum_394: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True);  view_1210 = None
        view_1211: "f32[1024]" = torch.ops.aten.reshape.default(sum_394, [1024]);  sum_394 = None
        view_1212: "f32[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_290, [4, 512, 1024]);  mm_290 = None
        add_347: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_346, view_1212);  add_346 = view_1212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:111 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_73: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_1105: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_73, 1.1111111111111112);  convert_element_type_73 = None
        mul_1106: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_347, mul_1105);  add_347 = mul_1105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:110 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_1108: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1106, primals_7);  primals_7 = None
        mul_1109: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1108, 1024)
        sum_395: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1108, [2], True)
        mul_1110: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1108, mul);  mul_1108 = None
        sum_396: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1110, [2], True);  mul_1110 = None
        mul_1111: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul, sum_396);  sum_396 = None
        sub_225: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_1109, sum_395);  mul_1109 = sum_395 = None
        sub_226: "f32[4, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_225, mul_1111);  sub_225 = mul_1111 = None
        mul_1112: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_75, sub_226);  div_75 = sub_226 = None
        mul_1113: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1106, mul);  mul = None
        sum_397: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1113, [0, 1]);  mul_1113 = None
        sum_398: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1106, [0, 1]);  mul_1106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:108 in forward, code: embeddings = embeddings + position_embeddings
        sum_399: "f32[1, 512, 1024]" = torch.ops.aten.sum.dim_IntList(mul_1112, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:107 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_24: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_2, -1)
        unsqueeze_5: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_24, -1);  eq_24 = None
        where_52: "f32[1, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_5, full_default_1, sum_399);  unsqueeze_5 = sum_399 = None
        full_default_78: "f32[512, 1024]" = torch.ops.aten.full.default([512, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[512, 1024]" = torch.ops.aten.index_put.default(full_default_78, [primals_2], where_52, True);  full_default_78 = primals_2 = where_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:98 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_1: "i64[4, 512]" = torch.ops.aten.expand.default(gather, [4, 512]);  gather = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:104 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq_25: "b8[4, 512]" = torch.ops.aten.eq.Scalar(expand_1, -1)
        unsqueeze_6: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_25, -1);  eq_25 = None
        where_53: "f32[4, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_6, full_default_1, mul_1112);  unsqueeze_6 = None
        full_default_80: "f32[2, 1024]" = torch.ops.aten.full.default([2, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[2, 1024]" = torch.ops.aten.index_put.default(full_default_80, [expand_1], where_53, True);  full_default_80 = expand_1 = where_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:103 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_26: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_7: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_26, -1);  eq_26 = None
        where_54: "f32[4, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_7, full_default_1, mul_1112);  unsqueeze_7 = full_default_1 = mul_1112 = None
        full_default_82: "f32[30522, 1024]" = torch.ops.aten.full.default([30522, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_2: "f32[30522, 1024]" = torch.ops.aten.index_put.default(full_default_82, [primals_1], where_54, True);  full_default_82 = primals_1 = where_54 = None
        add_348: "f32[30522, 1024]" = torch.ops.aten.add.Tensor(mm_1, index_put_2);  mm_1 = index_put_2 = None
        return (None, None, None, add_348, index_put_1, index_put, sum_397, sum_398, mm_291, view_1211, mm_289, view_1207, mm_287, view_1203, mm_285, view_1192, sum_388, sum_389, mm_283, view_1189, mm_281, view_1186, sum_382, sum_383, mm_279, view_1183, mm_277, view_1179, mm_275, view_1175, mm_273, view_1164, sum_373, sum_374, mm_271, view_1161, mm_269, view_1158, sum_367, sum_368, mm_267, view_1155, mm_265, view_1151, mm_263, view_1147, mm_261, view_1136, sum_358, sum_359, mm_259, view_1133, mm_257, view_1130, sum_352, sum_353, mm_255, view_1127, mm_253, view_1123, mm_251, view_1119, mm_249, view_1108, sum_343, sum_344, mm_247, view_1105, mm_245, view_1102, sum_337, sum_338, mm_243, view_1099, mm_241, view_1095, mm_239, view_1091, mm_237, view_1080, sum_328, sum_329, mm_235, view_1077, mm_233, view_1074, sum_322, sum_323, mm_231, view_1071, mm_229, view_1067, mm_227, view_1063, mm_225, view_1052, sum_313, sum_314, mm_223, view_1049, mm_221, view_1046, sum_307, sum_308, mm_219, view_1043, mm_217, view_1039, mm_215, view_1035, mm_213, view_1024, sum_298, sum_299, mm_211, view_1021, mm_209, view_1018, sum_292, sum_293, mm_207, view_1015, mm_205, view_1011, mm_203, view_1007, mm_201, view_996, sum_283, sum_284, mm_199, view_993, mm_197, view_990, sum_277, sum_278, mm_195, view_987, mm_193, view_983, mm_191, view_979, mm_189, view_968, sum_268, sum_269, mm_187, view_965, mm_185, view_962, sum_262, sum_263, mm_183, view_959, mm_181, view_955, mm_179, view_951, mm_177, view_940, sum_253, sum_254, mm_175, view_937, mm_173, view_934, sum_247, sum_248, mm_171, view_931, mm_169, view_927, mm_167, view_923, mm_165, view_912, sum_238, sum_239, mm_163, view_909, mm_161, view_906, sum_232, sum_233, mm_159, view_903, mm_157, view_899, mm_155, view_895, mm_153, view_884, sum_223, sum_224, mm_151, view_881, mm_149, view_878, sum_217, sum_218, mm_147, view_875, mm_145, view_871, mm_143, view_867, mm_141, view_856, sum_208, sum_209, mm_139, view_853, mm_137, view_850, sum_202, sum_203, mm_135, view_847, mm_133, view_843, mm_131, view_839, mm_129, view_828, sum_193, sum_194, mm_127, view_825, mm_125, view_822, sum_187, sum_188, mm_123, view_819, mm_121, view_815, mm_119, view_811, mm_117, view_800, sum_178, sum_179, mm_115, view_797, mm_113, view_794, sum_172, sum_173, mm_111, view_791, mm_109, view_787, mm_107, view_783, mm_105, view_772, sum_163, sum_164, mm_103, view_769, mm_101, view_766, sum_157, sum_158, mm_99, view_763, mm_97, view_759, mm_95, view_755, mm_93, view_744, sum_148, sum_149, mm_91, view_741, mm_89, view_738, sum_142, sum_143, mm_87, view_735, mm_85, view_731, mm_83, view_727, mm_81, view_716, sum_133, sum_134, mm_79, view_713, mm_77, view_710, sum_127, sum_128, mm_75, view_707, mm_73, view_703, mm_71, view_699, mm_69, view_688, sum_118, sum_119, mm_67, view_685, mm_65, view_682, sum_112, sum_113, mm_63, view_679, mm_61, view_675, mm_59, view_671, mm_57, view_660, sum_103, sum_104, mm_55, view_657, mm_53, view_654, sum_97, sum_98, mm_51, view_651, mm_49, view_647, mm_47, view_643, mm_45, view_632, sum_88, sum_89, mm_43, view_629, mm_41, view_626, sum_82, sum_83, mm_39, view_623, mm_37, view_619, mm_35, view_615, mm_33, view_604, sum_73, sum_74, mm_31, view_601, mm_29, view_598, sum_67, sum_68, mm_27, view_595, mm_25, view_591, mm_23, view_587, mm_21, view_576, sum_58, sum_59, mm_19, view_573, mm_17, view_570, sum_52, sum_53, mm_15, view_567, mm_13, view_563, mm_11, view_559, mm_9, view_548, sum_43, sum_44, mm_7, view_545, mm_5, view_542, sum_37, sum_38, mm_3, view_539, sum_32, sum_33, view_536, None)
