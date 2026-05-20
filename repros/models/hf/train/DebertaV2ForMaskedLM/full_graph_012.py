class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 512]", primals_2: "i64[1, 512]", primals_3: "f32[128100, 1536]", primals_5: "f32[1536]", primals_7: "f32[1536, 1536]", primals_9: "f32[1536, 1536]", primals_11: "f32[1536, 1536]", primals_13: "f32[1536, 1536]", primals_15: "f32[1536]", primals_17: "f32[6144, 1536]", primals_19: "f32[1536, 6144]", primals_21: "f32[1536]", primals_23: "f32[1536, 1536]", primals_25: "f32[1536, 1536]", primals_27: "f32[1536, 1536]", primals_29: "f32[1536, 1536]", primals_31: "f32[1536]", primals_33: "f32[6144, 1536]", primals_35: "f32[1536, 6144]", primals_37: "f32[1536]", primals_39: "f32[1536, 1536]", primals_41: "f32[1536, 1536]", primals_43: "f32[1536, 1536]", primals_45: "f32[1536, 1536]", primals_47: "f32[1536]", primals_49: "f32[6144, 1536]", primals_51: "f32[1536, 6144]", primals_53: "f32[1536]", primals_55: "f32[1536, 1536]", primals_57: "f32[1536, 1536]", primals_59: "f32[1536, 1536]", primals_61: "f32[1536, 1536]", primals_63: "f32[1536]", primals_65: "f32[6144, 1536]", primals_67: "f32[1536, 6144]", primals_69: "f32[1536]", primals_71: "f32[1536, 1536]", primals_73: "f32[1536, 1536]", primals_75: "f32[1536, 1536]", primals_77: "f32[1536, 1536]", primals_79: "f32[1536]", primals_81: "f32[6144, 1536]", primals_83: "f32[1536, 6144]", primals_85: "f32[1536]", primals_87: "f32[1536, 1536]", primals_89: "f32[1536, 1536]", primals_91: "f32[1536, 1536]", primals_93: "f32[1536, 1536]", primals_95: "f32[1536]", primals_97: "f32[6144, 1536]", primals_99: "f32[1536, 6144]", primals_101: "f32[1536]", primals_103: "f32[1536, 1536]", primals_105: "f32[1536, 1536]", primals_107: "f32[1536, 1536]", primals_109: "f32[1536, 1536]", primals_111: "f32[1536]", primals_113: "f32[6144, 1536]", primals_115: "f32[1536, 6144]", primals_117: "f32[1536]", primals_119: "f32[1536, 1536]", primals_121: "f32[1536, 1536]", primals_123: "f32[1536, 1536]", primals_125: "f32[1536, 1536]", primals_127: "f32[1536]", primals_129: "f32[6144, 1536]", primals_131: "f32[1536, 6144]", primals_133: "f32[1536]", primals_135: "f32[1536, 1536]", primals_137: "f32[1536, 1536]", primals_139: "f32[1536, 1536]", primals_141: "f32[1536, 1536]", primals_143: "f32[1536]", primals_145: "f32[6144, 1536]", primals_147: "f32[1536, 6144]", primals_149: "f32[1536]", primals_151: "f32[1536, 1536]", primals_153: "f32[1536, 1536]", primals_155: "f32[1536, 1536]", primals_157: "f32[1536, 1536]", primals_159: "f32[1536]", primals_161: "f32[6144, 1536]", primals_163: "f32[1536, 6144]", primals_165: "f32[1536]", primals_167: "f32[1536, 1536]", primals_169: "f32[1536, 1536]", primals_171: "f32[1536, 1536]", primals_173: "f32[1536, 1536]", primals_175: "f32[1536]", primals_177: "f32[6144, 1536]", primals_179: "f32[1536, 6144]", primals_181: "f32[1536]", primals_183: "f32[1536, 1536]", primals_185: "f32[1536, 1536]", primals_187: "f32[1536, 1536]", primals_189: "f32[1536, 1536]", primals_191: "f32[1536]", primals_193: "f32[6144, 1536]", primals_195: "f32[1536, 6144]", primals_197: "f32[1536]", primals_199: "f32[1536, 1536]", primals_201: "f32[1536, 1536]", primals_203: "f32[1536, 1536]", primals_205: "f32[1536, 1536]", primals_207: "f32[1536]", primals_209: "f32[6144, 1536]", primals_211: "f32[1536, 6144]", primals_213: "f32[1536]", primals_215: "f32[1536, 1536]", primals_217: "f32[1536, 1536]", primals_219: "f32[1536, 1536]", primals_221: "f32[1536, 1536]", primals_223: "f32[1536]", primals_225: "f32[6144, 1536]", primals_227: "f32[1536, 6144]", primals_229: "f32[1536]", primals_231: "f32[1536, 1536]", primals_233: "f32[1536, 1536]", primals_235: "f32[1536, 1536]", primals_237: "f32[1536, 1536]", primals_239: "f32[1536]", primals_241: "f32[6144, 1536]", primals_243: "f32[1536, 6144]", primals_245: "f32[1536]", primals_247: "f32[1536, 1536]", primals_249: "f32[1536, 1536]", primals_251: "f32[1536, 1536]", primals_253: "f32[1536, 1536]", primals_255: "f32[1536]", primals_257: "f32[6144, 1536]", primals_259: "f32[1536, 6144]", primals_261: "f32[1536]", primals_263: "f32[1536, 1536]", primals_265: "f32[1536, 1536]", primals_267: "f32[1536, 1536]", primals_269: "f32[1536, 1536]", primals_271: "f32[1536]", primals_273: "f32[6144, 1536]", primals_275: "f32[1536, 6144]", primals_277: "f32[1536]", primals_279: "f32[1536, 1536]", primals_281: "f32[1536, 1536]", primals_283: "f32[1536, 1536]", primals_285: "f32[1536, 1536]", primals_287: "f32[1536]", primals_289: "f32[6144, 1536]", primals_291: "f32[1536, 6144]", primals_293: "f32[1536]", primals_295: "f32[1536, 1536]", primals_297: "f32[1536, 1536]", primals_299: "f32[1536, 1536]", primals_301: "f32[1536, 1536]", primals_303: "f32[1536]", primals_305: "f32[6144, 1536]", primals_307: "f32[1536, 6144]", primals_309: "f32[1536]", primals_311: "f32[1536, 1536]", primals_313: "f32[1536, 1536]", primals_315: "f32[1536, 1536]", primals_317: "f32[1536, 1536]", primals_319: "f32[1536]", primals_321: "f32[6144, 1536]", primals_323: "f32[1536, 6144]", primals_325: "f32[1536]", primals_327: "f32[1536, 1536]", primals_329: "f32[1536, 1536]", primals_331: "f32[1536, 1536]", primals_333: "f32[1536, 1536]", primals_335: "f32[1536]", primals_337: "f32[6144, 1536]", primals_339: "f32[1536, 6144]", primals_341: "f32[1536]", primals_343: "f32[1536, 1536]", primals_345: "f32[1536, 1536]", primals_347: "f32[1536, 1536]", primals_349: "f32[1536, 1536]", primals_351: "f32[1536]", primals_353: "f32[6144, 1536]", primals_355: "f32[1536, 6144]", primals_357: "f32[1536]", primals_359: "f32[1536, 1536]", primals_361: "f32[1536, 1536]", primals_363: "f32[1536, 1536]", primals_365: "f32[1536, 1536]", primals_367: "f32[1536]", primals_369: "f32[6144, 1536]", primals_371: "f32[1536, 6144]", primals_373: "f32[1536]", primals_375: "f32[1536, 1536]", primals_377: "f32[1536, 1536]", primals_379: "f32[1536, 1536]", primals_381: "f32[1536, 1536]", primals_383: "f32[1536]", primals_385: "f32[6144, 1536]", primals_387: "f32[1536, 6144]", primals_389: "f32[1536]", primals_391: "f32[1536, 1536]", primals_393: "f32[1536]", primals_396: "i64[8, 512]", embedding: "f32[8, 512, 1536]", embedding_1: "f32[1, 512, 1536]", getitem_1: "f32[8, 512, 1]", rsqrt: "f32[8, 512, 1]", gt: "b8[8, 512, 1536]", view: "f32[4096, 1536]", bmm: "f32[192, 512, 512]", amax: "f32[8, 24, 512, 1]", sum_1: "f32[8, 24, 512, 1]", gt_1: "b8[8, 24, 512, 512]", view_16: "f32[4096, 1536]", gt_2: "b8[8, 512, 1536]", mul_11: "f32[8, 512, 1536]", view_18: "f32[4096, 1536]", addmm_4: "f32[4096, 6144]", view_20: "f32[4096, 6144]", gt_3: "b8[8, 512, 1536]", mul_18: "f32[8, 512, 1536]", view_22: "f32[4096, 1536]", div_3: "f32[8, 24, 512, 512]", gt_4: "b8[8, 24, 512, 512]", view_38: "f32[4096, 1536]", gt_5: "b8[8, 512, 1536]", mul_25: "f32[8, 512, 1536]", view_40: "f32[4096, 1536]", addmm_10: "f32[4096, 6144]", view_42: "f32[4096, 6144]", gt_6: "b8[8, 512, 1536]", mul_32: "f32[8, 512, 1536]", view_44: "f32[4096, 1536]", div_5: "f32[8, 24, 512, 512]", gt_7: "b8[8, 24, 512, 512]", view_60: "f32[4096, 1536]", gt_8: "b8[8, 512, 1536]", mul_39: "f32[8, 512, 1536]", view_62: "f32[4096, 1536]", addmm_16: "f32[4096, 6144]", view_64: "f32[4096, 6144]", gt_9: "b8[8, 512, 1536]", mul_46: "f32[8, 512, 1536]", view_66: "f32[4096, 1536]", div_7: "f32[8, 24, 512, 512]", gt_10: "b8[8, 24, 512, 512]", view_82: "f32[4096, 1536]", gt_11: "b8[8, 512, 1536]", mul_53: "f32[8, 512, 1536]", view_84: "f32[4096, 1536]", addmm_22: "f32[4096, 6144]", view_86: "f32[4096, 6144]", gt_12: "b8[8, 512, 1536]", mul_60: "f32[8, 512, 1536]", view_88: "f32[4096, 1536]", div_9: "f32[8, 24, 512, 512]", gt_13: "b8[8, 24, 512, 512]", view_104: "f32[4096, 1536]", gt_14: "b8[8, 512, 1536]", mul_67: "f32[8, 512, 1536]", view_106: "f32[4096, 1536]", addmm_28: "f32[4096, 6144]", view_108: "f32[4096, 6144]", gt_15: "b8[8, 512, 1536]", mul_74: "f32[8, 512, 1536]", view_110: "f32[4096, 1536]", div_11: "f32[8, 24, 512, 512]", gt_16: "b8[8, 24, 512, 512]", view_126: "f32[4096, 1536]", gt_17: "b8[8, 512, 1536]", mul_81: "f32[8, 512, 1536]", view_128: "f32[4096, 1536]", addmm_34: "f32[4096, 6144]", view_130: "f32[4096, 6144]", gt_18: "b8[8, 512, 1536]", mul_88: "f32[8, 512, 1536]", view_132: "f32[4096, 1536]", div_13: "f32[8, 24, 512, 512]", gt_19: "b8[8, 24, 512, 512]", view_148: "f32[4096, 1536]", gt_20: "b8[8, 512, 1536]", mul_95: "f32[8, 512, 1536]", view_150: "f32[4096, 1536]", addmm_40: "f32[4096, 6144]", view_152: "f32[4096, 6144]", gt_21: "b8[8, 512, 1536]", mul_102: "f32[8, 512, 1536]", view_154: "f32[4096, 1536]", div_15: "f32[8, 24, 512, 512]", gt_22: "b8[8, 24, 512, 512]", view_170: "f32[4096, 1536]", gt_23: "b8[8, 512, 1536]", mul_109: "f32[8, 512, 1536]", view_172: "f32[4096, 1536]", addmm_46: "f32[4096, 6144]", view_174: "f32[4096, 6144]", gt_24: "b8[8, 512, 1536]", mul_116: "f32[8, 512, 1536]", view_176: "f32[4096, 1536]", div_17: "f32[8, 24, 512, 512]", gt_25: "b8[8, 24, 512, 512]", view_192: "f32[4096, 1536]", gt_26: "b8[8, 512, 1536]", mul_123: "f32[8, 512, 1536]", view_194: "f32[4096, 1536]", addmm_52: "f32[4096, 6144]", view_196: "f32[4096, 6144]", gt_27: "b8[8, 512, 1536]", mul_130: "f32[8, 512, 1536]", view_198: "f32[4096, 1536]", div_19: "f32[8, 24, 512, 512]", gt_28: "b8[8, 24, 512, 512]", view_214: "f32[4096, 1536]", gt_29: "b8[8, 512, 1536]", mul_137: "f32[8, 512, 1536]", view_216: "f32[4096, 1536]", addmm_58: "f32[4096, 6144]", view_218: "f32[4096, 6144]", gt_30: "b8[8, 512, 1536]", mul_144: "f32[8, 512, 1536]", view_220: "f32[4096, 1536]", div_21: "f32[8, 24, 512, 512]", gt_31: "b8[8, 24, 512, 512]", view_236: "f32[4096, 1536]", gt_32: "b8[8, 512, 1536]", mul_151: "f32[8, 512, 1536]", view_238: "f32[4096, 1536]", addmm_64: "f32[4096, 6144]", view_240: "f32[4096, 6144]", gt_33: "b8[8, 512, 1536]", mul_158: "f32[8, 512, 1536]", view_242: "f32[4096, 1536]", div_23: "f32[8, 24, 512, 512]", gt_34: "b8[8, 24, 512, 512]", view_258: "f32[4096, 1536]", gt_35: "b8[8, 512, 1536]", mul_165: "f32[8, 512, 1536]", view_260: "f32[4096, 1536]", addmm_70: "f32[4096, 6144]", view_262: "f32[4096, 6144]", gt_36: "b8[8, 512, 1536]", mul_172: "f32[8, 512, 1536]", view_264: "f32[4096, 1536]", div_25: "f32[8, 24, 512, 512]", gt_37: "b8[8, 24, 512, 512]", view_280: "f32[4096, 1536]", gt_38: "b8[8, 512, 1536]", mul_179: "f32[8, 512, 1536]", view_282: "f32[4096, 1536]", addmm_76: "f32[4096, 6144]", view_284: "f32[4096, 6144]", gt_39: "b8[8, 512, 1536]", mul_186: "f32[8, 512, 1536]", view_286: "f32[4096, 1536]", div_27: "f32[8, 24, 512, 512]", gt_40: "b8[8, 24, 512, 512]", view_302: "f32[4096, 1536]", gt_41: "b8[8, 512, 1536]", mul_193: "f32[8, 512, 1536]", view_304: "f32[4096, 1536]", addmm_82: "f32[4096, 6144]", view_306: "f32[4096, 6144]", gt_42: "b8[8, 512, 1536]", mul_200: "f32[8, 512, 1536]", view_308: "f32[4096, 1536]", div_29: "f32[8, 24, 512, 512]", gt_43: "b8[8, 24, 512, 512]", view_324: "f32[4096, 1536]", gt_44: "b8[8, 512, 1536]", mul_207: "f32[8, 512, 1536]", view_326: "f32[4096, 1536]", addmm_88: "f32[4096, 6144]", view_328: "f32[4096, 6144]", gt_45: "b8[8, 512, 1536]", mul_214: "f32[8, 512, 1536]", view_330: "f32[4096, 1536]", div_31: "f32[8, 24, 512, 512]", gt_46: "b8[8, 24, 512, 512]", view_346: "f32[4096, 1536]", gt_47: "b8[8, 512, 1536]", mul_221: "f32[8, 512, 1536]", view_348: "f32[4096, 1536]", addmm_94: "f32[4096, 6144]", view_350: "f32[4096, 6144]", gt_48: "b8[8, 512, 1536]", mul_228: "f32[8, 512, 1536]", view_352: "f32[4096, 1536]", div_33: "f32[8, 24, 512, 512]", gt_49: "b8[8, 24, 512, 512]", view_368: "f32[4096, 1536]", gt_50: "b8[8, 512, 1536]", mul_235: "f32[8, 512, 1536]", view_370: "f32[4096, 1536]", addmm_100: "f32[4096, 6144]", view_372: "f32[4096, 6144]", gt_51: "b8[8, 512, 1536]", mul_242: "f32[8, 512, 1536]", view_374: "f32[4096, 1536]", div_35: "f32[8, 24, 512, 512]", gt_52: "b8[8, 24, 512, 512]", view_390: "f32[4096, 1536]", gt_53: "b8[8, 512, 1536]", mul_249: "f32[8, 512, 1536]", view_392: "f32[4096, 1536]", addmm_106: "f32[4096, 6144]", view_394: "f32[4096, 6144]", gt_54: "b8[8, 512, 1536]", mul_256: "f32[8, 512, 1536]", view_396: "f32[4096, 1536]", div_37: "f32[8, 24, 512, 512]", gt_55: "b8[8, 24, 512, 512]", view_412: "f32[4096, 1536]", gt_56: "b8[8, 512, 1536]", mul_263: "f32[8, 512, 1536]", view_414: "f32[4096, 1536]", addmm_112: "f32[4096, 6144]", view_416: "f32[4096, 6144]", gt_57: "b8[8, 512, 1536]", mul_270: "f32[8, 512, 1536]", view_418: "f32[4096, 1536]", div_39: "f32[8, 24, 512, 512]", gt_58: "b8[8, 24, 512, 512]", view_434: "f32[4096, 1536]", gt_59: "b8[8, 512, 1536]", mul_277: "f32[8, 512, 1536]", view_436: "f32[4096, 1536]", addmm_118: "f32[4096, 6144]", view_438: "f32[4096, 6144]", gt_60: "b8[8, 512, 1536]", mul_284: "f32[8, 512, 1536]", view_440: "f32[4096, 1536]", div_41: "f32[8, 24, 512, 512]", gt_61: "b8[8, 24, 512, 512]", view_456: "f32[4096, 1536]", gt_62: "b8[8, 512, 1536]", mul_291: "f32[8, 512, 1536]", view_458: "f32[4096, 1536]", addmm_124: "f32[4096, 6144]", view_460: "f32[4096, 6144]", gt_63: "b8[8, 512, 1536]", mul_298: "f32[8, 512, 1536]", view_462: "f32[4096, 1536]", div_43: "f32[8, 24, 512, 512]", gt_64: "b8[8, 24, 512, 512]", view_478: "f32[4096, 1536]", gt_65: "b8[8, 512, 1536]", mul_305: "f32[8, 512, 1536]", view_480: "f32[4096, 1536]", addmm_130: "f32[4096, 6144]", view_482: "f32[4096, 6144]", gt_66: "b8[8, 512, 1536]", mul_312: "f32[8, 512, 1536]", view_484: "f32[4096, 1536]", div_45: "f32[8, 24, 512, 512]", gt_67: "b8[8, 24, 512, 512]", view_500: "f32[4096, 1536]", gt_68: "b8[8, 512, 1536]", mul_319: "f32[8, 512, 1536]", view_502: "f32[4096, 1536]", addmm_136: "f32[4096, 6144]", view_504: "f32[4096, 6144]", gt_69: "b8[8, 512, 1536]", mul_326: "f32[8, 512, 1536]", view_506: "f32[4096, 1536]", div_47: "f32[8, 24, 512, 512]", gt_70: "b8[8, 24, 512, 512]", view_522: "f32[4096, 1536]", gt_71: "b8[8, 512, 1536]", mul_333: "f32[8, 512, 1536]", view_524: "f32[4096, 1536]", addmm_142: "f32[4096, 6144]", view_526: "f32[4096, 6144]", gt_72: "b8[8, 512, 1536]", mul_340: "f32[8, 512, 1536]", view_528: "f32[4096, 1536]", addmm_144: "f32[4096, 1536]", getitem_99: "f32[8, 512, 1]", rsqrt_49: "f32[8, 512, 1]", view_530: "f32[4096, 1536]", view_531: "f32[8, 512, 128100]", amax_24: "f32[4096, 1]", log: "f32[4096, 1]", convert_element_type_24: "f32[]", div_51: "f32[8, 512, 1]", div_52: "f32[8, 512, 1]", permute_287: "f32[192, 512, 512]", permute_288: "f32[192, 64, 512]", permute_289: "f32[192, 64, 512]", permute_290: "f32[192, 512, 64]", div_54: "f32[8, 512, 1]", div_55: "f32[8, 512, 1]", permute_320: "f32[192, 512, 512]", permute_321: "f32[192, 64, 512]", permute_322: "f32[192, 64, 512]", permute_323: "f32[192, 512, 64]", div_57: "f32[8, 512, 1]", div_58: "f32[8, 512, 1]", permute_353: "f32[192, 512, 512]", permute_354: "f32[192, 64, 512]", permute_355: "f32[192, 64, 512]", permute_356: "f32[192, 512, 64]", div_60: "f32[8, 512, 1]", div_61: "f32[8, 512, 1]", permute_386: "f32[192, 512, 512]", permute_387: "f32[192, 64, 512]", permute_388: "f32[192, 64, 512]", permute_389: "f32[192, 512, 64]", div_63: "f32[8, 512, 1]", div_64: "f32[8, 512, 1]", permute_419: "f32[192, 512, 512]", permute_420: "f32[192, 64, 512]", permute_421: "f32[192, 64, 512]", permute_422: "f32[192, 512, 64]", div_66: "f32[8, 512, 1]", div_67: "f32[8, 512, 1]", permute_452: "f32[192, 512, 512]", permute_453: "f32[192, 64, 512]", permute_454: "f32[192, 64, 512]", permute_455: "f32[192, 512, 64]", div_69: "f32[8, 512, 1]", div_70: "f32[8, 512, 1]", permute_485: "f32[192, 512, 512]", permute_486: "f32[192, 64, 512]", permute_487: "f32[192, 64, 512]", permute_488: "f32[192, 512, 64]", div_72: "f32[8, 512, 1]", div_73: "f32[8, 512, 1]", permute_518: "f32[192, 512, 512]", permute_519: "f32[192, 64, 512]", permute_520: "f32[192, 64, 512]", permute_521: "f32[192, 512, 64]", div_75: "f32[8, 512, 1]", div_76: "f32[8, 512, 1]", permute_551: "f32[192, 512, 512]", permute_552: "f32[192, 64, 512]", permute_553: "f32[192, 64, 512]", permute_554: "f32[192, 512, 64]", div_78: "f32[8, 512, 1]", div_79: "f32[8, 512, 1]", permute_584: "f32[192, 512, 512]", permute_585: "f32[192, 64, 512]", permute_586: "f32[192, 64, 512]", permute_587: "f32[192, 512, 64]", div_81: "f32[8, 512, 1]", div_82: "f32[8, 512, 1]", permute_617: "f32[192, 512, 512]", permute_618: "f32[192, 64, 512]", permute_619: "f32[192, 64, 512]", permute_620: "f32[192, 512, 64]", div_84: "f32[8, 512, 1]", div_85: "f32[8, 512, 1]", permute_650: "f32[192, 512, 512]", permute_651: "f32[192, 64, 512]", permute_652: "f32[192, 64, 512]", permute_653: "f32[192, 512, 64]", div_87: "f32[8, 512, 1]", div_88: "f32[8, 512, 1]", permute_683: "f32[192, 512, 512]", permute_684: "f32[192, 64, 512]", permute_685: "f32[192, 64, 512]", permute_686: "f32[192, 512, 64]", div_90: "f32[8, 512, 1]", div_91: "f32[8, 512, 1]", permute_716: "f32[192, 512, 512]", permute_717: "f32[192, 64, 512]", permute_718: "f32[192, 64, 512]", permute_719: "f32[192, 512, 64]", div_93: "f32[8, 512, 1]", div_94: "f32[8, 512, 1]", permute_749: "f32[192, 512, 512]", permute_750: "f32[192, 64, 512]", permute_751: "f32[192, 64, 512]", permute_752: "f32[192, 512, 64]", div_96: "f32[8, 512, 1]", div_97: "f32[8, 512, 1]", permute_782: "f32[192, 512, 512]", permute_783: "f32[192, 64, 512]", permute_784: "f32[192, 64, 512]", permute_785: "f32[192, 512, 64]", div_99: "f32[8, 512, 1]", div_100: "f32[8, 512, 1]", permute_815: "f32[192, 512, 512]", permute_816: "f32[192, 64, 512]", permute_817: "f32[192, 64, 512]", permute_818: "f32[192, 512, 64]", div_102: "f32[8, 512, 1]", div_103: "f32[8, 512, 1]", permute_848: "f32[192, 512, 512]", permute_849: "f32[192, 64, 512]", permute_850: "f32[192, 64, 512]", permute_851: "f32[192, 512, 64]", div_105: "f32[8, 512, 1]", div_106: "f32[8, 512, 1]", permute_881: "f32[192, 512, 512]", permute_882: "f32[192, 64, 512]", permute_883: "f32[192, 64, 512]", permute_884: "f32[192, 512, 64]", div_108: "f32[8, 512, 1]", div_109: "f32[8, 512, 1]", permute_914: "f32[192, 512, 512]", permute_915: "f32[192, 64, 512]", permute_916: "f32[192, 64, 512]", permute_917: "f32[192, 512, 64]", div_111: "f32[8, 512, 1]", div_112: "f32[8, 512, 1]", permute_947: "f32[192, 512, 512]", permute_948: "f32[192, 64, 512]", permute_949: "f32[192, 64, 512]", permute_950: "f32[192, 512, 64]", div_114: "f32[8, 512, 1]", div_115: "f32[8, 512, 1]", permute_980: "f32[192, 512, 512]", permute_981: "f32[192, 64, 512]", permute_982: "f32[192, 64, 512]", permute_983: "f32[192, 512, 64]", div_117: "f32[8, 512, 1]", div_118: "f32[8, 512, 1]", permute_1013: "f32[192, 512, 512]", permute_1014: "f32[192, 64, 512]", permute_1015: "f32[192, 64, 512]", permute_1016: "f32[192, 512, 64]", div_120: "f32[8, 512, 1]", div_121: "f32[8, 512, 1]", permute_1046: "f32[192, 512, 512]", permute_1047: "f32[192, 64, 512]", permute_1048: "f32[192, 64, 512]", permute_1049: "f32[192, 512, 64]", tangents_1: "f32[]", tangents_2: "f32[8, 512, 128100]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:968 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        div_49: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_24);  tangents_1 = convert_element_type_24 = None
        view_533: "i64[4096]" = torch.ops.aten.reshape.default(primals_396, [-1]);  primals_396 = None
        unsqueeze_5: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(view_533, 1);  view_533 = None
        ne_3: "b8[4096, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_5, -100)
        full_default_73: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "i64[4096, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_5, full_default_73);  unsqueeze_5 = full_default_73 = None

        # No stacktrace found for following nodes
        iota_default: "i64[128100]" = torch.ops.prims.iota.default(128100, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 128100]" = torch.ops.aten.reshape.default(iota_default, [1, 128100]);  iota_default = None
        expand_default: "i64[4096, 128100]" = torch.ops.aten.expand.default(where_26, [4096, 128100]);  where_26 = None
        eq_tensor: "b8[4096, 128100]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:968 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        where_self: "f32[4096, 128100]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_74: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27: "f32[4096, 1]" = torch.ops.aten.where.self(ne_3, div_49, full_default_74);  ne_3 = div_49 = None
        mul_347: "f32[4096, 128100]" = torch.ops.aten.mul.Tensor(where_self, where_27);  where_self = where_27 = None
        view_532: "f32[4096, 128100]" = torch.ops.aten.reshape.default(view_531, [-1, 128100]);  view_531 = None
        sub_74: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(view_532, amax_24);  view_532 = amax_24 = None
        sub_75: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(sub_74, log);  sub_74 = log = None
        exp_25: "f32[4096, 128100]" = torch.ops.aten.exp.default(sub_75);  sub_75 = None
        sum_28: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [1], True)
        mul_348: "f32[4096, 128100]" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
        sub_76: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
        view_534: "f32[8, 512, 128100]" = torch.ops.aten.reshape.default(sub_76, [8, 512, 128100]);  sub_76 = None
        add_174: "f32[8, 512, 128100]" = torch.ops.aten.add.Tensor(tangents_2, view_534);  tangents_2 = view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:838 in forward, code: hidden_states = self.decoder(hidden_states)
        view_535: "f32[4096, 128100]" = torch.ops.aten.reshape.default(add_174, [4096, 128100]);  add_174 = None
        permute_265: "f32[1536, 128100]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        permute_266: "f32[128100, 1536]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        mm: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_535, permute_266);  permute_266 = None
        permute_267: "f32[128100, 4096]" = torch.ops.aten.permute.default(view_535, [1, 0])
        mm_1: "f32[128100, 1536]" = torch.ops.aten.mm.default(permute_267, view_530);  permute_267 = view_530 = None
        sum_29: "f32[1, 128100]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
        view_536: "f32[128100]" = torch.ops.aten.reshape.default(sum_29, [128100]);  sum_29 = None
        view_537: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm, [8, 512, 1536]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:820 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_350: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(view_537, primals_393);  primals_393 = None
        mul_351: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_350, 1536)
        sum_30: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_350, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:818 in forward, code: hidden_states = self.dense(hidden_states)
        view_529: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(addmm_144, [8, 512, 1536]);  addmm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_342: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(view_529, 0.5)
        mul_343: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(view_529, 0.7071067811865476)
        erf_24: "f32[8, 512, 1536]" = torch.ops.aten.erf.default(mul_343);  mul_343 = None
        add_171: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_344: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_342, add_171);  mul_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:820 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_73: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_344, getitem_99);  mul_344 = getitem_99 = None
        mul_345: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_49);  sub_73 = None
        mul_352: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_350, mul_345);  mul_350 = None
        sum_31: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [2], True);  mul_352 = None
        mul_353: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_345, sum_31);  sum_31 = None
        sub_78: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_351, sum_30);  mul_351 = sum_30 = None
        sub_79: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_78, mul_353);  sub_78 = mul_353 = None
        div_50: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 1536);  rsqrt_49 = None
        mul_354: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_50, sub_79);  div_50 = sub_79 = None
        mul_355: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(view_537, mul_345);  mul_345 = None
        sum_32: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_355, [0, 1]);  mul_355 = None
        sum_33: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_537, [0, 1]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_357: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_171, 0.5);  add_171 = None
        mul_358: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(view_529, view_529)
        mul_359: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_358, -0.5);  mul_358 = None
        exp_26: "f32[8, 512, 1536]" = torch.ops.aten.exp.default(mul_359);  mul_359 = None
        mul_360: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_361: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(view_529, mul_360);  view_529 = mul_360 = None
        add_176: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_357, mul_361);  mul_357 = mul_361 = None
        mul_362: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_354, add_176);  mul_354 = add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:818 in forward, code: hidden_states = self.dense(hidden_states)
        view_538: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_362, [4096, 1536]);  mul_362 = None
        permute_264: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_391, [1, 0]);  primals_391 = None
        permute_270: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        mm_2: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_538, permute_270);  permute_270 = None
        permute_271: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_538, [1, 0])
        mm_3: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_271, view_528);  permute_271 = view_528 = None
        sum_34: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_538, [0], True);  view_538 = None
        view_539: "f32[1536]" = torch.ops.aten.reshape.default(sum_34, [1536]);  sum_34 = None
        view_540: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_2, [8, 512, 1536]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_364: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(view_540, primals_389);  primals_389 = None
        mul_365: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_364, 1536)
        sum_35: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_364, [2], True)
        mul_366: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_364, mul_340);  mul_364 = None
        sum_36: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_366, [2], True);  mul_366 = None
        mul_367: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_340, sum_36);  sum_36 = None
        sub_81: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_365, sum_35);  mul_365 = sum_35 = None
        sub_82: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_81, mul_367);  sub_81 = mul_367 = None
        mul_368: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_51, sub_82);  div_51 = sub_82 = None
        mul_369: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(view_540, mul_340);  mul_340 = None
        sum_37: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_369, [0, 1]);  mul_369 = None
        sum_38: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_540, [0, 1]);  view_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_25: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_72, torch.float32);  gt_72 = None
        mul_370: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_371: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_368, mul_370);  mul_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_541: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_371, [4096, 1536]);  mul_371 = None
        permute_263: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_387, [1, 0]);  primals_387 = None
        permute_274: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        mm_4: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_541, permute_274);  permute_274 = None
        permute_275: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_541, [1, 0])
        mm_5: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_275, view_526);  permute_275 = view_526 = None
        sum_39: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_541, [0], True);  view_541 = None
        view_542: "f32[1536]" = torch.ops.aten.reshape.default(sum_39, [1536]);  sum_39 = None
        view_543: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_4, [8, 512, 6144]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_525: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_142, [8, 512, 6144]);  addmm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_336: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_525, 0.7071067811865476)
        erf_23: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_336);  mul_336 = None
        add_167: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_373: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_167, 0.5);  add_167 = None
        mul_374: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_525, view_525)
        mul_375: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_374, -0.5);  mul_374 = None
        exp_27: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_375);  mul_375 = None
        mul_376: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
        mul_377: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_525, mul_376);  view_525 = mul_376 = None
        add_178: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_373, mul_377);  mul_373 = mul_377 = None
        mul_378: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_543, add_178);  view_543 = add_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_544: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_378, [4096, 6144]);  mul_378 = None
        permute_262: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_385, [1, 0]);  primals_385 = None
        permute_278: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
        mm_6: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_544, permute_278);  permute_278 = None
        permute_279: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_544, [1, 0])
        mm_7: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_279, view_524);  permute_279 = view_524 = None
        sum_40: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_544, [0], True);  view_544 = None
        view_545: "f32[6144]" = torch.ops.aten.reshape.default(sum_40, [6144]);  sum_40 = None
        view_546: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_6, [8, 512, 1536]);  mm_6 = None
        add_179: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_368, view_546);  mul_368 = view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_380: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_179, primals_383);  primals_383 = None
        mul_381: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_380, 1536)
        sum_41: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [2], True)
        mul_382: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_380, mul_333);  mul_380 = None
        sum_42: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_382, [2], True);  mul_382 = None
        mul_383: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_333, sum_42);  sum_42 = None
        sub_84: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_381, sum_41);  mul_381 = sum_41 = None
        sub_85: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_84, mul_383);  sub_84 = mul_383 = None
        mul_384: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_52, sub_85);  div_52 = sub_85 = None
        mul_385: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_179, mul_333);  mul_333 = None
        sum_43: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_385, [0, 1]);  mul_385 = None
        sum_44: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_179, [0, 1]);  add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_26: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_71, torch.float32);  gt_71 = None
        mul_386: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_387: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_384, mul_386);  mul_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_547: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_387, [4096, 1536]);  mul_387 = None
        permute_261: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_381, [1, 0]);  primals_381 = None
        permute_282: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
        mm_8: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_547, permute_282);  permute_282 = None
        permute_283: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_547, [1, 0])
        mm_9: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_283, view_522);  permute_283 = view_522 = None
        sum_45: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_547, [0], True);  view_547 = None
        view_548: "f32[1536]" = torch.ops.aten.reshape.default(sum_45, [1536]);  sum_45 = None
        view_549: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_8, [8, 512, 1536]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_550: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_549, [8, 512, 24, 64]);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_286: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_550, [0, 2, 1, 3]);  view_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_98: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
        view_551: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_98, [192, 512, 64]);  clone_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_48: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_287, view_551);  permute_287 = None
        bmm_49: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_551, permute_288);  view_551 = permute_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_552: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_49, [8, 24, 512, 512]);  bmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_27: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_70, torch.float32);  gt_70 = None
        mul_388: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
        mul_389: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_552, mul_388);  view_552 = mul_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_390: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_389, div_47);  mul_389 = None
        sum_46: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [-1], True)
        neg_1: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_47);  div_47 = None
        fma: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_1, sum_46, mul_390);  neg_1 = sum_46 = mul_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_2: "b8[8, 1, 512, 512]" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_553: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_28, [192, 512, 512]);  where_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_50: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_289, view_553);  permute_289 = None
        bmm_51: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_553, permute_290);  view_553 = permute_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:121 in scaled_size_sqrt, code: return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        div_53: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_50, full_default_1);  bmm_50 = None
        permute_291: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_53, [0, 2, 1]);  div_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_554: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_48, [8, 24, 512, 64]);  bmm_48 = None
        permute_292: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_100: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
        view_555: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_100, [8, 512, 1536]);  clone_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_556: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_555, [4096, 1536]);  view_555 = None
        permute_257: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_379, [1, 0]);  primals_379 = None
        permute_293: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
        mm_10: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_556, permute_293);  permute_293 = None
        permute_294: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_556, [1, 0])
        mm_11: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_294, view_506);  permute_294 = None
        sum_47: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_556, [0], True);  view_556 = None
        view_557: "f32[1536]" = torch.ops.aten.reshape.default(sum_47, [1536]);  sum_47 = None
        view_558: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_10, [8, 512, 1536]);  mm_10 = None
        add_180: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_384, view_558);  mul_384 = view_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_559: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_291, [8, 24, 512, 64]);  permute_291 = None
        permute_297: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_559, [0, 2, 1, 3]);  view_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_560: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_297, [8, 512, 1536]);  permute_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_101: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_560, memory_format = torch.contiguous_format);  view_560 = None
        view_561: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_101, [4096, 1536]);  clone_101 = None
        permute_255: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_377, [1, 0]);  primals_377 = None
        permute_298: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
        mm_12: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_561, permute_298);  permute_298 = None
        permute_299: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_561, [1, 0])
        mm_13: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_299, view_506);  permute_299 = None
        sum_48: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_561, [0], True);  view_561 = None
        view_562: "f32[1536]" = torch.ops.aten.reshape.default(sum_48, [1536]);  sum_48 = None
        view_563: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_12, [8, 512, 1536]);  mm_12 = None
        add_181: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_180, view_563);  add_180 = view_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_564: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_51, [8, 24, 512, 64]);  bmm_51 = None
        permute_302: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_564, [0, 2, 1, 3]);  view_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_102: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_302, memory_format = torch.contiguous_format);  permute_302 = None
        view_565: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_102, [8, 512, 1536]);  clone_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_566: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_565, [4096, 1536]);  view_565 = None
        permute_253: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_375, [1, 0]);  primals_375 = None
        permute_303: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
        mm_14: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_566, permute_303);  permute_303 = None
        permute_304: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_566, [1, 0])
        mm_15: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_304, view_506);  permute_304 = view_506 = None
        sum_49: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_566, [0], True);  view_566 = None
        view_567: "f32[1536]" = torch.ops.aten.reshape.default(sum_49, [1536]);  sum_49 = None
        view_568: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_14, [8, 512, 1536]);  mm_14 = None
        add_182: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_181, view_568);  add_181 = view_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_392: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_182, primals_373);  primals_373 = None
        mul_393: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_392, 1536)
        sum_50: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_392, [2], True)
        mul_394: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_392, mul_326);  mul_392 = None
        sum_51: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_394, [2], True);  mul_394 = None
        mul_395: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_326, sum_51);  sum_51 = None
        sub_87: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_393, sum_50);  mul_393 = sum_50 = None
        sub_88: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_87, mul_395);  sub_87 = mul_395 = None
        mul_396: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_54, sub_88);  div_54 = sub_88 = None
        mul_397: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_182, mul_326);  mul_326 = None
        sum_52: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_397, [0, 1]);  mul_397 = None
        sum_53: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_182, [0, 1]);  add_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_28: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_69, torch.float32);  gt_69 = None
        mul_398: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
        mul_399: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_396, mul_398);  mul_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_569: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_399, [4096, 1536]);  mul_399 = None
        permute_252: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_371, [1, 0]);  primals_371 = None
        permute_307: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
        mm_16: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_569, permute_307);  permute_307 = None
        permute_308: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_569, [1, 0])
        mm_17: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_308, view_504);  permute_308 = view_504 = None
        sum_54: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_569, [0], True);  view_569 = None
        view_570: "f32[1536]" = torch.ops.aten.reshape.default(sum_54, [1536]);  sum_54 = None
        view_571: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_16, [8, 512, 6144]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_503: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_136, [8, 512, 6144]);  addmm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_322: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_503, 0.7071067811865476)
        erf_22: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_322);  mul_322 = None
        add_160: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_401: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_160, 0.5);  add_160 = None
        mul_402: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_503, view_503)
        mul_403: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_402, -0.5);  mul_402 = None
        exp_28: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_403);  mul_403 = None
        mul_404: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
        mul_405: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_503, mul_404);  view_503 = mul_404 = None
        add_184: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_401, mul_405);  mul_401 = mul_405 = None
        mul_406: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_571, add_184);  view_571 = add_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_572: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_406, [4096, 6144]);  mul_406 = None
        permute_251: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_369, [1, 0]);  primals_369 = None
        permute_311: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
        mm_18: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_572, permute_311);  permute_311 = None
        permute_312: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_572, [1, 0])
        mm_19: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_312, view_502);  permute_312 = view_502 = None
        sum_55: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_572, [0], True);  view_572 = None
        view_573: "f32[6144]" = torch.ops.aten.reshape.default(sum_55, [6144]);  sum_55 = None
        view_574: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_18, [8, 512, 1536]);  mm_18 = None
        add_185: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_396, view_574);  mul_396 = view_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_408: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_185, primals_367);  primals_367 = None
        mul_409: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_408, 1536)
        sum_56: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_408, [2], True)
        mul_410: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_408, mul_319);  mul_408 = None
        sum_57: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_410, [2], True);  mul_410 = None
        mul_411: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_319, sum_57);  sum_57 = None
        sub_90: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_409, sum_56);  mul_409 = sum_56 = None
        sub_91: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_90, mul_411);  sub_90 = mul_411 = None
        mul_412: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_55, sub_91);  div_55 = sub_91 = None
        mul_413: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_185, mul_319);  mul_319 = None
        sum_58: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_413, [0, 1]);  mul_413 = None
        sum_59: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_185, [0, 1]);  add_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_29: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_68, torch.float32);  gt_68 = None
        mul_414: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
        mul_415: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_412, mul_414);  mul_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_575: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_415, [4096, 1536]);  mul_415 = None
        permute_250: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_365, [1, 0]);  primals_365 = None
        permute_315: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
        mm_20: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_575, permute_315);  permute_315 = None
        permute_316: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_575, [1, 0])
        mm_21: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_316, view_500);  permute_316 = view_500 = None
        sum_60: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_575, [0], True);  view_575 = None
        view_576: "f32[1536]" = torch.ops.aten.reshape.default(sum_60, [1536]);  sum_60 = None
        view_577: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_20, [8, 512, 1536]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_578: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_577, [8, 512, 24, 64]);  view_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_319: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_105: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
        view_579: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_105, [192, 512, 64]);  clone_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_52: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_320, view_579);  permute_320 = None
        bmm_53: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_579, permute_321);  view_579 = permute_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_580: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_53, [8, 24, 512, 512]);  bmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_30: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_67, torch.float32);  gt_67 = None
        mul_416: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
        mul_417: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_580, mul_416);  view_580 = mul_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_418: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_417, div_45);  mul_417 = None
        sum_61: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [-1], True)
        neg_2: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_45);  div_45 = None
        fma_1: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_2, sum_61, mul_418);  neg_2 = sum_61 = mul_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_29: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_1);  fma_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_581: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_29, [192, 512, 512]);  where_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_54: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_322, view_581);  permute_322 = None
        bmm_55: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_581, permute_323);  view_581 = permute_323 = None
        div_56: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_54, full_default_1);  bmm_54 = None
        permute_324: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_56, [0, 2, 1]);  div_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_582: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_52, [8, 24, 512, 64]);  bmm_52 = None
        permute_325: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_582, [0, 2, 1, 3]);  view_582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_107: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
        view_583: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_107, [8, 512, 1536]);  clone_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_584: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_583, [4096, 1536]);  view_583 = None
        permute_246: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_363, [1, 0]);  primals_363 = None
        permute_326: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
        mm_22: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_584, permute_326);  permute_326 = None
        permute_327: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_584, [1, 0])
        mm_23: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_327, view_484);  permute_327 = None
        sum_62: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_584, [0], True);  view_584 = None
        view_585: "f32[1536]" = torch.ops.aten.reshape.default(sum_62, [1536]);  sum_62 = None
        view_586: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_22, [8, 512, 1536]);  mm_22 = None
        add_186: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_412, view_586);  mul_412 = view_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_587: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_324, [8, 24, 512, 64]);  permute_324 = None
        permute_330: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_587, [0, 2, 1, 3]);  view_587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_588: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_330, [8, 512, 1536]);  permute_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_108: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_588, memory_format = torch.contiguous_format);  view_588 = None
        view_589: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_108, [4096, 1536]);  clone_108 = None
        permute_244: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_361, [1, 0]);  primals_361 = None
        permute_331: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
        mm_24: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_589, permute_331);  permute_331 = None
        permute_332: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_589, [1, 0])
        mm_25: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_332, view_484);  permute_332 = None
        sum_63: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_589, [0], True);  view_589 = None
        view_590: "f32[1536]" = torch.ops.aten.reshape.default(sum_63, [1536]);  sum_63 = None
        view_591: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_24, [8, 512, 1536]);  mm_24 = None
        add_187: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_186, view_591);  add_186 = view_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_592: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_55, [8, 24, 512, 64]);  bmm_55 = None
        permute_335: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_592, [0, 2, 1, 3]);  view_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_109: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
        view_593: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_109, [8, 512, 1536]);  clone_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_594: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_593, [4096, 1536]);  view_593 = None
        permute_242: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_359, [1, 0]);  primals_359 = None
        permute_336: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        mm_26: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_594, permute_336);  permute_336 = None
        permute_337: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_594, [1, 0])
        mm_27: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_337, view_484);  permute_337 = view_484 = None
        sum_64: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_594, [0], True);  view_594 = None
        view_595: "f32[1536]" = torch.ops.aten.reshape.default(sum_64, [1536]);  sum_64 = None
        view_596: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_26, [8, 512, 1536]);  mm_26 = None
        add_188: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_187, view_596);  add_187 = view_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_420: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_188, primals_357);  primals_357 = None
        mul_421: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_420, 1536)
        sum_65: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_420, [2], True)
        mul_422: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_420, mul_312);  mul_420 = None
        sum_66: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_422, [2], True);  mul_422 = None
        mul_423: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_312, sum_66);  sum_66 = None
        sub_93: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_421, sum_65);  mul_421 = sum_65 = None
        sub_94: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_93, mul_423);  sub_93 = mul_423 = None
        mul_424: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_57, sub_94);  div_57 = sub_94 = None
        mul_425: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_188, mul_312);  mul_312 = None
        sum_67: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_425, [0, 1]);  mul_425 = None
        sum_68: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_188, [0, 1]);  add_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_31: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_66, torch.float32);  gt_66 = None
        mul_426: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
        mul_427: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_424, mul_426);  mul_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_597: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_427, [4096, 1536]);  mul_427 = None
        permute_241: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_355, [1, 0]);  primals_355 = None
        permute_340: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        mm_28: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_597, permute_340);  permute_340 = None
        permute_341: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_597, [1, 0])
        mm_29: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_341, view_482);  permute_341 = view_482 = None
        sum_69: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_597, [0], True);  view_597 = None
        view_598: "f32[1536]" = torch.ops.aten.reshape.default(sum_69, [1536]);  sum_69 = None
        view_599: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_28, [8, 512, 6144]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_481: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_130, [8, 512, 6144]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_308: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_481, 0.7071067811865476)
        erf_21: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_308);  mul_308 = None
        add_153: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_429: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_153, 0.5);  add_153 = None
        mul_430: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_481, view_481)
        mul_431: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_430, -0.5);  mul_430 = None
        exp_29: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_431);  mul_431 = None
        mul_432: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
        mul_433: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_481, mul_432);  view_481 = mul_432 = None
        add_190: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_429, mul_433);  mul_429 = mul_433 = None
        mul_434: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_599, add_190);  view_599 = add_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_600: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_434, [4096, 6144]);  mul_434 = None
        permute_240: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_353, [1, 0]);  primals_353 = None
        permute_344: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
        mm_30: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_600, permute_344);  permute_344 = None
        permute_345: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_600, [1, 0])
        mm_31: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_345, view_480);  permute_345 = view_480 = None
        sum_70: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_600, [0], True);  view_600 = None
        view_601: "f32[6144]" = torch.ops.aten.reshape.default(sum_70, [6144]);  sum_70 = None
        view_602: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_30, [8, 512, 1536]);  mm_30 = None
        add_191: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_424, view_602);  mul_424 = view_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_436: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_191, primals_351);  primals_351 = None
        mul_437: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_436, 1536)
        sum_71: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_436, [2], True)
        mul_438: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_436, mul_305);  mul_436 = None
        sum_72: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_438, [2], True);  mul_438 = None
        mul_439: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_305, sum_72);  sum_72 = None
        sub_96: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_437, sum_71);  mul_437 = sum_71 = None
        sub_97: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_96, mul_439);  sub_96 = mul_439 = None
        mul_440: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_58, sub_97);  div_58 = sub_97 = None
        mul_441: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_191, mul_305);  mul_305 = None
        sum_73: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_441, [0, 1]);  mul_441 = None
        sum_74: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_191, [0, 1]);  add_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_32: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_65, torch.float32);  gt_65 = None
        mul_442: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
        mul_443: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_440, mul_442);  mul_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_603: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_443, [4096, 1536]);  mul_443 = None
        permute_239: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_349, [1, 0]);  primals_349 = None
        permute_348: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
        mm_32: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_603, permute_348);  permute_348 = None
        permute_349: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_603, [1, 0])
        mm_33: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_349, view_478);  permute_349 = view_478 = None
        sum_75: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_603, [0], True);  view_603 = None
        view_604: "f32[1536]" = torch.ops.aten.reshape.default(sum_75, [1536]);  sum_75 = None
        view_605: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_32, [8, 512, 1536]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_606: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_605, [8, 512, 24, 64]);  view_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_352: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_606, [0, 2, 1, 3]);  view_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_112: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None
        view_607: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_112, [192, 512, 64]);  clone_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_56: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_353, view_607);  permute_353 = None
        bmm_57: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_607, permute_354);  view_607 = permute_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_608: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_57, [8, 24, 512, 512]);  bmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_33: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_64, torch.float32);  gt_64 = None
        mul_444: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
        mul_445: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_608, mul_444);  view_608 = mul_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_446: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_445, div_43);  mul_445 = None
        sum_76: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_446, [-1], True)
        neg_3: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_43);  div_43 = None
        fma_2: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_3, sum_76, mul_446);  neg_3 = sum_76 = mul_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_30: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_2);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_609: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_30, [192, 512, 512]);  where_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_58: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_355, view_609);  permute_355 = None
        bmm_59: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_609, permute_356);  view_609 = permute_356 = None
        div_59: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_58, full_default_1);  bmm_58 = None
        permute_357: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_59, [0, 2, 1]);  div_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_610: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_56, [8, 24, 512, 64]);  bmm_56 = None
        permute_358: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_610, [0, 2, 1, 3]);  view_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_114: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
        view_611: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_114, [8, 512, 1536]);  clone_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_612: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_611, [4096, 1536]);  view_611 = None
        permute_235: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_347, [1, 0]);  primals_347 = None
        permute_359: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
        mm_34: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_612, permute_359);  permute_359 = None
        permute_360: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_612, [1, 0])
        mm_35: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_360, view_462);  permute_360 = None
        sum_77: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_612, [0], True);  view_612 = None
        view_613: "f32[1536]" = torch.ops.aten.reshape.default(sum_77, [1536]);  sum_77 = None
        view_614: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_34, [8, 512, 1536]);  mm_34 = None
        add_192: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_440, view_614);  mul_440 = view_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_615: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_357, [8, 24, 512, 64]);  permute_357 = None
        permute_363: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_615, [0, 2, 1, 3]);  view_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_616: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_363, [8, 512, 1536]);  permute_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_115: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_616, memory_format = torch.contiguous_format);  view_616 = None
        view_617: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_115, [4096, 1536]);  clone_115 = None
        permute_233: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_345, [1, 0]);  primals_345 = None
        permute_364: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        mm_36: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_617, permute_364);  permute_364 = None
        permute_365: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_617, [1, 0])
        mm_37: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_365, view_462);  permute_365 = None
        sum_78: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_617, [0], True);  view_617 = None
        view_618: "f32[1536]" = torch.ops.aten.reshape.default(sum_78, [1536]);  sum_78 = None
        view_619: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_36, [8, 512, 1536]);  mm_36 = None
        add_193: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_192, view_619);  add_192 = view_619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_620: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_59, [8, 24, 512, 64]);  bmm_59 = None
        permute_368: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_620, [0, 2, 1, 3]);  view_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_116: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
        view_621: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_116, [8, 512, 1536]);  clone_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_622: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_621, [4096, 1536]);  view_621 = None
        permute_231: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_343, [1, 0]);  primals_343 = None
        permute_369: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
        mm_38: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_622, permute_369);  permute_369 = None
        permute_370: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_622, [1, 0])
        mm_39: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_370, view_462);  permute_370 = view_462 = None
        sum_79: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_622, [0], True);  view_622 = None
        view_623: "f32[1536]" = torch.ops.aten.reshape.default(sum_79, [1536]);  sum_79 = None
        view_624: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_38, [8, 512, 1536]);  mm_38 = None
        add_194: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_193, view_624);  add_193 = view_624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_448: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_194, primals_341);  primals_341 = None
        mul_449: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_448, 1536)
        sum_80: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_448, [2], True)
        mul_450: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_448, mul_298);  mul_448 = None
        sum_81: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_450, [2], True);  mul_450 = None
        mul_451: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_298, sum_81);  sum_81 = None
        sub_99: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_449, sum_80);  mul_449 = sum_80 = None
        sub_100: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_99, mul_451);  sub_99 = mul_451 = None
        mul_452: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_60, sub_100);  div_60 = sub_100 = None
        mul_453: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_194, mul_298);  mul_298 = None
        sum_82: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_453, [0, 1]);  mul_453 = None
        sum_83: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_194, [0, 1]);  add_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_34: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_63, torch.float32);  gt_63 = None
        mul_454: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
        mul_455: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_452, mul_454);  mul_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_625: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_455, [4096, 1536]);  mul_455 = None
        permute_230: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_339, [1, 0]);  primals_339 = None
        permute_373: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
        mm_40: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_625, permute_373);  permute_373 = None
        permute_374: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_625, [1, 0])
        mm_41: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_374, view_460);  permute_374 = view_460 = None
        sum_84: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_625, [0], True);  view_625 = None
        view_626: "f32[1536]" = torch.ops.aten.reshape.default(sum_84, [1536]);  sum_84 = None
        view_627: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_40, [8, 512, 6144]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_459: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_124, [8, 512, 6144]);  addmm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_294: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_459, 0.7071067811865476)
        erf_20: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_294);  mul_294 = None
        add_146: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_457: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_146, 0.5);  add_146 = None
        mul_458: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_459, view_459)
        mul_459: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_458, -0.5);  mul_458 = None
        exp_30: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_459);  mul_459 = None
        mul_460: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
        mul_461: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_459, mul_460);  view_459 = mul_460 = None
        add_196: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_457, mul_461);  mul_457 = mul_461 = None
        mul_462: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_627, add_196);  view_627 = add_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_628: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_462, [4096, 6144]);  mul_462 = None
        permute_229: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_337, [1, 0]);  primals_337 = None
        permute_377: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
        mm_42: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_628, permute_377);  permute_377 = None
        permute_378: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_628, [1, 0])
        mm_43: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_378, view_458);  permute_378 = view_458 = None
        sum_85: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_628, [0], True);  view_628 = None
        view_629: "f32[6144]" = torch.ops.aten.reshape.default(sum_85, [6144]);  sum_85 = None
        view_630: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_42, [8, 512, 1536]);  mm_42 = None
        add_197: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_452, view_630);  mul_452 = view_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_464: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_197, primals_335);  primals_335 = None
        mul_465: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_464, 1536)
        sum_86: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True)
        mul_466: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_464, mul_291);  mul_464 = None
        sum_87: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_466, [2], True);  mul_466 = None
        mul_467: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_291, sum_87);  sum_87 = None
        sub_102: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_465, sum_86);  mul_465 = sum_86 = None
        sub_103: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_102, mul_467);  sub_102 = mul_467 = None
        mul_468: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_61, sub_103);  div_61 = sub_103 = None
        mul_469: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_197, mul_291);  mul_291 = None
        sum_88: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 1]);  mul_469 = None
        sum_89: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_197, [0, 1]);  add_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_35: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_62, torch.float32);  gt_62 = None
        mul_470: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
        mul_471: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_468, mul_470);  mul_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_631: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_471, [4096, 1536]);  mul_471 = None
        permute_228: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_333, [1, 0]);  primals_333 = None
        permute_381: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
        mm_44: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_631, permute_381);  permute_381 = None
        permute_382: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_631, [1, 0])
        mm_45: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_382, view_456);  permute_382 = view_456 = None
        sum_90: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_631, [0], True);  view_631 = None
        view_632: "f32[1536]" = torch.ops.aten.reshape.default(sum_90, [1536]);  sum_90 = None
        view_633: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_44, [8, 512, 1536]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_634: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_633, [8, 512, 24, 64]);  view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_385: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_634, [0, 2, 1, 3]);  view_634 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_119: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
        view_635: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_119, [192, 512, 64]);  clone_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_60: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_386, view_635);  permute_386 = None
        bmm_61: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_635, permute_387);  view_635 = permute_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_636: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_61, [8, 24, 512, 512]);  bmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_36: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_61, torch.float32);  gt_61 = None
        mul_472: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
        mul_473: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_636, mul_472);  view_636 = mul_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_474: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_473, div_41);  mul_473 = None
        sum_91: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_474, [-1], True)
        neg_4: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_41);  div_41 = None
        fma_3: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_4, sum_91, mul_474);  neg_4 = sum_91 = mul_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_31: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_3);  fma_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_637: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_31, [192, 512, 512]);  where_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_62: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_388, view_637);  permute_388 = None
        bmm_63: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_637, permute_389);  view_637 = permute_389 = None
        div_62: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_62, full_default_1);  bmm_62 = None
        permute_390: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_62, [0, 2, 1]);  div_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_638: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_60, [8, 24, 512, 64]);  bmm_60 = None
        permute_391: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_638, [0, 2, 1, 3]);  view_638 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_121: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
        view_639: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_121, [8, 512, 1536]);  clone_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_640: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_639, [4096, 1536]);  view_639 = None
        permute_224: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_331, [1, 0]);  primals_331 = None
        permute_392: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
        mm_46: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_640, permute_392);  permute_392 = None
        permute_393: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_640, [1, 0])
        mm_47: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_393, view_440);  permute_393 = None
        sum_92: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_640, [0], True);  view_640 = None
        view_641: "f32[1536]" = torch.ops.aten.reshape.default(sum_92, [1536]);  sum_92 = None
        view_642: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_46, [8, 512, 1536]);  mm_46 = None
        add_198: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_468, view_642);  mul_468 = view_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_643: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_390, [8, 24, 512, 64]);  permute_390 = None
        permute_396: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_643, [0, 2, 1, 3]);  view_643 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_644: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_396, [8, 512, 1536]);  permute_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_122: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_644, memory_format = torch.contiguous_format);  view_644 = None
        view_645: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_122, [4096, 1536]);  clone_122 = None
        permute_222: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_329, [1, 0]);  primals_329 = None
        permute_397: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        mm_48: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_645, permute_397);  permute_397 = None
        permute_398: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_645, [1, 0])
        mm_49: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_398, view_440);  permute_398 = None
        sum_93: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_645, [0], True);  view_645 = None
        view_646: "f32[1536]" = torch.ops.aten.reshape.default(sum_93, [1536]);  sum_93 = None
        view_647: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_48, [8, 512, 1536]);  mm_48 = None
        add_199: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_198, view_647);  add_198 = view_647 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_648: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_63, [8, 24, 512, 64]);  bmm_63 = None
        permute_401: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_123: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_401, memory_format = torch.contiguous_format);  permute_401 = None
        view_649: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_123, [8, 512, 1536]);  clone_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_650: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_649, [4096, 1536]);  view_649 = None
        permute_220: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_327, [1, 0]);  primals_327 = None
        permute_402: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
        mm_50: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_650, permute_402);  permute_402 = None
        permute_403: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_650, [1, 0])
        mm_51: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_403, view_440);  permute_403 = view_440 = None
        sum_94: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_650, [0], True);  view_650 = None
        view_651: "f32[1536]" = torch.ops.aten.reshape.default(sum_94, [1536]);  sum_94 = None
        view_652: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_50, [8, 512, 1536]);  mm_50 = None
        add_200: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_199, view_652);  add_199 = view_652 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_476: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_200, primals_325);  primals_325 = None
        mul_477: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_476, 1536)
        sum_95: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True)
        mul_478: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_476, mul_284);  mul_476 = None
        sum_96: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_478, [2], True);  mul_478 = None
        mul_479: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_284, sum_96);  sum_96 = None
        sub_105: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_477, sum_95);  mul_477 = sum_95 = None
        sub_106: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_105, mul_479);  sub_105 = mul_479 = None
        mul_480: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_63, sub_106);  div_63 = sub_106 = None
        mul_481: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_200, mul_284);  mul_284 = None
        sum_97: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 1]);  mul_481 = None
        sum_98: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_200, [0, 1]);  add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_37: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_60, torch.float32);  gt_60 = None
        mul_482: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
        mul_483: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_480, mul_482);  mul_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_653: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_483, [4096, 1536]);  mul_483 = None
        permute_219: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_323, [1, 0]);  primals_323 = None
        permute_406: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
        mm_52: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_653, permute_406);  permute_406 = None
        permute_407: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_653, [1, 0])
        mm_53: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_407, view_438);  permute_407 = view_438 = None
        sum_99: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_653, [0], True);  view_653 = None
        view_654: "f32[1536]" = torch.ops.aten.reshape.default(sum_99, [1536]);  sum_99 = None
        view_655: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_52, [8, 512, 6144]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_437: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_118, [8, 512, 6144]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_280: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_437, 0.7071067811865476)
        erf_19: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_280);  mul_280 = None
        add_139: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_485: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_139, 0.5);  add_139 = None
        mul_486: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_437, view_437)
        mul_487: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_486, -0.5);  mul_486 = None
        exp_31: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_487);  mul_487 = None
        mul_488: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
        mul_489: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_437, mul_488);  view_437 = mul_488 = None
        add_202: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_485, mul_489);  mul_485 = mul_489 = None
        mul_490: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_655, add_202);  view_655 = add_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_656: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_490, [4096, 6144]);  mul_490 = None
        permute_218: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_321, [1, 0]);  primals_321 = None
        permute_410: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
        mm_54: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_656, permute_410);  permute_410 = None
        permute_411: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_656, [1, 0])
        mm_55: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_411, view_436);  permute_411 = view_436 = None
        sum_100: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_656, [0], True);  view_656 = None
        view_657: "f32[6144]" = torch.ops.aten.reshape.default(sum_100, [6144]);  sum_100 = None
        view_658: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_54, [8, 512, 1536]);  mm_54 = None
        add_203: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_480, view_658);  mul_480 = view_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_492: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_203, primals_319);  primals_319 = None
        mul_493: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_492, 1536)
        sum_101: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_492, [2], True)
        mul_494: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_492, mul_277);  mul_492 = None
        sum_102: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_494, [2], True);  mul_494 = None
        mul_495: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_277, sum_102);  sum_102 = None
        sub_108: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_493, sum_101);  mul_493 = sum_101 = None
        sub_109: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_108, mul_495);  sub_108 = mul_495 = None
        mul_496: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_64, sub_109);  div_64 = sub_109 = None
        mul_497: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_203, mul_277);  mul_277 = None
        sum_103: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_497, [0, 1]);  mul_497 = None
        sum_104: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_203, [0, 1]);  add_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_38: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_59, torch.float32);  gt_59 = None
        mul_498: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
        mul_499: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_496, mul_498);  mul_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_659: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_499, [4096, 1536]);  mul_499 = None
        permute_217: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_317, [1, 0]);  primals_317 = None
        permute_414: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
        mm_56: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_659, permute_414);  permute_414 = None
        permute_415: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_659, [1, 0])
        mm_57: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_415, view_434);  permute_415 = view_434 = None
        sum_105: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_659, [0], True);  view_659 = None
        view_660: "f32[1536]" = torch.ops.aten.reshape.default(sum_105, [1536]);  sum_105 = None
        view_661: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_56, [8, 512, 1536]);  mm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_662: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_661, [8, 512, 24, 64]);  view_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_418: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_662, [0, 2, 1, 3]);  view_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_126: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
        view_663: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_126, [192, 512, 64]);  clone_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_64: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_419, view_663);  permute_419 = None
        bmm_65: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_663, permute_420);  view_663 = permute_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_664: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_65, [8, 24, 512, 512]);  bmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_39: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_58, torch.float32);  gt_58 = None
        mul_500: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.1111111111111112);  convert_element_type_39 = None
        mul_501: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_664, mul_500);  view_664 = mul_500 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_502: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_501, div_39);  mul_501 = None
        sum_106: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_502, [-1], True)
        neg_5: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_39);  div_39 = None
        fma_4: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_5, sum_106, mul_502);  neg_5 = sum_106 = mul_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_32: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_4);  fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_665: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_32, [192, 512, 512]);  where_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_66: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_421, view_665);  permute_421 = None
        bmm_67: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_665, permute_422);  view_665 = permute_422 = None
        div_65: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_66, full_default_1);  bmm_66 = None
        permute_423: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_65, [0, 2, 1]);  div_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_666: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_64, [8, 24, 512, 64]);  bmm_64 = None
        permute_424: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_666, [0, 2, 1, 3]);  view_666 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_128: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_424, memory_format = torch.contiguous_format);  permute_424 = None
        view_667: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_128, [8, 512, 1536]);  clone_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_668: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_667, [4096, 1536]);  view_667 = None
        permute_213: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_315, [1, 0]);  primals_315 = None
        permute_425: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
        mm_58: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_668, permute_425);  permute_425 = None
        permute_426: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_668, [1, 0])
        mm_59: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_426, view_418);  permute_426 = None
        sum_107: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_668, [0], True);  view_668 = None
        view_669: "f32[1536]" = torch.ops.aten.reshape.default(sum_107, [1536]);  sum_107 = None
        view_670: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_58, [8, 512, 1536]);  mm_58 = None
        add_204: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_496, view_670);  mul_496 = view_670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_671: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_423, [8, 24, 512, 64]);  permute_423 = None
        permute_429: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_671, [0, 2, 1, 3]);  view_671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_672: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_429, [8, 512, 1536]);  permute_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_129: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_672, memory_format = torch.contiguous_format);  view_672 = None
        view_673: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_129, [4096, 1536]);  clone_129 = None
        permute_211: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_313, [1, 0]);  primals_313 = None
        permute_430: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        mm_60: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_673, permute_430);  permute_430 = None
        permute_431: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_673, [1, 0])
        mm_61: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_431, view_418);  permute_431 = None
        sum_108: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_673, [0], True);  view_673 = None
        view_674: "f32[1536]" = torch.ops.aten.reshape.default(sum_108, [1536]);  sum_108 = None
        view_675: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_60, [8, 512, 1536]);  mm_60 = None
        add_205: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_204, view_675);  add_204 = view_675 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_676: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_67, [8, 24, 512, 64]);  bmm_67 = None
        permute_434: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_676, [0, 2, 1, 3]);  view_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_130: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_434, memory_format = torch.contiguous_format);  permute_434 = None
        view_677: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_130, [8, 512, 1536]);  clone_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_678: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_677, [4096, 1536]);  view_677 = None
        permute_209: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_311, [1, 0]);  primals_311 = None
        permute_435: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
        mm_62: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_678, permute_435);  permute_435 = None
        permute_436: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_678, [1, 0])
        mm_63: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_436, view_418);  permute_436 = view_418 = None
        sum_109: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_678, [0], True);  view_678 = None
        view_679: "f32[1536]" = torch.ops.aten.reshape.default(sum_109, [1536]);  sum_109 = None
        view_680: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_62, [8, 512, 1536]);  mm_62 = None
        add_206: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_205, view_680);  add_205 = view_680 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_504: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_206, primals_309);  primals_309 = None
        mul_505: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_504, 1536)
        sum_110: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_504, [2], True)
        mul_506: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_504, mul_270);  mul_504 = None
        sum_111: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_506, [2], True);  mul_506 = None
        mul_507: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_270, sum_111);  sum_111 = None
        sub_111: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_505, sum_110);  mul_505 = sum_110 = None
        sub_112: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_111, mul_507);  sub_111 = mul_507 = None
        mul_508: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_66, sub_112);  div_66 = sub_112 = None
        mul_509: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_206, mul_270);  mul_270 = None
        sum_112: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_509, [0, 1]);  mul_509 = None
        sum_113: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_206, [0, 1]);  add_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_40: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_57, torch.float32);  gt_57 = None
        mul_510: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
        mul_511: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_508, mul_510);  mul_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_681: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_511, [4096, 1536]);  mul_511 = None
        permute_208: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_307, [1, 0]);  primals_307 = None
        permute_439: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
        mm_64: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_681, permute_439);  permute_439 = None
        permute_440: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_681, [1, 0])
        mm_65: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_440, view_416);  permute_440 = view_416 = None
        sum_114: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_681, [0], True);  view_681 = None
        view_682: "f32[1536]" = torch.ops.aten.reshape.default(sum_114, [1536]);  sum_114 = None
        view_683: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_64, [8, 512, 6144]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_415: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_112, [8, 512, 6144]);  addmm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_266: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476)
        erf_18: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_266);  mul_266 = None
        add_132: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_513: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_132, 0.5);  add_132 = None
        mul_514: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_415, view_415)
        mul_515: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_514, -0.5);  mul_514 = None
        exp_32: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_515);  mul_515 = None
        mul_516: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
        mul_517: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_415, mul_516);  view_415 = mul_516 = None
        add_208: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_513, mul_517);  mul_513 = mul_517 = None
        mul_518: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_683, add_208);  view_683 = add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_684: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_518, [4096, 6144]);  mul_518 = None
        permute_207: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_305, [1, 0]);  primals_305 = None
        permute_443: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        mm_66: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_684, permute_443);  permute_443 = None
        permute_444: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_684, [1, 0])
        mm_67: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_444, view_414);  permute_444 = view_414 = None
        sum_115: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_684, [0], True);  view_684 = None
        view_685: "f32[6144]" = torch.ops.aten.reshape.default(sum_115, [6144]);  sum_115 = None
        view_686: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_66, [8, 512, 1536]);  mm_66 = None
        add_209: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_508, view_686);  mul_508 = view_686 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_520: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_209, primals_303);  primals_303 = None
        mul_521: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_520, 1536)
        sum_116: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_520, [2], True)
        mul_522: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_520, mul_263);  mul_520 = None
        sum_117: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [2], True);  mul_522 = None
        mul_523: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_263, sum_117);  sum_117 = None
        sub_114: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_521, sum_116);  mul_521 = sum_116 = None
        sub_115: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_114, mul_523);  sub_114 = mul_523 = None
        mul_524: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_67, sub_115);  div_67 = sub_115 = None
        mul_525: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_209, mul_263);  mul_263 = None
        sum_118: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_525, [0, 1]);  mul_525 = None
        sum_119: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_209, [0, 1]);  add_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_41: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_56, torch.float32);  gt_56 = None
        mul_526: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
        mul_527: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_524, mul_526);  mul_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_687: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_527, [4096, 1536]);  mul_527 = None
        permute_206: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_301, [1, 0]);  primals_301 = None
        permute_447: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
        mm_68: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_687, permute_447);  permute_447 = None
        permute_448: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_687, [1, 0])
        mm_69: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_448, view_412);  permute_448 = view_412 = None
        sum_120: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_687, [0], True);  view_687 = None
        view_688: "f32[1536]" = torch.ops.aten.reshape.default(sum_120, [1536]);  sum_120 = None
        view_689: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_68, [8, 512, 1536]);  mm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_690: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_689, [8, 512, 24, 64]);  view_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_451: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_690, [0, 2, 1, 3]);  view_690 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_133: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_451, memory_format = torch.contiguous_format);  permute_451 = None
        view_691: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_133, [192, 512, 64]);  clone_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_68: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_452, view_691);  permute_452 = None
        bmm_69: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_691, permute_453);  view_691 = permute_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_692: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_69, [8, 24, 512, 512]);  bmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_42: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_55, torch.float32);  gt_55 = None
        mul_528: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 1.1111111111111112);  convert_element_type_42 = None
        mul_529: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_692, mul_528);  view_692 = mul_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_530: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_529, div_37);  mul_529 = None
        sum_121: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_530, [-1], True)
        neg_6: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_37);  div_37 = None
        fma_5: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_6, sum_121, mul_530);  neg_6 = sum_121 = mul_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_33: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_5);  fma_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_693: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_33, [192, 512, 512]);  where_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_70: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_454, view_693);  permute_454 = None
        bmm_71: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_693, permute_455);  view_693 = permute_455 = None
        div_68: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_70, full_default_1);  bmm_70 = None
        permute_456: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_68, [0, 2, 1]);  div_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_694: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_68, [8, 24, 512, 64]);  bmm_68 = None
        permute_457: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_694, [0, 2, 1, 3]);  view_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_135: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
        view_695: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_135, [8, 512, 1536]);  clone_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_696: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_695, [4096, 1536]);  view_695 = None
        permute_202: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_299, [1, 0]);  primals_299 = None
        permute_458: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
        mm_70: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_696, permute_458);  permute_458 = None
        permute_459: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_696, [1, 0])
        mm_71: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_459, view_396);  permute_459 = None
        sum_122: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_696, [0], True);  view_696 = None
        view_697: "f32[1536]" = torch.ops.aten.reshape.default(sum_122, [1536]);  sum_122 = None
        view_698: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_70, [8, 512, 1536]);  mm_70 = None
        add_210: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_524, view_698);  mul_524 = view_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_699: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_456, [8, 24, 512, 64]);  permute_456 = None
        permute_462: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_699, [0, 2, 1, 3]);  view_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_700: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_462, [8, 512, 1536]);  permute_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_136: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_700, memory_format = torch.contiguous_format);  view_700 = None
        view_701: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_136, [4096, 1536]);  clone_136 = None
        permute_200: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_297, [1, 0]);  primals_297 = None
        permute_463: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
        mm_72: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_701, permute_463);  permute_463 = None
        permute_464: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_701, [1, 0])
        mm_73: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_464, view_396);  permute_464 = None
        sum_123: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_701, [0], True);  view_701 = None
        view_702: "f32[1536]" = torch.ops.aten.reshape.default(sum_123, [1536]);  sum_123 = None
        view_703: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_72, [8, 512, 1536]);  mm_72 = None
        add_211: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_210, view_703);  add_210 = view_703 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_704: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_71, [8, 24, 512, 64]);  bmm_71 = None
        permute_467: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_704, [0, 2, 1, 3]);  view_704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_137: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_467, memory_format = torch.contiguous_format);  permute_467 = None
        view_705: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_137, [8, 512, 1536]);  clone_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_706: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_705, [4096, 1536]);  view_705 = None
        permute_198: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_295, [1, 0]);  primals_295 = None
        permute_468: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
        mm_74: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_706, permute_468);  permute_468 = None
        permute_469: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_706, [1, 0])
        mm_75: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_469, view_396);  permute_469 = view_396 = None
        sum_124: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_706, [0], True);  view_706 = None
        view_707: "f32[1536]" = torch.ops.aten.reshape.default(sum_124, [1536]);  sum_124 = None
        view_708: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_74, [8, 512, 1536]);  mm_74 = None
        add_212: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_211, view_708);  add_211 = view_708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_532: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_212, primals_293);  primals_293 = None
        mul_533: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_532, 1536)
        sum_125: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_532, [2], True)
        mul_534: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_532, mul_256);  mul_532 = None
        sum_126: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_534, [2], True);  mul_534 = None
        mul_535: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_256, sum_126);  sum_126 = None
        sub_117: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_533, sum_125);  mul_533 = sum_125 = None
        sub_118: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_117, mul_535);  sub_117 = mul_535 = None
        mul_536: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_69, sub_118);  div_69 = sub_118 = None
        mul_537: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_212, mul_256);  mul_256 = None
        sum_127: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_537, [0, 1]);  mul_537 = None
        sum_128: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_212, [0, 1]);  add_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_43: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_54, torch.float32);  gt_54 = None
        mul_538: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.1111111111111112);  convert_element_type_43 = None
        mul_539: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_536, mul_538);  mul_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_709: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_539, [4096, 1536]);  mul_539 = None
        permute_197: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
        permute_472: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        mm_76: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_709, permute_472);  permute_472 = None
        permute_473: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_709, [1, 0])
        mm_77: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_473, view_394);  permute_473 = view_394 = None
        sum_129: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_709, [0], True);  view_709 = None
        view_710: "f32[1536]" = torch.ops.aten.reshape.default(sum_129, [1536]);  sum_129 = None
        view_711: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_76, [8, 512, 6144]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_393: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_106, [8, 512, 6144]);  addmm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_252: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476)
        erf_17: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_252);  mul_252 = None
        add_125: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_541: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_125, 0.5);  add_125 = None
        mul_542: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_393, view_393)
        mul_543: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_542, -0.5);  mul_542 = None
        exp_33: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_543);  mul_543 = None
        mul_544: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
        mul_545: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_393, mul_544);  view_393 = mul_544 = None
        add_214: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_541, mul_545);  mul_541 = mul_545 = None
        mul_546: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_711, add_214);  view_711 = add_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_712: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_546, [4096, 6144]);  mul_546 = None
        permute_196: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_289, [1, 0]);  primals_289 = None
        permute_476: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
        mm_78: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_712, permute_476);  permute_476 = None
        permute_477: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_712, [1, 0])
        mm_79: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_477, view_392);  permute_477 = view_392 = None
        sum_130: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_712, [0], True);  view_712 = None
        view_713: "f32[6144]" = torch.ops.aten.reshape.default(sum_130, [6144]);  sum_130 = None
        view_714: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_78, [8, 512, 1536]);  mm_78 = None
        add_215: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_536, view_714);  mul_536 = view_714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_548: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_215, primals_287);  primals_287 = None
        mul_549: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_548, 1536)
        sum_131: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_548, [2], True)
        mul_550: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_548, mul_249);  mul_548 = None
        sum_132: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_550, [2], True);  mul_550 = None
        mul_551: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_249, sum_132);  sum_132 = None
        sub_120: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_549, sum_131);  mul_549 = sum_131 = None
        sub_121: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_120, mul_551);  sub_120 = mul_551 = None
        mul_552: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_70, sub_121);  div_70 = sub_121 = None
        mul_553: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_215, mul_249);  mul_249 = None
        sum_133: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_553, [0, 1]);  mul_553 = None
        sum_134: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_215, [0, 1]);  add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_44: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_53, torch.float32);  gt_53 = None
        mul_554: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1.1111111111111112);  convert_element_type_44 = None
        mul_555: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_552, mul_554);  mul_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_715: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_555, [4096, 1536]);  mul_555 = None
        permute_195: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
        permute_480: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        mm_80: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_715, permute_480);  permute_480 = None
        permute_481: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_715, [1, 0])
        mm_81: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_481, view_390);  permute_481 = view_390 = None
        sum_135: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_715, [0], True);  view_715 = None
        view_716: "f32[1536]" = torch.ops.aten.reshape.default(sum_135, [1536]);  sum_135 = None
        view_717: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_80, [8, 512, 1536]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_718: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_717, [8, 512, 24, 64]);  view_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_484: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_718, [0, 2, 1, 3]);  view_718 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_140: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
        view_719: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_140, [192, 512, 64]);  clone_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_72: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_485, view_719);  permute_485 = None
        bmm_73: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_719, permute_486);  view_719 = permute_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_720: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_73, [8, 24, 512, 512]);  bmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_45: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_52, torch.float32);  gt_52 = None
        mul_556: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 1.1111111111111112);  convert_element_type_45 = None
        mul_557: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_720, mul_556);  view_720 = mul_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_558: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_557, div_35);  mul_557 = None
        sum_136: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_558, [-1], True)
        neg_7: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_35);  div_35 = None
        fma_6: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_7, sum_136, mul_558);  neg_7 = sum_136 = mul_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_34: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_6);  fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_721: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_34, [192, 512, 512]);  where_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_74: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_487, view_721);  permute_487 = None
        bmm_75: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_721, permute_488);  view_721 = permute_488 = None
        div_71: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_74, full_default_1);  bmm_74 = None
        permute_489: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_71, [0, 2, 1]);  div_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_722: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_72, [8, 24, 512, 64]);  bmm_72 = None
        permute_490: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_722, [0, 2, 1, 3]);  view_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_142: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
        view_723: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_142, [8, 512, 1536]);  clone_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_724: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_723, [4096, 1536]);  view_723 = None
        permute_191: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_283, [1, 0]);  primals_283 = None
        permute_491: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
        mm_82: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_724, permute_491);  permute_491 = None
        permute_492: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_724, [1, 0])
        mm_83: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_492, view_374);  permute_492 = None
        sum_137: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_724, [0], True);  view_724 = None
        view_725: "f32[1536]" = torch.ops.aten.reshape.default(sum_137, [1536]);  sum_137 = None
        view_726: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_82, [8, 512, 1536]);  mm_82 = None
        add_216: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_552, view_726);  mul_552 = view_726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_727: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_489, [8, 24, 512, 64]);  permute_489 = None
        permute_495: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_727, [0, 2, 1, 3]);  view_727 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_728: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_495, [8, 512, 1536]);  permute_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_143: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_728, memory_format = torch.contiguous_format);  view_728 = None
        view_729: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_143, [4096, 1536]);  clone_143 = None
        permute_189: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_281, [1, 0]);  primals_281 = None
        permute_496: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
        mm_84: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_729, permute_496);  permute_496 = None
        permute_497: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_729, [1, 0])
        mm_85: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_497, view_374);  permute_497 = None
        sum_138: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_729, [0], True);  view_729 = None
        view_730: "f32[1536]" = torch.ops.aten.reshape.default(sum_138, [1536]);  sum_138 = None
        view_731: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_84, [8, 512, 1536]);  mm_84 = None
        add_217: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_216, view_731);  add_216 = view_731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_732: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_75, [8, 24, 512, 64]);  bmm_75 = None
        permute_500: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_732, [0, 2, 1, 3]);  view_732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_144: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_500, memory_format = torch.contiguous_format);  permute_500 = None
        view_733: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_144, [8, 512, 1536]);  clone_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_734: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_733, [4096, 1536]);  view_733 = None
        permute_187: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_279, [1, 0]);  primals_279 = None
        permute_501: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
        mm_86: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_734, permute_501);  permute_501 = None
        permute_502: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_734, [1, 0])
        mm_87: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_502, view_374);  permute_502 = view_374 = None
        sum_139: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_734, [0], True);  view_734 = None
        view_735: "f32[1536]" = torch.ops.aten.reshape.default(sum_139, [1536]);  sum_139 = None
        view_736: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_86, [8, 512, 1536]);  mm_86 = None
        add_218: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_217, view_736);  add_217 = view_736 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_560: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_218, primals_277);  primals_277 = None
        mul_561: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_560, 1536)
        sum_140: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_560, [2], True)
        mul_562: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_560, mul_242);  mul_560 = None
        sum_141: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_562, [2], True);  mul_562 = None
        mul_563: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_242, sum_141);  sum_141 = None
        sub_123: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_561, sum_140);  mul_561 = sum_140 = None
        sub_124: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_123, mul_563);  sub_123 = mul_563 = None
        mul_564: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_72, sub_124);  div_72 = sub_124 = None
        mul_565: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_218, mul_242);  mul_242 = None
        sum_142: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_565, [0, 1]);  mul_565 = None
        sum_143: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_218, [0, 1]);  add_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_46: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_51, torch.float32);  gt_51 = None
        mul_566: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 1.1111111111111112);  convert_element_type_46 = None
        mul_567: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_564, mul_566);  mul_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_737: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_567, [4096, 1536]);  mul_567 = None
        permute_186: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_275, [1, 0]);  primals_275 = None
        permute_505: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        mm_88: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_737, permute_505);  permute_505 = None
        permute_506: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_737, [1, 0])
        mm_89: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_506, view_372);  permute_506 = view_372 = None
        sum_144: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_737, [0], True);  view_737 = None
        view_738: "f32[1536]" = torch.ops.aten.reshape.default(sum_144, [1536]);  sum_144 = None
        view_739: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_88, [8, 512, 6144]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_371: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_100, [8, 512, 6144]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_238: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476)
        erf_16: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_238);  mul_238 = None
        add_118: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_569: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_118, 0.5);  add_118 = None
        mul_570: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_371, view_371)
        mul_571: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_570, -0.5);  mul_570 = None
        exp_34: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_571);  mul_571 = None
        mul_572: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
        mul_573: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_371, mul_572);  view_371 = mul_572 = None
        add_220: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_569, mul_573);  mul_569 = mul_573 = None
        mul_574: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_739, add_220);  view_739 = add_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_740: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_574, [4096, 6144]);  mul_574 = None
        permute_185: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_273, [1, 0]);  primals_273 = None
        permute_509: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        mm_90: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_740, permute_509);  permute_509 = None
        permute_510: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_740, [1, 0])
        mm_91: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_510, view_370);  permute_510 = view_370 = None
        sum_145: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_740, [0], True);  view_740 = None
        view_741: "f32[6144]" = torch.ops.aten.reshape.default(sum_145, [6144]);  sum_145 = None
        view_742: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_90, [8, 512, 1536]);  mm_90 = None
        add_221: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_564, view_742);  mul_564 = view_742 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_576: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_221, primals_271);  primals_271 = None
        mul_577: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_576, 1536)
        sum_146: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_576, [2], True)
        mul_578: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_576, mul_235);  mul_576 = None
        sum_147: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_578, [2], True);  mul_578 = None
        mul_579: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_235, sum_147);  sum_147 = None
        sub_126: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_577, sum_146);  mul_577 = sum_146 = None
        sub_127: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_126, mul_579);  sub_126 = mul_579 = None
        mul_580: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_73, sub_127);  div_73 = sub_127 = None
        mul_581: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_221, mul_235);  mul_235 = None
        sum_148: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_581, [0, 1]);  mul_581 = None
        sum_149: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_221, [0, 1]);  add_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_47: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_50, torch.float32);  gt_50 = None
        mul_582: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
        mul_583: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_580, mul_582);  mul_582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_743: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_583, [4096, 1536]);  mul_583 = None
        permute_184: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_269, [1, 0]);  primals_269 = None
        permute_513: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
        mm_92: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_743, permute_513);  permute_513 = None
        permute_514: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_743, [1, 0])
        mm_93: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_514, view_368);  permute_514 = view_368 = None
        sum_150: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_743, [0], True);  view_743 = None
        view_744: "f32[1536]" = torch.ops.aten.reshape.default(sum_150, [1536]);  sum_150 = None
        view_745: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_92, [8, 512, 1536]);  mm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_746: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_745, [8, 512, 24, 64]);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_517: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_746, [0, 2, 1, 3]);  view_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_147: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format);  permute_517 = None
        view_747: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_147, [192, 512, 64]);  clone_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_76: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_518, view_747);  permute_518 = None
        bmm_77: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_747, permute_519);  view_747 = permute_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_748: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_77, [8, 24, 512, 512]);  bmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_48: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_49, torch.float32);  gt_49 = None
        mul_584: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 1.1111111111111112);  convert_element_type_48 = None
        mul_585: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_748, mul_584);  view_748 = mul_584 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_586: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_585, div_33);  mul_585 = None
        sum_151: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_586, [-1], True)
        neg_8: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_33);  div_33 = None
        fma_7: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_8, sum_151, mul_586);  neg_8 = sum_151 = mul_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_35: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_7);  fma_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_749: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_35, [192, 512, 512]);  where_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_78: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_520, view_749);  permute_520 = None
        bmm_79: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_749, permute_521);  view_749 = permute_521 = None
        div_74: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_78, full_default_1);  bmm_78 = None
        permute_522: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_74, [0, 2, 1]);  div_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_750: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_76, [8, 24, 512, 64]);  bmm_76 = None
        permute_523: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_750, [0, 2, 1, 3]);  view_750 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_149: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
        view_751: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_149, [8, 512, 1536]);  clone_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_752: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_751, [4096, 1536]);  view_751 = None
        permute_180: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_267, [1, 0]);  primals_267 = None
        permute_524: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
        mm_94: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_752, permute_524);  permute_524 = None
        permute_525: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_752, [1, 0])
        mm_95: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_525, view_352);  permute_525 = None
        sum_152: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_752, [0], True);  view_752 = None
        view_753: "f32[1536]" = torch.ops.aten.reshape.default(sum_152, [1536]);  sum_152 = None
        view_754: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_94, [8, 512, 1536]);  mm_94 = None
        add_222: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_580, view_754);  mul_580 = view_754 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_755: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_522, [8, 24, 512, 64]);  permute_522 = None
        permute_528: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_755, [0, 2, 1, 3]);  view_755 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_756: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_528, [8, 512, 1536]);  permute_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_150: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_756, memory_format = torch.contiguous_format);  view_756 = None
        view_757: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_150, [4096, 1536]);  clone_150 = None
        permute_178: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
        permute_529: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        mm_96: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_757, permute_529);  permute_529 = None
        permute_530: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_757, [1, 0])
        mm_97: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_530, view_352);  permute_530 = None
        sum_153: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_757, [0], True);  view_757 = None
        view_758: "f32[1536]" = torch.ops.aten.reshape.default(sum_153, [1536]);  sum_153 = None
        view_759: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_96, [8, 512, 1536]);  mm_96 = None
        add_223: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_222, view_759);  add_222 = view_759 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_760: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_79, [8, 24, 512, 64]);  bmm_79 = None
        permute_533: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_760, [0, 2, 1, 3]);  view_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_151: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_533, memory_format = torch.contiguous_format);  permute_533 = None
        view_761: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_151, [8, 512, 1536]);  clone_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_762: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_761, [4096, 1536]);  view_761 = None
        permute_176: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_263, [1, 0]);  primals_263 = None
        permute_534: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        mm_98: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_762, permute_534);  permute_534 = None
        permute_535: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_762, [1, 0])
        mm_99: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_535, view_352);  permute_535 = view_352 = None
        sum_154: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_762, [0], True);  view_762 = None
        view_763: "f32[1536]" = torch.ops.aten.reshape.default(sum_154, [1536]);  sum_154 = None
        view_764: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_98, [8, 512, 1536]);  mm_98 = None
        add_224: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_223, view_764);  add_223 = view_764 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_588: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_224, primals_261);  primals_261 = None
        mul_589: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_588, 1536)
        sum_155: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_588, [2], True)
        mul_590: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_588, mul_228);  mul_588 = None
        sum_156: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_590, [2], True);  mul_590 = None
        mul_591: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_228, sum_156);  sum_156 = None
        sub_129: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_589, sum_155);  mul_589 = sum_155 = None
        sub_130: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_129, mul_591);  sub_129 = mul_591 = None
        mul_592: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_75, sub_130);  div_75 = sub_130 = None
        mul_593: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_224, mul_228);  mul_228 = None
        sum_157: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_593, [0, 1]);  mul_593 = None
        sum_158: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_224, [0, 1]);  add_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_49: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_48, torch.float32);  gt_48 = None
        mul_594: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 1.1111111111111112);  convert_element_type_49 = None
        mul_595: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_592, mul_594);  mul_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_765: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_595, [4096, 1536]);  mul_595 = None
        permute_175: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
        permute_538: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
        mm_100: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_765, permute_538);  permute_538 = None
        permute_539: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_765, [1, 0])
        mm_101: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_539, view_350);  permute_539 = view_350 = None
        sum_159: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_765, [0], True);  view_765 = None
        view_766: "f32[1536]" = torch.ops.aten.reshape.default(sum_159, [1536]);  sum_159 = None
        view_767: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_100, [8, 512, 6144]);  mm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_349: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_94, [8, 512, 6144]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_224: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_349, 0.7071067811865476)
        erf_15: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_224);  mul_224 = None
        add_111: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_597: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_111, 0.5);  add_111 = None
        mul_598: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_349, view_349)
        mul_599: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_598, -0.5);  mul_598 = None
        exp_35: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_599);  mul_599 = None
        mul_600: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
        mul_601: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_349, mul_600);  view_349 = mul_600 = None
        add_226: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_597, mul_601);  mul_597 = mul_601 = None
        mul_602: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_767, add_226);  view_767 = add_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_768: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_602, [4096, 6144]);  mul_602 = None
        permute_174: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_257, [1, 0]);  primals_257 = None
        permute_542: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        mm_102: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_768, permute_542);  permute_542 = None
        permute_543: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_768, [1, 0])
        mm_103: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_543, view_348);  permute_543 = view_348 = None
        sum_160: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_768, [0], True);  view_768 = None
        view_769: "f32[6144]" = torch.ops.aten.reshape.default(sum_160, [6144]);  sum_160 = None
        view_770: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_102, [8, 512, 1536]);  mm_102 = None
        add_227: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_592, view_770);  mul_592 = view_770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_604: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_227, primals_255);  primals_255 = None
        mul_605: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_604, 1536)
        sum_161: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_604, [2], True)
        mul_606: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_604, mul_221);  mul_604 = None
        sum_162: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_606, [2], True);  mul_606 = None
        mul_607: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_221, sum_162);  sum_162 = None
        sub_132: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_605, sum_161);  mul_605 = sum_161 = None
        sub_133: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_132, mul_607);  sub_132 = mul_607 = None
        mul_608: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_76, sub_133);  div_76 = sub_133 = None
        mul_609: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_227, mul_221);  mul_221 = None
        sum_163: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_609, [0, 1]);  mul_609 = None
        sum_164: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_227, [0, 1]);  add_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_50: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_47, torch.float32);  gt_47 = None
        mul_610: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 1.1111111111111112);  convert_element_type_50 = None
        mul_611: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_608, mul_610);  mul_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_771: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_611, [4096, 1536]);  mul_611 = None
        permute_173: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_253, [1, 0]);  primals_253 = None
        permute_546: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        mm_104: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_771, permute_546);  permute_546 = None
        permute_547: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_771, [1, 0])
        mm_105: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_547, view_346);  permute_547 = view_346 = None
        sum_165: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_771, [0], True);  view_771 = None
        view_772: "f32[1536]" = torch.ops.aten.reshape.default(sum_165, [1536]);  sum_165 = None
        view_773: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_104, [8, 512, 1536]);  mm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_774: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_773, [8, 512, 24, 64]);  view_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_550: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_154: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_550, memory_format = torch.contiguous_format);  permute_550 = None
        view_775: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_154, [192, 512, 64]);  clone_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_80: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_551, view_775);  permute_551 = None
        bmm_81: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_775, permute_552);  view_775 = permute_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_776: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_81, [8, 24, 512, 512]);  bmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_51: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_46, torch.float32);  gt_46 = None
        mul_612: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_51, 1.1111111111111112);  convert_element_type_51 = None
        mul_613: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_776, mul_612);  view_776 = mul_612 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_614: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_613, div_31);  mul_613 = None
        sum_166: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_614, [-1], True)
        neg_9: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_31);  div_31 = None
        fma_8: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_9, sum_166, mul_614);  neg_9 = sum_166 = mul_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_36: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_8);  fma_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_777: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_36, [192, 512, 512]);  where_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_82: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_553, view_777);  permute_553 = None
        bmm_83: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_777, permute_554);  view_777 = permute_554 = None
        div_77: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_82, full_default_1);  bmm_82 = None
        permute_555: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_77, [0, 2, 1]);  div_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_778: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_80, [8, 24, 512, 64]);  bmm_80 = None
        permute_556: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_778, [0, 2, 1, 3]);  view_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_156: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_556, memory_format = torch.contiguous_format);  permute_556 = None
        view_779: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_156, [8, 512, 1536]);  clone_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_780: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_779, [4096, 1536]);  view_779 = None
        permute_169: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_251, [1, 0]);  primals_251 = None
        permute_557: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
        mm_106: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_780, permute_557);  permute_557 = None
        permute_558: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_780, [1, 0])
        mm_107: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_558, view_330);  permute_558 = None
        sum_167: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_780, [0], True);  view_780 = None
        view_781: "f32[1536]" = torch.ops.aten.reshape.default(sum_167, [1536]);  sum_167 = None
        view_782: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_106, [8, 512, 1536]);  mm_106 = None
        add_228: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_608, view_782);  mul_608 = view_782 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_783: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_555, [8, 24, 512, 64]);  permute_555 = None
        permute_561: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_783, [0, 2, 1, 3]);  view_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_784: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_561, [8, 512, 1536]);  permute_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_157: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_784, memory_format = torch.contiguous_format);  view_784 = None
        view_785: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_157, [4096, 1536]);  clone_157 = None
        permute_167: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
        permute_562: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        mm_108: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_785, permute_562);  permute_562 = None
        permute_563: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_785, [1, 0])
        mm_109: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_563, view_330);  permute_563 = None
        sum_168: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_785, [0], True);  view_785 = None
        view_786: "f32[1536]" = torch.ops.aten.reshape.default(sum_168, [1536]);  sum_168 = None
        view_787: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_108, [8, 512, 1536]);  mm_108 = None
        add_229: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_228, view_787);  add_228 = view_787 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_788: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_83, [8, 24, 512, 64]);  bmm_83 = None
        permute_566: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_788, [0, 2, 1, 3]);  view_788 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_158: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_566, memory_format = torch.contiguous_format);  permute_566 = None
        view_789: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_158, [8, 512, 1536]);  clone_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_790: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_789, [4096, 1536]);  view_789 = None
        permute_165: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_247, [1, 0]);  primals_247 = None
        permute_567: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        mm_110: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_790, permute_567);  permute_567 = None
        permute_568: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_790, [1, 0])
        mm_111: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_568, view_330);  permute_568 = view_330 = None
        sum_169: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_790, [0], True);  view_790 = None
        view_791: "f32[1536]" = torch.ops.aten.reshape.default(sum_169, [1536]);  sum_169 = None
        view_792: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_110, [8, 512, 1536]);  mm_110 = None
        add_230: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_229, view_792);  add_229 = view_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_616: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_230, primals_245);  primals_245 = None
        mul_617: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_616, 1536)
        sum_170: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_616, [2], True)
        mul_618: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_616, mul_214);  mul_616 = None
        sum_171: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_618, [2], True);  mul_618 = None
        mul_619: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_214, sum_171);  sum_171 = None
        sub_135: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_617, sum_170);  mul_617 = sum_170 = None
        sub_136: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_135, mul_619);  sub_135 = mul_619 = None
        mul_620: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_78, sub_136);  div_78 = sub_136 = None
        mul_621: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_230, mul_214);  mul_214 = None
        sum_172: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_621, [0, 1]);  mul_621 = None
        sum_173: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_52: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_45, torch.float32);  gt_45 = None
        mul_622: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
        mul_623: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_620, mul_622);  mul_622 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_793: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_623, [4096, 1536]);  mul_623 = None
        permute_164: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_243, [1, 0]);  primals_243 = None
        permute_571: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
        mm_112: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_793, permute_571);  permute_571 = None
        permute_572: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_793, [1, 0])
        mm_113: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_572, view_328);  permute_572 = view_328 = None
        sum_174: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_793, [0], True);  view_793 = None
        view_794: "f32[1536]" = torch.ops.aten.reshape.default(sum_174, [1536]);  sum_174 = None
        view_795: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_112, [8, 512, 6144]);  mm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_327: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_88, [8, 512, 6144]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_210: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476)
        erf_14: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_210);  mul_210 = None
        add_104: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_625: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_104, 0.5);  add_104 = None
        mul_626: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_327, view_327)
        mul_627: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_626, -0.5);  mul_626 = None
        exp_36: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_627);  mul_627 = None
        mul_628: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
        mul_629: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_327, mul_628);  view_327 = mul_628 = None
        add_232: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_625, mul_629);  mul_625 = mul_629 = None
        mul_630: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_795, add_232);  view_795 = add_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_796: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_630, [4096, 6144]);  mul_630 = None
        permute_163: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
        permute_575: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        mm_114: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_796, permute_575);  permute_575 = None
        permute_576: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_796, [1, 0])
        mm_115: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_576, view_326);  permute_576 = view_326 = None
        sum_175: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True);  view_796 = None
        view_797: "f32[6144]" = torch.ops.aten.reshape.default(sum_175, [6144]);  sum_175 = None
        view_798: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_114, [8, 512, 1536]);  mm_114 = None
        add_233: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_620, view_798);  mul_620 = view_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_632: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_233, primals_239);  primals_239 = None
        mul_633: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_632, 1536)
        sum_176: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_632, [2], True)
        mul_634: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_632, mul_207);  mul_632 = None
        sum_177: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_634, [2], True);  mul_634 = None
        mul_635: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_207, sum_177);  sum_177 = None
        sub_138: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_633, sum_176);  mul_633 = sum_176 = None
        sub_139: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_138, mul_635);  sub_138 = mul_635 = None
        mul_636: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_79, sub_139);  div_79 = sub_139 = None
        mul_637: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_233, mul_207);  mul_207 = None
        sum_178: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_637, [0, 1]);  mul_637 = None
        sum_179: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_53: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_44, torch.float32);  gt_44 = None
        mul_638: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.1111111111111112);  convert_element_type_53 = None
        mul_639: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_636, mul_638);  mul_638 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_799: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_639, [4096, 1536]);  mul_639 = None
        permute_162: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
        permute_579: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
        mm_116: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_799, permute_579);  permute_579 = None
        permute_580: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_799, [1, 0])
        mm_117: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_580, view_324);  permute_580 = view_324 = None
        sum_180: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_799, [0], True);  view_799 = None
        view_800: "f32[1536]" = torch.ops.aten.reshape.default(sum_180, [1536]);  sum_180 = None
        view_801: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_116, [8, 512, 1536]);  mm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_802: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_801, [8, 512, 24, 64]);  view_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_583: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_802, [0, 2, 1, 3]);  view_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_161: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_583, memory_format = torch.contiguous_format);  permute_583 = None
        view_803: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_161, [192, 512, 64]);  clone_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_84: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_584, view_803);  permute_584 = None
        bmm_85: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_803, permute_585);  view_803 = permute_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_804: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_85, [8, 24, 512, 512]);  bmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_54: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_43, torch.float32);  gt_43 = None
        mul_640: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1.1111111111111112);  convert_element_type_54 = None
        mul_641: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_804, mul_640);  view_804 = mul_640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_642: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_641, div_29);  mul_641 = None
        sum_181: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_642, [-1], True)
        neg_10: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_29);  div_29 = None
        fma_9: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_10, sum_181, mul_642);  neg_10 = sum_181 = mul_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_37: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_9);  fma_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_805: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_37, [192, 512, 512]);  where_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_86: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_586, view_805);  permute_586 = None
        bmm_87: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_805, permute_587);  view_805 = permute_587 = None
        div_80: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_86, full_default_1);  bmm_86 = None
        permute_588: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_80, [0, 2, 1]);  div_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_806: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_84, [8, 24, 512, 64]);  bmm_84 = None
        permute_589: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_806, [0, 2, 1, 3]);  view_806 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_163: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_589, memory_format = torch.contiguous_format);  permute_589 = None
        view_807: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_163, [8, 512, 1536]);  clone_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_808: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_807, [4096, 1536]);  view_807 = None
        permute_158: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
        permute_590: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
        mm_118: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_808, permute_590);  permute_590 = None
        permute_591: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_808, [1, 0])
        mm_119: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_591, view_308);  permute_591 = None
        sum_182: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_808, [0], True);  view_808 = None
        view_809: "f32[1536]" = torch.ops.aten.reshape.default(sum_182, [1536]);  sum_182 = None
        view_810: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_118, [8, 512, 1536]);  mm_118 = None
        add_234: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_636, view_810);  mul_636 = view_810 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_811: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_588, [8, 24, 512, 64]);  permute_588 = None
        permute_594: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_811, [0, 2, 1, 3]);  view_811 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_812: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_594, [8, 512, 1536]);  permute_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_164: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_812, memory_format = torch.contiguous_format);  view_812 = None
        view_813: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_164, [4096, 1536]);  clone_164 = None
        permute_156: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_233, [1, 0]);  primals_233 = None
        permute_595: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        mm_120: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_813, permute_595);  permute_595 = None
        permute_596: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_813, [1, 0])
        mm_121: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_596, view_308);  permute_596 = None
        sum_183: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_813, [0], True);  view_813 = None
        view_814: "f32[1536]" = torch.ops.aten.reshape.default(sum_183, [1536]);  sum_183 = None
        view_815: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_120, [8, 512, 1536]);  mm_120 = None
        add_235: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_234, view_815);  add_234 = view_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_816: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_87, [8, 24, 512, 64]);  bmm_87 = None
        permute_599: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_816, [0, 2, 1, 3]);  view_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_165: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_599, memory_format = torch.contiguous_format);  permute_599 = None
        view_817: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_165, [8, 512, 1536]);  clone_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_818: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_817, [4096, 1536]);  view_817 = None
        permute_154: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
        permute_600: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        mm_122: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_818, permute_600);  permute_600 = None
        permute_601: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_818, [1, 0])
        mm_123: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_601, view_308);  permute_601 = view_308 = None
        sum_184: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_818, [0], True);  view_818 = None
        view_819: "f32[1536]" = torch.ops.aten.reshape.default(sum_184, [1536]);  sum_184 = None
        view_820: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_122, [8, 512, 1536]);  mm_122 = None
        add_236: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_235, view_820);  add_235 = view_820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_644: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_236, primals_229);  primals_229 = None
        mul_645: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_644, 1536)
        sum_185: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_644, [2], True)
        mul_646: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_644, mul_200);  mul_644 = None
        sum_186: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_646, [2], True);  mul_646 = None
        mul_647: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_200, sum_186);  sum_186 = None
        sub_141: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_645, sum_185);  mul_645 = sum_185 = None
        sub_142: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_141, mul_647);  sub_141 = mul_647 = None
        mul_648: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_81, sub_142);  div_81 = sub_142 = None
        mul_649: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_236, mul_200);  mul_200 = None
        sum_187: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_649, [0, 1]);  mul_649 = None
        sum_188: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_236, [0, 1]);  add_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_55: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_42, torch.float32);  gt_42 = None
        mul_650: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.1111111111111112);  convert_element_type_55 = None
        mul_651: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_648, mul_650);  mul_650 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_821: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_651, [4096, 1536]);  mul_651 = None
        permute_153: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
        permute_604: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        mm_124: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_821, permute_604);  permute_604 = None
        permute_605: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_821, [1, 0])
        mm_125: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_605, view_306);  permute_605 = view_306 = None
        sum_189: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_821, [0], True);  view_821 = None
        view_822: "f32[1536]" = torch.ops.aten.reshape.default(sum_189, [1536]);  sum_189 = None
        view_823: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_124, [8, 512, 6144]);  mm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_305: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_82, [8, 512, 6144]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_196: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_305, 0.7071067811865476)
        erf_13: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_196);  mul_196 = None
        add_97: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_653: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_97, 0.5);  add_97 = None
        mul_654: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_305, view_305)
        mul_655: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_654, -0.5);  mul_654 = None
        exp_37: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_655);  mul_655 = None
        mul_656: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
        mul_657: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_305, mul_656);  view_305 = mul_656 = None
        add_238: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_653, mul_657);  mul_653 = mul_657 = None
        mul_658: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_823, add_238);  view_823 = add_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_824: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_658, [4096, 6144]);  mul_658 = None
        permute_152: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
        permute_608: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        mm_126: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_824, permute_608);  permute_608 = None
        permute_609: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_824, [1, 0])
        mm_127: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_609, view_304);  permute_609 = view_304 = None
        sum_190: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_824, [0], True);  view_824 = None
        view_825: "f32[6144]" = torch.ops.aten.reshape.default(sum_190, [6144]);  sum_190 = None
        view_826: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_126, [8, 512, 1536]);  mm_126 = None
        add_239: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_648, view_826);  mul_648 = view_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_660: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_239, primals_223);  primals_223 = None
        mul_661: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_660, 1536)
        sum_191: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_660, [2], True)
        mul_662: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_660, mul_193);  mul_660 = None
        sum_192: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_662, [2], True);  mul_662 = None
        mul_663: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_193, sum_192);  sum_192 = None
        sub_144: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_661, sum_191);  mul_661 = sum_191 = None
        sub_145: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_144, mul_663);  sub_144 = mul_663 = None
        mul_664: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_82, sub_145);  div_82 = sub_145 = None
        mul_665: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_239, mul_193);  mul_193 = None
        sum_193: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_665, [0, 1]);  mul_665 = None
        sum_194: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_239, [0, 1]);  add_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_56: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_41, torch.float32);  gt_41 = None
        mul_666: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 1.1111111111111112);  convert_element_type_56 = None
        mul_667: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_664, mul_666);  mul_666 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_827: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_667, [4096, 1536]);  mul_667 = None
        permute_151: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
        permute_612: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        mm_128: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_827, permute_612);  permute_612 = None
        permute_613: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_827, [1, 0])
        mm_129: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_613, view_302);  permute_613 = view_302 = None
        sum_195: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_827, [0], True);  view_827 = None
        view_828: "f32[1536]" = torch.ops.aten.reshape.default(sum_195, [1536]);  sum_195 = None
        view_829: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_128, [8, 512, 1536]);  mm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_830: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_829, [8, 512, 24, 64]);  view_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_616: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_830, [0, 2, 1, 3]);  view_830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_168: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_616, memory_format = torch.contiguous_format);  permute_616 = None
        view_831: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_168, [192, 512, 64]);  clone_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_88: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_617, view_831);  permute_617 = None
        bmm_89: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_831, permute_618);  view_831 = permute_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_832: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_89, [8, 24, 512, 512]);  bmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_57: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_40, torch.float32);  gt_40 = None
        mul_668: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_57, 1.1111111111111112);  convert_element_type_57 = None
        mul_669: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_832, mul_668);  view_832 = mul_668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_670: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_669, div_27);  mul_669 = None
        sum_196: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_670, [-1], True)
        neg_11: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_27);  div_27 = None
        fma_10: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_11, sum_196, mul_670);  neg_11 = sum_196 = mul_670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_38: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_10);  fma_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_833: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_38, [192, 512, 512]);  where_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_90: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_619, view_833);  permute_619 = None
        bmm_91: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_833, permute_620);  view_833 = permute_620 = None
        div_83: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_90, full_default_1);  bmm_90 = None
        permute_621: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_83, [0, 2, 1]);  div_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_834: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_88, [8, 24, 512, 64]);  bmm_88 = None
        permute_622: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_834, [0, 2, 1, 3]);  view_834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_170: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_622, memory_format = torch.contiguous_format);  permute_622 = None
        view_835: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_170, [8, 512, 1536]);  clone_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_836: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_835, [4096, 1536]);  view_835 = None
        permute_147: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
        permute_623: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
        mm_130: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_836, permute_623);  permute_623 = None
        permute_624: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_836, [1, 0])
        mm_131: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_624, view_286);  permute_624 = None
        sum_197: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_836, [0], True);  view_836 = None
        view_837: "f32[1536]" = torch.ops.aten.reshape.default(sum_197, [1536]);  sum_197 = None
        view_838: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_130, [8, 512, 1536]);  mm_130 = None
        add_240: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_664, view_838);  mul_664 = view_838 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_839: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_621, [8, 24, 512, 64]);  permute_621 = None
        permute_627: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_839, [0, 2, 1, 3]);  view_839 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_840: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_627, [8, 512, 1536]);  permute_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_171: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_840, memory_format = torch.contiguous_format);  view_840 = None
        view_841: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_171, [4096, 1536]);  clone_171 = None
        permute_145: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_217, [1, 0]);  primals_217 = None
        permute_628: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        mm_132: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_841, permute_628);  permute_628 = None
        permute_629: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_841, [1, 0])
        mm_133: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_629, view_286);  permute_629 = None
        sum_198: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_841, [0], True);  view_841 = None
        view_842: "f32[1536]" = torch.ops.aten.reshape.default(sum_198, [1536]);  sum_198 = None
        view_843: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_132, [8, 512, 1536]);  mm_132 = None
        add_241: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_240, view_843);  add_240 = view_843 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_844: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_91, [8, 24, 512, 64]);  bmm_91 = None
        permute_632: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_844, [0, 2, 1, 3]);  view_844 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_172: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_632, memory_format = torch.contiguous_format);  permute_632 = None
        view_845: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_172, [8, 512, 1536]);  clone_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_846: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_845, [4096, 1536]);  view_845 = None
        permute_143: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
        permute_633: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        mm_134: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_846, permute_633);  permute_633 = None
        permute_634: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_846, [1, 0])
        mm_135: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_634, view_286);  permute_634 = view_286 = None
        sum_199: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_846, [0], True);  view_846 = None
        view_847: "f32[1536]" = torch.ops.aten.reshape.default(sum_199, [1536]);  sum_199 = None
        view_848: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_134, [8, 512, 1536]);  mm_134 = None
        add_242: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_241, view_848);  add_241 = view_848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_672: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_242, primals_213);  primals_213 = None
        mul_673: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_672, 1536)
        sum_200: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_672, [2], True)
        mul_674: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_672, mul_186);  mul_672 = None
        sum_201: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_674, [2], True);  mul_674 = None
        mul_675: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_186, sum_201);  sum_201 = None
        sub_147: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_673, sum_200);  mul_673 = sum_200 = None
        sub_148: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_147, mul_675);  sub_147 = mul_675 = None
        mul_676: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_84, sub_148);  div_84 = sub_148 = None
        mul_677: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_242, mul_186);  mul_186 = None
        sum_202: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_677, [0, 1]);  mul_677 = None
        sum_203: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_242, [0, 1]);  add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_58: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_39, torch.float32);  gt_39 = None
        mul_678: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 1.1111111111111112);  convert_element_type_58 = None
        mul_679: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_676, mul_678);  mul_678 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_849: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_679, [4096, 1536]);  mul_679 = None
        permute_142: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
        permute_637: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        mm_136: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_849, permute_637);  permute_637 = None
        permute_638: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_849, [1, 0])
        mm_137: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_638, view_284);  permute_638 = view_284 = None
        sum_204: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_849, [0], True);  view_849 = None
        view_850: "f32[1536]" = torch.ops.aten.reshape.default(sum_204, [1536]);  sum_204 = None
        view_851: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_136, [8, 512, 6144]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_283: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_76, [8, 512, 6144]);  addmm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_182: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_283, 0.7071067811865476)
        erf_12: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_182);  mul_182 = None
        add_90: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_681: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_90, 0.5);  add_90 = None
        mul_682: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_283, view_283)
        mul_683: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_682, -0.5);  mul_682 = None
        exp_38: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_683);  mul_683 = None
        mul_684: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
        mul_685: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_283, mul_684);  view_283 = mul_684 = None
        add_244: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_681, mul_685);  mul_681 = mul_685 = None
        mul_686: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_851, add_244);  view_851 = add_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_852: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_686, [4096, 6144]);  mul_686 = None
        permute_141: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
        permute_641: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        mm_138: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_852, permute_641);  permute_641 = None
        permute_642: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_852, [1, 0])
        mm_139: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_642, view_282);  permute_642 = view_282 = None
        sum_205: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_852, [0], True);  view_852 = None
        view_853: "f32[6144]" = torch.ops.aten.reshape.default(sum_205, [6144]);  sum_205 = None
        view_854: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_138, [8, 512, 1536]);  mm_138 = None
        add_245: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_676, view_854);  mul_676 = view_854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_688: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_245, primals_207);  primals_207 = None
        mul_689: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_688, 1536)
        sum_206: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_688, [2], True)
        mul_690: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_688, mul_179);  mul_688 = None
        sum_207: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_690, [2], True);  mul_690 = None
        mul_691: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_179, sum_207);  sum_207 = None
        sub_150: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_689, sum_206);  mul_689 = sum_206 = None
        sub_151: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_150, mul_691);  sub_150 = mul_691 = None
        mul_692: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_85, sub_151);  div_85 = sub_151 = None
        mul_693: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_245, mul_179);  mul_179 = None
        sum_208: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_693, [0, 1]);  mul_693 = None
        sum_209: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_245, [0, 1]);  add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_59: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_38, torch.float32);  gt_38 = None
        mul_694: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 1.1111111111111112);  convert_element_type_59 = None
        mul_695: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_692, mul_694);  mul_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_855: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_695, [4096, 1536]);  mul_695 = None
        permute_140: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
        permute_645: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        mm_140: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_855, permute_645);  permute_645 = None
        permute_646: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_855, [1, 0])
        mm_141: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_646, view_280);  permute_646 = view_280 = None
        sum_210: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_855, [0], True);  view_855 = None
        view_856: "f32[1536]" = torch.ops.aten.reshape.default(sum_210, [1536]);  sum_210 = None
        view_857: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_140, [8, 512, 1536]);  mm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_858: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_857, [8, 512, 24, 64]);  view_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_649: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_858, [0, 2, 1, 3]);  view_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_175: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_649, memory_format = torch.contiguous_format);  permute_649 = None
        view_859: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_175, [192, 512, 64]);  clone_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_92: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_650, view_859);  permute_650 = None
        bmm_93: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_859, permute_651);  view_859 = permute_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_860: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_93, [8, 24, 512, 512]);  bmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_60: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_37, torch.float32);  gt_37 = None
        mul_696: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 1.1111111111111112);  convert_element_type_60 = None
        mul_697: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_860, mul_696);  view_860 = mul_696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_698: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_697, div_25);  mul_697 = None
        sum_211: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_698, [-1], True)
        neg_12: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_25);  div_25 = None
        fma_11: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_12, sum_211, mul_698);  neg_12 = sum_211 = mul_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_39: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_11);  fma_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_861: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_39, [192, 512, 512]);  where_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_94: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_652, view_861);  permute_652 = None
        bmm_95: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_861, permute_653);  view_861 = permute_653 = None
        div_86: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_94, full_default_1);  bmm_94 = None
        permute_654: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_86, [0, 2, 1]);  div_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_862: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_92, [8, 24, 512, 64]);  bmm_92 = None
        permute_655: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_862, [0, 2, 1, 3]);  view_862 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_177: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_655, memory_format = torch.contiguous_format);  permute_655 = None
        view_863: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_177, [8, 512, 1536]);  clone_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_864: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_863, [4096, 1536]);  view_863 = None
        permute_136: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
        permute_656: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        mm_142: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_864, permute_656);  permute_656 = None
        permute_657: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_864, [1, 0])
        mm_143: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_657, view_264);  permute_657 = None
        sum_212: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_864, [0], True);  view_864 = None
        view_865: "f32[1536]" = torch.ops.aten.reshape.default(sum_212, [1536]);  sum_212 = None
        view_866: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_142, [8, 512, 1536]);  mm_142 = None
        add_246: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_692, view_866);  mul_692 = view_866 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_867: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_654, [8, 24, 512, 64]);  permute_654 = None
        permute_660: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_867, [0, 2, 1, 3]);  view_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_868: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_660, [8, 512, 1536]);  permute_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_178: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_868, memory_format = torch.contiguous_format);  view_868 = None
        view_869: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_178, [4096, 1536]);  clone_178 = None
        permute_134: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
        permute_661: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        mm_144: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_869, permute_661);  permute_661 = None
        permute_662: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_869, [1, 0])
        mm_145: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_662, view_264);  permute_662 = None
        sum_213: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_869, [0], True);  view_869 = None
        view_870: "f32[1536]" = torch.ops.aten.reshape.default(sum_213, [1536]);  sum_213 = None
        view_871: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_144, [8, 512, 1536]);  mm_144 = None
        add_247: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_246, view_871);  add_246 = view_871 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_872: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_95, [8, 24, 512, 64]);  bmm_95 = None
        permute_665: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_872, [0, 2, 1, 3]);  view_872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_179: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_665, memory_format = torch.contiguous_format);  permute_665 = None
        view_873: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_179, [8, 512, 1536]);  clone_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_874: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_873, [4096, 1536]);  view_873 = None
        permute_132: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
        permute_666: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        mm_146: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_874, permute_666);  permute_666 = None
        permute_667: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_874, [1, 0])
        mm_147: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_667, view_264);  permute_667 = view_264 = None
        sum_214: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_874, [0], True);  view_874 = None
        view_875: "f32[1536]" = torch.ops.aten.reshape.default(sum_214, [1536]);  sum_214 = None
        view_876: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_146, [8, 512, 1536]);  mm_146 = None
        add_248: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_247, view_876);  add_247 = view_876 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_700: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_248, primals_197);  primals_197 = None
        mul_701: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_700, 1536)
        sum_215: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_700, [2], True)
        mul_702: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_700, mul_172);  mul_700 = None
        sum_216: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_702, [2], True);  mul_702 = None
        mul_703: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_172, sum_216);  sum_216 = None
        sub_153: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_701, sum_215);  mul_701 = sum_215 = None
        sub_154: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_153, mul_703);  sub_153 = mul_703 = None
        mul_704: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_87, sub_154);  div_87 = sub_154 = None
        mul_705: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_248, mul_172);  mul_172 = None
        sum_217: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_705, [0, 1]);  mul_705 = None
        sum_218: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_248, [0, 1]);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_61: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_36, torch.float32);  gt_36 = None
        mul_706: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 1.1111111111111112);  convert_element_type_61 = None
        mul_707: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_704, mul_706);  mul_706 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_877: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_707, [4096, 1536]);  mul_707 = None
        permute_131: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
        permute_670: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        mm_148: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_877, permute_670);  permute_670 = None
        permute_671: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_877, [1, 0])
        mm_149: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_671, view_262);  permute_671 = view_262 = None
        sum_219: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_877, [0], True);  view_877 = None
        view_878: "f32[1536]" = torch.ops.aten.reshape.default(sum_219, [1536]);  sum_219 = None
        view_879: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_148, [8, 512, 6144]);  mm_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_261: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_70, [8, 512, 6144]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_168: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
        erf_11: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_83: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_709: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_83, 0.5);  add_83 = None
        mul_710: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_261, view_261)
        mul_711: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_710, -0.5);  mul_710 = None
        exp_39: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_711);  mul_711 = None
        mul_712: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
        mul_713: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_261, mul_712);  view_261 = mul_712 = None
        add_250: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_709, mul_713);  mul_709 = mul_713 = None
        mul_714: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_879, add_250);  view_879 = add_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_880: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_714, [4096, 6144]);  mul_714 = None
        permute_130: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_193, [1, 0]);  primals_193 = None
        permute_674: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        mm_150: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_880, permute_674);  permute_674 = None
        permute_675: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_880, [1, 0])
        mm_151: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_675, view_260);  permute_675 = view_260 = None
        sum_220: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_880, [0], True);  view_880 = None
        view_881: "f32[6144]" = torch.ops.aten.reshape.default(sum_220, [6144]);  sum_220 = None
        view_882: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_150, [8, 512, 1536]);  mm_150 = None
        add_251: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_704, view_882);  mul_704 = view_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_716: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_251, primals_191);  primals_191 = None
        mul_717: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_716, 1536)
        sum_221: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_716, [2], True)
        mul_718: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_716, mul_165);  mul_716 = None
        sum_222: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_718, [2], True);  mul_718 = None
        mul_719: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_165, sum_222);  sum_222 = None
        sub_156: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_717, sum_221);  mul_717 = sum_221 = None
        sub_157: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_156, mul_719);  sub_156 = mul_719 = None
        mul_720: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_88, sub_157);  div_88 = sub_157 = None
        mul_721: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_251, mul_165);  mul_165 = None
        sum_223: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_721, [0, 1]);  mul_721 = None
        sum_224: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_251, [0, 1]);  add_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_62: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_35, torch.float32);  gt_35 = None
        mul_722: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
        mul_723: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_720, mul_722);  mul_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_883: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_723, [4096, 1536]);  mul_723 = None
        permute_129: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
        permute_678: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        mm_152: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_883, permute_678);  permute_678 = None
        permute_679: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_883, [1, 0])
        mm_153: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_679, view_258);  permute_679 = view_258 = None
        sum_225: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_883, [0], True);  view_883 = None
        view_884: "f32[1536]" = torch.ops.aten.reshape.default(sum_225, [1536]);  sum_225 = None
        view_885: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_152, [8, 512, 1536]);  mm_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_886: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_885, [8, 512, 24, 64]);  view_885 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_682: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_886, [0, 2, 1, 3]);  view_886 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_182: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_682, memory_format = torch.contiguous_format);  permute_682 = None
        view_887: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_182, [192, 512, 64]);  clone_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_96: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_683, view_887);  permute_683 = None
        bmm_97: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_887, permute_684);  view_887 = permute_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_888: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_97, [8, 24, 512, 512]);  bmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_63: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_34, torch.float32);  gt_34 = None
        mul_724: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.1111111111111112);  convert_element_type_63 = None
        mul_725: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_888, mul_724);  view_888 = mul_724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_726: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_725, div_23);  mul_725 = None
        sum_226: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_726, [-1], True)
        neg_13: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_23);  div_23 = None
        fma_12: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_13, sum_226, mul_726);  neg_13 = sum_226 = mul_726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_40: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_12);  fma_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_889: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_40, [192, 512, 512]);  where_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_98: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_685, view_889);  permute_685 = None
        bmm_99: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_889, permute_686);  view_889 = permute_686 = None
        div_89: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_98, full_default_1);  bmm_98 = None
        permute_687: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_89, [0, 2, 1]);  div_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_890: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_96, [8, 24, 512, 64]);  bmm_96 = None
        permute_688: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_890, [0, 2, 1, 3]);  view_890 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_184: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_688, memory_format = torch.contiguous_format);  permute_688 = None
        view_891: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_184, [8, 512, 1536]);  clone_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_892: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_891, [4096, 1536]);  view_891 = None
        permute_125: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
        permute_689: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        mm_154: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_892, permute_689);  permute_689 = None
        permute_690: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_892, [1, 0])
        mm_155: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_690, view_242);  permute_690 = None
        sum_227: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_892, [0], True);  view_892 = None
        view_893: "f32[1536]" = torch.ops.aten.reshape.default(sum_227, [1536]);  sum_227 = None
        view_894: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_154, [8, 512, 1536]);  mm_154 = None
        add_252: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_720, view_894);  mul_720 = view_894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_895: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_687, [8, 24, 512, 64]);  permute_687 = None
        permute_693: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_895, [0, 2, 1, 3]);  view_895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_896: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_693, [8, 512, 1536]);  permute_693 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_185: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_896, memory_format = torch.contiguous_format);  view_896 = None
        view_897: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_185, [4096, 1536]);  clone_185 = None
        permute_123: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
        permute_694: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        mm_156: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_897, permute_694);  permute_694 = None
        permute_695: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_897, [1, 0])
        mm_157: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_695, view_242);  permute_695 = None
        sum_228: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_897, [0], True);  view_897 = None
        view_898: "f32[1536]" = torch.ops.aten.reshape.default(sum_228, [1536]);  sum_228 = None
        view_899: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_156, [8, 512, 1536]);  mm_156 = None
        add_253: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_252, view_899);  add_252 = view_899 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_900: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_99, [8, 24, 512, 64]);  bmm_99 = None
        permute_698: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_900, [0, 2, 1, 3]);  view_900 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_186: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_698, memory_format = torch.contiguous_format);  permute_698 = None
        view_901: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_186, [8, 512, 1536]);  clone_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_902: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_901, [4096, 1536]);  view_901 = None
        permute_121: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
        permute_699: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_158: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_902, permute_699);  permute_699 = None
        permute_700: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_902, [1, 0])
        mm_159: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_700, view_242);  permute_700 = view_242 = None
        sum_229: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_902, [0], True);  view_902 = None
        view_903: "f32[1536]" = torch.ops.aten.reshape.default(sum_229, [1536]);  sum_229 = None
        view_904: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_158, [8, 512, 1536]);  mm_158 = None
        add_254: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_253, view_904);  add_253 = view_904 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_728: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_254, primals_181);  primals_181 = None
        mul_729: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_728, 1536)
        sum_230: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_728, [2], True)
        mul_730: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_728, mul_158);  mul_728 = None
        sum_231: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_730, [2], True);  mul_730 = None
        mul_731: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_158, sum_231);  sum_231 = None
        sub_159: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_729, sum_230);  mul_729 = sum_230 = None
        sub_160: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_159, mul_731);  sub_159 = mul_731 = None
        mul_732: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_90, sub_160);  div_90 = sub_160 = None
        mul_733: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_254, mul_158);  mul_158 = None
        sum_232: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 1]);  mul_733 = None
        sum_233: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_254, [0, 1]);  add_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_64: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_33, torch.float32);  gt_33 = None
        mul_734: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1.1111111111111112);  convert_element_type_64 = None
        mul_735: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_732, mul_734);  mul_734 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_905: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_735, [4096, 1536]);  mul_735 = None
        permute_120: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
        permute_703: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        mm_160: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_905, permute_703);  permute_703 = None
        permute_704: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_905, [1, 0])
        mm_161: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_704, view_240);  permute_704 = view_240 = None
        sum_234: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_905, [0], True);  view_905 = None
        view_906: "f32[1536]" = torch.ops.aten.reshape.default(sum_234, [1536]);  sum_234 = None
        view_907: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_160, [8, 512, 6144]);  mm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_239: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_64, [8, 512, 6144]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_154: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
        erf_10: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_154);  mul_154 = None
        add_76: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_737: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_76, 0.5);  add_76 = None
        mul_738: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_239, view_239)
        mul_739: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_738, -0.5);  mul_738 = None
        exp_40: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_739);  mul_739 = None
        mul_740: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
        mul_741: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_239, mul_740);  view_239 = mul_740 = None
        add_256: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_737, mul_741);  mul_737 = mul_741 = None
        mul_742: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_907, add_256);  view_907 = add_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_908: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_742, [4096, 6144]);  mul_742 = None
        permute_119: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
        permute_707: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        mm_162: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_908, permute_707);  permute_707 = None
        permute_708: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_908, [1, 0])
        mm_163: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_708, view_238);  permute_708 = view_238 = None
        sum_235: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_908, [0], True);  view_908 = None
        view_909: "f32[6144]" = torch.ops.aten.reshape.default(sum_235, [6144]);  sum_235 = None
        view_910: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_162, [8, 512, 1536]);  mm_162 = None
        add_257: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_732, view_910);  mul_732 = view_910 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_744: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_257, primals_175);  primals_175 = None
        mul_745: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_744, 1536)
        sum_236: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_744, [2], True)
        mul_746: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_744, mul_151);  mul_744 = None
        sum_237: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_746, [2], True);  mul_746 = None
        mul_747: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_151, sum_237);  sum_237 = None
        sub_162: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_745, sum_236);  mul_745 = sum_236 = None
        sub_163: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_162, mul_747);  sub_162 = mul_747 = None
        mul_748: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_91, sub_163);  div_91 = sub_163 = None
        mul_749: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_257, mul_151);  mul_151 = None
        sum_238: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_749, [0, 1]);  mul_749 = None
        sum_239: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_257, [0, 1]);  add_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_65: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_32, torch.float32);  gt_32 = None
        mul_750: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
        mul_751: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_748, mul_750);  mul_750 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_911: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_751, [4096, 1536]);  mul_751 = None
        permute_118: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
        permute_711: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        mm_164: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_911, permute_711);  permute_711 = None
        permute_712: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_911, [1, 0])
        mm_165: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_712, view_236);  permute_712 = view_236 = None
        sum_240: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_911, [0], True);  view_911 = None
        view_912: "f32[1536]" = torch.ops.aten.reshape.default(sum_240, [1536]);  sum_240 = None
        view_913: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_164, [8, 512, 1536]);  mm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_914: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_913, [8, 512, 24, 64]);  view_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_715: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_914, [0, 2, 1, 3]);  view_914 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_189: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_715, memory_format = torch.contiguous_format);  permute_715 = None
        view_915: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_189, [192, 512, 64]);  clone_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_100: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_716, view_915);  permute_716 = None
        bmm_101: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_915, permute_717);  view_915 = permute_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_916: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_101, [8, 24, 512, 512]);  bmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_66: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_31, torch.float32);  gt_31 = None
        mul_752: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
        mul_753: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_916, mul_752);  view_916 = mul_752 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_754: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_753, div_21);  mul_753 = None
        sum_241: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_754, [-1], True)
        neg_14: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_21);  div_21 = None
        fma_13: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_14, sum_241, mul_754);  neg_14 = sum_241 = mul_754 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_41: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_13);  fma_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_917: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_41, [192, 512, 512]);  where_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_102: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_718, view_917);  permute_718 = None
        bmm_103: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_917, permute_719);  view_917 = permute_719 = None
        div_92: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_102, full_default_1);  bmm_102 = None
        permute_720: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_92, [0, 2, 1]);  div_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_918: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_100, [8, 24, 512, 64]);  bmm_100 = None
        permute_721: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_918, [0, 2, 1, 3]);  view_918 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_191: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_721, memory_format = torch.contiguous_format);  permute_721 = None
        view_919: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_191, [8, 512, 1536]);  clone_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_920: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_919, [4096, 1536]);  view_919 = None
        permute_114: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
        permute_722: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
        mm_166: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_920, permute_722);  permute_722 = None
        permute_723: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_920, [1, 0])
        mm_167: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_723, view_220);  permute_723 = None
        sum_242: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_920, [0], True);  view_920 = None
        view_921: "f32[1536]" = torch.ops.aten.reshape.default(sum_242, [1536]);  sum_242 = None
        view_922: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_166, [8, 512, 1536]);  mm_166 = None
        add_258: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_748, view_922);  mul_748 = view_922 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_923: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_720, [8, 24, 512, 64]);  permute_720 = None
        permute_726: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_923, [0, 2, 1, 3]);  view_923 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_924: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_726, [8, 512, 1536]);  permute_726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_192: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_924, memory_format = torch.contiguous_format);  view_924 = None
        view_925: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_192, [4096, 1536]);  clone_192 = None
        permute_112: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
        permute_727: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        mm_168: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_925, permute_727);  permute_727 = None
        permute_728: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_925, [1, 0])
        mm_169: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_728, view_220);  permute_728 = None
        sum_243: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_925, [0], True);  view_925 = None
        view_926: "f32[1536]" = torch.ops.aten.reshape.default(sum_243, [1536]);  sum_243 = None
        view_927: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_168, [8, 512, 1536]);  mm_168 = None
        add_259: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_258, view_927);  add_258 = view_927 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_928: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_103, [8, 24, 512, 64]);  bmm_103 = None
        permute_731: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_928, [0, 2, 1, 3]);  view_928 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_193: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_731, memory_format = torch.contiguous_format);  permute_731 = None
        view_929: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_193, [8, 512, 1536]);  clone_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_930: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_929, [4096, 1536]);  view_929 = None
        permute_110: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
        permute_732: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        mm_170: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_930, permute_732);  permute_732 = None
        permute_733: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_930, [1, 0])
        mm_171: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_733, view_220);  permute_733 = view_220 = None
        sum_244: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_930, [0], True);  view_930 = None
        view_931: "f32[1536]" = torch.ops.aten.reshape.default(sum_244, [1536]);  sum_244 = None
        view_932: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_170, [8, 512, 1536]);  mm_170 = None
        add_260: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_259, view_932);  add_259 = view_932 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_756: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_260, primals_165);  primals_165 = None
        mul_757: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_756, 1536)
        sum_245: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_756, [2], True)
        mul_758: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_756, mul_144);  mul_756 = None
        sum_246: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_758, [2], True);  mul_758 = None
        mul_759: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_144, sum_246);  sum_246 = None
        sub_165: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_757, sum_245);  mul_757 = sum_245 = None
        sub_166: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_165, mul_759);  sub_165 = mul_759 = None
        mul_760: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_93, sub_166);  div_93 = sub_166 = None
        mul_761: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_260, mul_144);  mul_144 = None
        sum_247: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_761, [0, 1]);  mul_761 = None
        sum_248: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_260, [0, 1]);  add_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_67: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_30, torch.float32);  gt_30 = None
        mul_762: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.1111111111111112);  convert_element_type_67 = None
        mul_763: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_760, mul_762);  mul_762 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_933: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_763, [4096, 1536]);  mul_763 = None
        permute_109: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
        permute_736: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        mm_172: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_933, permute_736);  permute_736 = None
        permute_737: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_933, [1, 0])
        mm_173: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_737, view_218);  permute_737 = view_218 = None
        sum_249: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_933, [0], True);  view_933 = None
        view_934: "f32[1536]" = torch.ops.aten.reshape.default(sum_249, [1536]);  sum_249 = None
        view_935: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_172, [8, 512, 6144]);  mm_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_217: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_58, [8, 512, 6144]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_140: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
        erf_9: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_140);  mul_140 = None
        add_69: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_765: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
        mul_766: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_217, view_217)
        mul_767: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_766, -0.5);  mul_766 = None
        exp_41: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_767);  mul_767 = None
        mul_768: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
        mul_769: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_217, mul_768);  view_217 = mul_768 = None
        add_262: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_765, mul_769);  mul_765 = mul_769 = None
        mul_770: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_935, add_262);  view_935 = add_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_936: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_770, [4096, 6144]);  mul_770 = None
        permute_108: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
        permute_740: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        mm_174: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_936, permute_740);  permute_740 = None
        permute_741: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_936, [1, 0])
        mm_175: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_741, view_216);  permute_741 = view_216 = None
        sum_250: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_936, [0], True);  view_936 = None
        view_937: "f32[6144]" = torch.ops.aten.reshape.default(sum_250, [6144]);  sum_250 = None
        view_938: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_174, [8, 512, 1536]);  mm_174 = None
        add_263: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_760, view_938);  mul_760 = view_938 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_772: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_263, primals_159);  primals_159 = None
        mul_773: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_772, 1536)
        sum_251: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_772, [2], True)
        mul_774: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_772, mul_137);  mul_772 = None
        sum_252: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_774, [2], True);  mul_774 = None
        mul_775: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_137, sum_252);  sum_252 = None
        sub_168: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_773, sum_251);  mul_773 = sum_251 = None
        sub_169: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_168, mul_775);  sub_168 = mul_775 = None
        mul_776: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_94, sub_169);  div_94 = sub_169 = None
        mul_777: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_263, mul_137);  mul_137 = None
        sum_253: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_777, [0, 1]);  mul_777 = None
        sum_254: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_68: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_29, torch.float32);  gt_29 = None
        mul_778: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
        mul_779: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_776, mul_778);  mul_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_939: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_779, [4096, 1536]);  mul_779 = None
        permute_107: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
        permute_744: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        mm_176: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_939, permute_744);  permute_744 = None
        permute_745: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_939, [1, 0])
        mm_177: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_745, view_214);  permute_745 = view_214 = None
        sum_255: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_939, [0], True);  view_939 = None
        view_940: "f32[1536]" = torch.ops.aten.reshape.default(sum_255, [1536]);  sum_255 = None
        view_941: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_176, [8, 512, 1536]);  mm_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_942: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_941, [8, 512, 24, 64]);  view_941 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_748: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_942, [0, 2, 1, 3]);  view_942 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_196: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_748, memory_format = torch.contiguous_format);  permute_748 = None
        view_943: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_196, [192, 512, 64]);  clone_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_104: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_749, view_943);  permute_749 = None
        bmm_105: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_943, permute_750);  view_943 = permute_750 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_944: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_105, [8, 24, 512, 512]);  bmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_69: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_28, torch.float32);  gt_28 = None
        mul_780: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_69, 1.1111111111111112);  convert_element_type_69 = None
        mul_781: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_944, mul_780);  view_944 = mul_780 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_782: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_781, div_19);  mul_781 = None
        sum_256: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_782, [-1], True)
        neg_15: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_19);  div_19 = None
        fma_14: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_15, sum_256, mul_782);  neg_15 = sum_256 = mul_782 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_42: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_14);  fma_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_945: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_42, [192, 512, 512]);  where_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_106: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_751, view_945);  permute_751 = None
        bmm_107: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_945, permute_752);  view_945 = permute_752 = None
        div_95: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_106, full_default_1);  bmm_106 = None
        permute_753: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_95, [0, 2, 1]);  div_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_946: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_104, [8, 24, 512, 64]);  bmm_104 = None
        permute_754: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_946, [0, 2, 1, 3]);  view_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_198: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_754, memory_format = torch.contiguous_format);  permute_754 = None
        view_947: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_198, [8, 512, 1536]);  clone_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_948: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_947, [4096, 1536]);  view_947 = None
        permute_103: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
        permute_755: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        mm_178: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_948, permute_755);  permute_755 = None
        permute_756: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_948, [1, 0])
        mm_179: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_756, view_198);  permute_756 = None
        sum_257: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_948, [0], True);  view_948 = None
        view_949: "f32[1536]" = torch.ops.aten.reshape.default(sum_257, [1536]);  sum_257 = None
        view_950: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_178, [8, 512, 1536]);  mm_178 = None
        add_264: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_776, view_950);  mul_776 = view_950 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_951: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_753, [8, 24, 512, 64]);  permute_753 = None
        permute_759: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_951, [0, 2, 1, 3]);  view_951 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_952: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_759, [8, 512, 1536]);  permute_759 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_199: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_952, memory_format = torch.contiguous_format);  view_952 = None
        view_953: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_199, [4096, 1536]);  clone_199 = None
        permute_101: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
        permute_760: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_180: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_953, permute_760);  permute_760 = None
        permute_761: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_953, [1, 0])
        mm_181: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_761, view_198);  permute_761 = None
        sum_258: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_953, [0], True);  view_953 = None
        view_954: "f32[1536]" = torch.ops.aten.reshape.default(sum_258, [1536]);  sum_258 = None
        view_955: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_180, [8, 512, 1536]);  mm_180 = None
        add_265: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_264, view_955);  add_264 = view_955 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_956: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_107, [8, 24, 512, 64]);  bmm_107 = None
        permute_764: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_956, [0, 2, 1, 3]);  view_956 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_200: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_764, memory_format = torch.contiguous_format);  permute_764 = None
        view_957: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_200, [8, 512, 1536]);  clone_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_958: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_957, [4096, 1536]);  view_957 = None
        permute_99: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
        permute_765: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_182: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_958, permute_765);  permute_765 = None
        permute_766: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_958, [1, 0])
        mm_183: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_766, view_198);  permute_766 = view_198 = None
        sum_259: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_958, [0], True);  view_958 = None
        view_959: "f32[1536]" = torch.ops.aten.reshape.default(sum_259, [1536]);  sum_259 = None
        view_960: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_182, [8, 512, 1536]);  mm_182 = None
        add_266: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_265, view_960);  add_265 = view_960 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_784: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_266, primals_149);  primals_149 = None
        mul_785: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_784, 1536)
        sum_260: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_784, [2], True)
        mul_786: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_784, mul_130);  mul_784 = None
        sum_261: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_786, [2], True);  mul_786 = None
        mul_787: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_130, sum_261);  sum_261 = None
        sub_171: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_785, sum_260);  mul_785 = sum_260 = None
        sub_172: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_171, mul_787);  sub_171 = mul_787 = None
        mul_788: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_96, sub_172);  div_96 = sub_172 = None
        mul_789: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_266, mul_130);  mul_130 = None
        sum_262: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_789, [0, 1]);  mul_789 = None
        sum_263: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_70: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_790: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 1.1111111111111112);  convert_element_type_70 = None
        mul_791: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_788, mul_790);  mul_790 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_961: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_791, [4096, 1536]);  mul_791 = None
        permute_98: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
        permute_769: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        mm_184: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_961, permute_769);  permute_769 = None
        permute_770: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_961, [1, 0])
        mm_185: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_770, view_196);  permute_770 = view_196 = None
        sum_264: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_961, [0], True);  view_961 = None
        view_962: "f32[1536]" = torch.ops.aten.reshape.default(sum_264, [1536]);  sum_264 = None
        view_963: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_184, [8, 512, 6144]);  mm_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_195: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_52, [8, 512, 6144]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_126: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
        erf_8: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_126);  mul_126 = None
        add_62: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_793: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
        mul_794: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_195, view_195)
        mul_795: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_794, -0.5);  mul_794 = None
        exp_42: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_795);  mul_795 = None
        mul_796: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
        mul_797: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_195, mul_796);  view_195 = mul_796 = None
        add_268: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_793, mul_797);  mul_793 = mul_797 = None
        mul_798: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_963, add_268);  view_963 = add_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_964: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_798, [4096, 6144]);  mul_798 = None
        permute_97: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
        permute_773: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_186: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_964, permute_773);  permute_773 = None
        permute_774: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_964, [1, 0])
        mm_187: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_774, view_194);  permute_774 = view_194 = None
        sum_265: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_964, [0], True);  view_964 = None
        view_965: "f32[6144]" = torch.ops.aten.reshape.default(sum_265, [6144]);  sum_265 = None
        view_966: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_186, [8, 512, 1536]);  mm_186 = None
        add_269: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_788, view_966);  mul_788 = view_966 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_800: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_269, primals_143);  primals_143 = None
        mul_801: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_800, 1536)
        sum_266: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_800, [2], True)
        mul_802: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_800, mul_123);  mul_800 = None
        sum_267: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_802, [2], True);  mul_802 = None
        mul_803: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_123, sum_267);  sum_267 = None
        sub_174: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_801, sum_266);  mul_801 = sum_266 = None
        sub_175: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_174, mul_803);  sub_174 = mul_803 = None
        mul_804: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_97, sub_175);  div_97 = sub_175 = None
        mul_805: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_269, mul_123);  mul_123 = None
        sum_268: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_805, [0, 1]);  mul_805 = None
        sum_269: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_269, [0, 1]);  add_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_71: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_806: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
        mul_807: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_804, mul_806);  mul_806 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_967: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_807, [4096, 1536]);  mul_807 = None
        permute_96: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        permute_777: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        mm_188: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_967, permute_777);  permute_777 = None
        permute_778: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_967, [1, 0])
        mm_189: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_778, view_192);  permute_778 = view_192 = None
        sum_270: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_967, [0], True);  view_967 = None
        view_968: "f32[1536]" = torch.ops.aten.reshape.default(sum_270, [1536]);  sum_270 = None
        view_969: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_188, [8, 512, 1536]);  mm_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_970: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_969, [8, 512, 24, 64]);  view_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_781: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_970, [0, 2, 1, 3]);  view_970 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_203: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_781, memory_format = torch.contiguous_format);  permute_781 = None
        view_971: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_203, [192, 512, 64]);  clone_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_108: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_782, view_971);  permute_782 = None
        bmm_109: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_971, permute_783);  view_971 = permute_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_972: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_109, [8, 24, 512, 512]);  bmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_72: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_808: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 1.1111111111111112);  convert_element_type_72 = None
        mul_809: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_972, mul_808);  view_972 = mul_808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_810: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_809, div_17);  mul_809 = None
        sum_271: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_810, [-1], True)
        neg_16: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_17);  div_17 = None
        fma_15: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_16, sum_271, mul_810);  neg_16 = sum_271 = mul_810 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_43: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_15);  fma_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_973: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_43, [192, 512, 512]);  where_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_110: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_784, view_973);  permute_784 = None
        bmm_111: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_973, permute_785);  view_973 = permute_785 = None
        div_98: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_110, full_default_1);  bmm_110 = None
        permute_786: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_98, [0, 2, 1]);  div_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_974: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_108, [8, 24, 512, 64]);  bmm_108 = None
        permute_787: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_974, [0, 2, 1, 3]);  view_974 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_205: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_787, memory_format = torch.contiguous_format);  permute_787 = None
        view_975: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_205, [8, 512, 1536]);  clone_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_976: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_975, [4096, 1536]);  view_975 = None
        permute_92: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        permute_788: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        mm_190: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_976, permute_788);  permute_788 = None
        permute_789: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_976, [1, 0])
        mm_191: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_789, view_176);  permute_789 = None
        sum_272: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_976, [0], True);  view_976 = None
        view_977: "f32[1536]" = torch.ops.aten.reshape.default(sum_272, [1536]);  sum_272 = None
        view_978: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_190, [8, 512, 1536]);  mm_190 = None
        add_270: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_804, view_978);  mul_804 = view_978 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_979: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_786, [8, 24, 512, 64]);  permute_786 = None
        permute_792: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_979, [0, 2, 1, 3]);  view_979 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_980: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_792, [8, 512, 1536]);  permute_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_206: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_980, memory_format = torch.contiguous_format);  view_980 = None
        view_981: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_206, [4096, 1536]);  clone_206 = None
        permute_90: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
        permute_793: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        mm_192: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_981, permute_793);  permute_793 = None
        permute_794: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_981, [1, 0])
        mm_193: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_794, view_176);  permute_794 = None
        sum_273: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_981, [0], True);  view_981 = None
        view_982: "f32[1536]" = torch.ops.aten.reshape.default(sum_273, [1536]);  sum_273 = None
        view_983: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_192, [8, 512, 1536]);  mm_192 = None
        add_271: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_270, view_983);  add_270 = view_983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_984: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_111, [8, 24, 512, 64]);  bmm_111 = None
        permute_797: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_984, [0, 2, 1, 3]);  view_984 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_207: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_797, memory_format = torch.contiguous_format);  permute_797 = None
        view_985: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_207, [8, 512, 1536]);  clone_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_986: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_985, [4096, 1536]);  view_985 = None
        permute_88: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
        permute_798: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_194: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_986, permute_798);  permute_798 = None
        permute_799: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_986, [1, 0])
        mm_195: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_799, view_176);  permute_799 = view_176 = None
        sum_274: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_986, [0], True);  view_986 = None
        view_987: "f32[1536]" = torch.ops.aten.reshape.default(sum_274, [1536]);  sum_274 = None
        view_988: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_194, [8, 512, 1536]);  mm_194 = None
        add_272: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_271, view_988);  add_271 = view_988 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_812: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_272, primals_133);  primals_133 = None
        mul_813: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_812, 1536)
        sum_275: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_812, [2], True)
        mul_814: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_812, mul_116);  mul_812 = None
        sum_276: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_814, [2], True);  mul_814 = None
        mul_815: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_116, sum_276);  sum_276 = None
        sub_177: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_813, sum_275);  mul_813 = sum_275 = None
        sub_178: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_177, mul_815);  sub_177 = mul_815 = None
        mul_816: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_99, sub_178);  div_99 = sub_178 = None
        mul_817: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_272, mul_116);  mul_116 = None
        sum_277: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_817, [0, 1]);  mul_817 = None
        sum_278: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_272, [0, 1]);  add_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_73: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_818: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_73, 1.1111111111111112);  convert_element_type_73 = None
        mul_819: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_816, mul_818);  mul_818 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_989: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_819, [4096, 1536]);  mul_819 = None
        permute_87: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
        permute_802: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_196: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_989, permute_802);  permute_802 = None
        permute_803: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_989, [1, 0])
        mm_197: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_803, view_174);  permute_803 = view_174 = None
        sum_279: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_989, [0], True);  view_989 = None
        view_990: "f32[1536]" = torch.ops.aten.reshape.default(sum_279, [1536]);  sum_279 = None
        view_991: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_196, [8, 512, 6144]);  mm_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_173: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_46, [8, 512, 6144]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_112: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
        erf_7: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_55: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_821: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_55, 0.5);  add_55 = None
        mul_822: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_173, view_173)
        mul_823: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_822, -0.5);  mul_822 = None
        exp_43: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_823);  mul_823 = None
        mul_824: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
        mul_825: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_173, mul_824);  view_173 = mul_824 = None
        add_274: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_821, mul_825);  mul_821 = mul_825 = None
        mul_826: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_991, add_274);  view_991 = add_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_992: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_826, [4096, 6144]);  mul_826 = None
        permute_86: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
        permute_806: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_198: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_992, permute_806);  permute_806 = None
        permute_807: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_992, [1, 0])
        mm_199: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_807, view_172);  permute_807 = view_172 = None
        sum_280: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_992, [0], True);  view_992 = None
        view_993: "f32[6144]" = torch.ops.aten.reshape.default(sum_280, [6144]);  sum_280 = None
        view_994: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_198, [8, 512, 1536]);  mm_198 = None
        add_275: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_816, view_994);  mul_816 = view_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_828: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_275, primals_127);  primals_127 = None
        mul_829: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_828, 1536)
        sum_281: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_828, [2], True)
        mul_830: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_828, mul_109);  mul_828 = None
        sum_282: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_830, [2], True);  mul_830 = None
        mul_831: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_109, sum_282);  sum_282 = None
        sub_180: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_829, sum_281);  mul_829 = sum_281 = None
        sub_181: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_180, mul_831);  sub_180 = mul_831 = None
        mul_832: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_100, sub_181);  div_100 = sub_181 = None
        mul_833: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_275, mul_109);  mul_109 = None
        sum_283: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_833, [0, 1]);  mul_833 = None
        sum_284: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_275, [0, 1]);  add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_74: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_834: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_74, 1.1111111111111112);  convert_element_type_74 = None
        mul_835: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_832, mul_834);  mul_834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_995: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_835, [4096, 1536]);  mul_835 = None
        permute_85: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
        permute_810: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        mm_200: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_995, permute_810);  permute_810 = None
        permute_811: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_995, [1, 0])
        mm_201: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_811, view_170);  permute_811 = view_170 = None
        sum_285: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_995, [0], True);  view_995 = None
        view_996: "f32[1536]" = torch.ops.aten.reshape.default(sum_285, [1536]);  sum_285 = None
        view_997: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_200, [8, 512, 1536]);  mm_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_998: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_997, [8, 512, 24, 64]);  view_997 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_814: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_998, [0, 2, 1, 3]);  view_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_210: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_814, memory_format = torch.contiguous_format);  permute_814 = None
        view_999: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_210, [192, 512, 64]);  clone_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_112: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_815, view_999);  permute_815 = None
        bmm_113: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_999, permute_816);  view_999 = permute_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_1000: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_113, [8, 24, 512, 512]);  bmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_75: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_836: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_75, 1.1111111111111112);  convert_element_type_75 = None
        mul_837: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_1000, mul_836);  view_1000 = mul_836 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_838: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_837, div_15);  mul_837 = None
        sum_286: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_838, [-1], True)
        neg_17: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_15);  div_15 = None
        fma_16: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_17, sum_286, mul_838);  neg_17 = sum_286 = mul_838 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_44: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_16);  fma_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_1001: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_44, [192, 512, 512]);  where_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_114: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_817, view_1001);  permute_817 = None
        bmm_115: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_1001, permute_818);  view_1001 = permute_818 = None
        div_101: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_114, full_default_1);  bmm_114 = None
        permute_819: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_101, [0, 2, 1]);  div_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1002: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_112, [8, 24, 512, 64]);  bmm_112 = None
        permute_820: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1002, [0, 2, 1, 3]);  view_1002 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_212: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_820, memory_format = torch.contiguous_format);  permute_820 = None
        view_1003: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_212, [8, 512, 1536]);  clone_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_1004: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_1003, [4096, 1536]);  view_1003 = None
        permute_81: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
        permute_821: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        mm_202: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1004, permute_821);  permute_821 = None
        permute_822: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1004, [1, 0])
        mm_203: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_822, view_154);  permute_822 = None
        sum_287: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1004, [0], True);  view_1004 = None
        view_1005: "f32[1536]" = torch.ops.aten.reshape.default(sum_287, [1536]);  sum_287 = None
        view_1006: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_202, [8, 512, 1536]);  mm_202 = None
        add_276: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_832, view_1006);  mul_832 = view_1006 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1007: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_819, [8, 24, 512, 64]);  permute_819 = None
        permute_825: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1007, [0, 2, 1, 3]);  view_1007 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_1008: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_825, [8, 512, 1536]);  permute_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_213: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_1008, memory_format = torch.contiguous_format);  view_1008 = None
        view_1009: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_213, [4096, 1536]);  clone_213 = None
        permute_79: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
        permute_826: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        mm_204: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1009, permute_826);  permute_826 = None
        permute_827: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1009, [1, 0])
        mm_205: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_827, view_154);  permute_827 = None
        sum_288: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1009, [0], True);  view_1009 = None
        view_1010: "f32[1536]" = torch.ops.aten.reshape.default(sum_288, [1536]);  sum_288 = None
        view_1011: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_204, [8, 512, 1536]);  mm_204 = None
        add_277: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_276, view_1011);  add_276 = view_1011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1012: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_115, [8, 24, 512, 64]);  bmm_115 = None
        permute_830: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1012, [0, 2, 1, 3]);  view_1012 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_214: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_830, memory_format = torch.contiguous_format);  permute_830 = None
        view_1013: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_214, [8, 512, 1536]);  clone_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_1014: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_1013, [4096, 1536]);  view_1013 = None
        permute_77: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
        permute_831: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_206: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1014, permute_831);  permute_831 = None
        permute_832: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1014, [1, 0])
        mm_207: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_832, view_154);  permute_832 = view_154 = None
        sum_289: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True);  view_1014 = None
        view_1015: "f32[1536]" = torch.ops.aten.reshape.default(sum_289, [1536]);  sum_289 = None
        view_1016: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_206, [8, 512, 1536]);  mm_206 = None
        add_278: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_277, view_1016);  add_277 = view_1016 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_840: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_278, primals_117);  primals_117 = None
        mul_841: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_840, 1536)
        sum_290: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_840, [2], True)
        mul_842: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_840, mul_102);  mul_840 = None
        sum_291: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_842, [2], True);  mul_842 = None
        mul_843: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_102, sum_291);  sum_291 = None
        sub_183: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_841, sum_290);  mul_841 = sum_290 = None
        sub_184: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_183, mul_843);  sub_183 = mul_843 = None
        mul_844: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_102, sub_184);  div_102 = sub_184 = None
        mul_845: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_278, mul_102);  mul_102 = None
        sum_292: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_845, [0, 1]);  mul_845 = None
        sum_293: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 1]);  add_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_76: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_846: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_76, 1.1111111111111112);  convert_element_type_76 = None
        mul_847: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_844, mul_846);  mul_846 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_1017: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_847, [4096, 1536]);  mul_847 = None
        permute_76: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        permute_835: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_208: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_1017, permute_835);  permute_835 = None
        permute_836: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1017, [1, 0])
        mm_209: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_836, view_152);  permute_836 = view_152 = None
        sum_294: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1017, [0], True);  view_1017 = None
        view_1018: "f32[1536]" = torch.ops.aten.reshape.default(sum_294, [1536]);  sum_294 = None
        view_1019: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_208, [8, 512, 6144]);  mm_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_151: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_40, [8, 512, 6144]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_98: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
        erf_6: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
        add_48: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_849: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
        mul_850: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_151, view_151)
        mul_851: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_850, -0.5);  mul_850 = None
        exp_44: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_851);  mul_851 = None
        mul_852: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
        mul_853: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_151, mul_852);  view_151 = mul_852 = None
        add_280: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_849, mul_853);  mul_849 = mul_853 = None
        mul_854: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_1019, add_280);  view_1019 = add_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_1020: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_854, [4096, 6144]);  mul_854 = None
        permute_75: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
        permute_839: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_210: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1020, permute_839);  permute_839 = None
        permute_840: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_1020, [1, 0])
        mm_211: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_840, view_150);  permute_840 = view_150 = None
        sum_295: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1020, [0], True);  view_1020 = None
        view_1021: "f32[6144]" = torch.ops.aten.reshape.default(sum_295, [6144]);  sum_295 = None
        view_1022: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_210, [8, 512, 1536]);  mm_210 = None
        add_281: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_844, view_1022);  mul_844 = view_1022 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_856: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_281, primals_111);  primals_111 = None
        mul_857: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_856, 1536)
        sum_296: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_856, [2], True)
        mul_858: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_856, mul_95);  mul_856 = None
        sum_297: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_858, [2], True);  mul_858 = None
        mul_859: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_95, sum_297);  sum_297 = None
        sub_186: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_857, sum_296);  mul_857 = sum_296 = None
        sub_187: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_186, mul_859);  sub_186 = mul_859 = None
        mul_860: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_103, sub_187);  div_103 = sub_187 = None
        mul_861: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_281, mul_95);  mul_95 = None
        sum_298: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_861, [0, 1]);  mul_861 = None
        sum_299: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_281, [0, 1]);  add_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_77: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_862: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_77, 1.1111111111111112);  convert_element_type_77 = None
        mul_863: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_860, mul_862);  mul_862 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_1023: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_863, [4096, 1536]);  mul_863 = None
        permute_74: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        permute_843: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        mm_212: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1023, permute_843);  permute_843 = None
        permute_844: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1023, [1, 0])
        mm_213: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_844, view_148);  permute_844 = view_148 = None
        sum_300: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1023, [0], True);  view_1023 = None
        view_1024: "f32[1536]" = torch.ops.aten.reshape.default(sum_300, [1536]);  sum_300 = None
        view_1025: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_212, [8, 512, 1536]);  mm_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1026: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_1025, [8, 512, 24, 64]);  view_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_847: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_1026, [0, 2, 1, 3]);  view_1026 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_217: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_847, memory_format = torch.contiguous_format);  permute_847 = None
        view_1027: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_217, [192, 512, 64]);  clone_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_116: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_848, view_1027);  permute_848 = None
        bmm_117: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_1027, permute_849);  view_1027 = permute_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_1028: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_117, [8, 24, 512, 512]);  bmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_78: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_19, torch.float32);  gt_19 = None
        mul_864: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_78, 1.1111111111111112);  convert_element_type_78 = None
        mul_865: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_1028, mul_864);  view_1028 = mul_864 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_866: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_865, div_13);  mul_865 = None
        sum_301: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_866, [-1], True)
        neg_18: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_17: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_18, sum_301, mul_866);  neg_18 = sum_301 = mul_866 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_45: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_17);  fma_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_1029: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_45, [192, 512, 512]);  where_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_118: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_850, view_1029);  permute_850 = None
        bmm_119: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_1029, permute_851);  view_1029 = permute_851 = None
        div_104: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_118, full_default_1);  bmm_118 = None
        permute_852: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_104, [0, 2, 1]);  div_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1030: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_116, [8, 24, 512, 64]);  bmm_116 = None
        permute_853: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1030, [0, 2, 1, 3]);  view_1030 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_219: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_853, memory_format = torch.contiguous_format);  permute_853 = None
        view_1031: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_219, [8, 512, 1536]);  clone_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_1032: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_1031, [4096, 1536]);  view_1031 = None
        permute_70: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
        permute_854: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        mm_214: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1032, permute_854);  permute_854 = None
        permute_855: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1032, [1, 0])
        mm_215: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_855, view_132);  permute_855 = None
        sum_302: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1032, [0], True);  view_1032 = None
        view_1033: "f32[1536]" = torch.ops.aten.reshape.default(sum_302, [1536]);  sum_302 = None
        view_1034: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_214, [8, 512, 1536]);  mm_214 = None
        add_282: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_860, view_1034);  mul_860 = view_1034 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1035: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_852, [8, 24, 512, 64]);  permute_852 = None
        permute_858: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1035, [0, 2, 1, 3]);  view_1035 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_1036: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_858, [8, 512, 1536]);  permute_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_220: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_1036, memory_format = torch.contiguous_format);  view_1036 = None
        view_1037: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_220, [4096, 1536]);  clone_220 = None
        permute_68: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        permute_859: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        mm_216: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1037, permute_859);  permute_859 = None
        permute_860: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1037, [1, 0])
        mm_217: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_860, view_132);  permute_860 = None
        sum_303: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1037, [0], True);  view_1037 = None
        view_1038: "f32[1536]" = torch.ops.aten.reshape.default(sum_303, [1536]);  sum_303 = None
        view_1039: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_216, [8, 512, 1536]);  mm_216 = None
        add_283: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_282, view_1039);  add_282 = view_1039 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1040: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_119, [8, 24, 512, 64]);  bmm_119 = None
        permute_863: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1040, [0, 2, 1, 3]);  view_1040 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_221: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_863, memory_format = torch.contiguous_format);  permute_863 = None
        view_1041: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_221, [8, 512, 1536]);  clone_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_1042: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_1041, [4096, 1536]);  view_1041 = None
        permute_66: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        permute_864: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_218: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1042, permute_864);  permute_864 = None
        permute_865: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1042, [1, 0])
        mm_219: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_865, view_132);  permute_865 = view_132 = None
        sum_304: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1042, [0], True);  view_1042 = None
        view_1043: "f32[1536]" = torch.ops.aten.reshape.default(sum_304, [1536]);  sum_304 = None
        view_1044: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_218, [8, 512, 1536]);  mm_218 = None
        add_284: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_283, view_1044);  add_283 = view_1044 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_868: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_284, primals_101);  primals_101 = None
        mul_869: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_868, 1536)
        sum_305: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_868, [2], True)
        mul_870: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_868, mul_88);  mul_868 = None
        sum_306: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_870, [2], True);  mul_870 = None
        mul_871: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_88, sum_306);  sum_306 = None
        sub_189: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_869, sum_305);  mul_869 = sum_305 = None
        sub_190: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_189, mul_871);  sub_189 = mul_871 = None
        mul_872: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_105, sub_190);  div_105 = sub_190 = None
        mul_873: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_284, mul_88);  mul_88 = None
        sum_307: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_873, [0, 1]);  mul_873 = None
        sum_308: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_284, [0, 1]);  add_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_79: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_874: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_79, 1.1111111111111112);  convert_element_type_79 = None
        mul_875: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_872, mul_874);  mul_874 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_1045: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_875, [4096, 1536]);  mul_875 = None
        permute_65: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        permute_868: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_220: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_1045, permute_868);  permute_868 = None
        permute_869: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1045, [1, 0])
        mm_221: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_869, view_130);  permute_869 = view_130 = None
        sum_309: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1045, [0], True);  view_1045 = None
        view_1046: "f32[1536]" = torch.ops.aten.reshape.default(sum_309, [1536]);  sum_309 = None
        view_1047: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_220, [8, 512, 6144]);  mm_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_129: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_34, [8, 512, 6144]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_84: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
        erf_5: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
        add_41: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_877: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_41, 0.5);  add_41 = None
        mul_878: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_129, view_129)
        mul_879: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_878, -0.5);  mul_878 = None
        exp_45: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_879);  mul_879 = None
        mul_880: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
        mul_881: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_129, mul_880);  view_129 = mul_880 = None
        add_286: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_877, mul_881);  mul_877 = mul_881 = None
        mul_882: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_1047, add_286);  view_1047 = add_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_1048: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_882, [4096, 6144]);  mul_882 = None
        permute_64: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        permute_872: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_222: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1048, permute_872);  permute_872 = None
        permute_873: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_1048, [1, 0])
        mm_223: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_873, view_128);  permute_873 = view_128 = None
        sum_310: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1048, [0], True);  view_1048 = None
        view_1049: "f32[6144]" = torch.ops.aten.reshape.default(sum_310, [6144]);  sum_310 = None
        view_1050: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_222, [8, 512, 1536]);  mm_222 = None
        add_287: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_872, view_1050);  mul_872 = view_1050 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_884: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_287, primals_95);  primals_95 = None
        mul_885: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_884, 1536)
        sum_311: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_884, [2], True)
        mul_886: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_884, mul_81);  mul_884 = None
        sum_312: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_886, [2], True);  mul_886 = None
        mul_887: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_81, sum_312);  sum_312 = None
        sub_192: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_885, sum_311);  mul_885 = sum_311 = None
        sub_193: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_192, mul_887);  sub_192 = mul_887 = None
        mul_888: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_106, sub_193);  div_106 = sub_193 = None
        mul_889: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_287, mul_81);  mul_81 = None
        sum_313: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_889, [0, 1]);  mul_889 = None
        sum_314: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_287, [0, 1]);  add_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_80: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_890: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_80, 1.1111111111111112);  convert_element_type_80 = None
        mul_891: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_888, mul_890);  mul_890 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_1051: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_891, [4096, 1536]);  mul_891 = None
        permute_63: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        permute_876: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_224: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1051, permute_876);  permute_876 = None
        permute_877: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1051, [1, 0])
        mm_225: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_877, view_126);  permute_877 = view_126 = None
        sum_315: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1051, [0], True);  view_1051 = None
        view_1052: "f32[1536]" = torch.ops.aten.reshape.default(sum_315, [1536]);  sum_315 = None
        view_1053: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_224, [8, 512, 1536]);  mm_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1054: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_1053, [8, 512, 24, 64]);  view_1053 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_880: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_1054, [0, 2, 1, 3]);  view_1054 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_224: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_880, memory_format = torch.contiguous_format);  permute_880 = None
        view_1055: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_224, [192, 512, 64]);  clone_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_120: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_881, view_1055);  permute_881 = None
        bmm_121: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_1055, permute_882);  view_1055 = permute_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_1056: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_121, [8, 24, 512, 512]);  bmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_81: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_892: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_81, 1.1111111111111112);  convert_element_type_81 = None
        mul_893: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_1056, mul_892);  view_1056 = mul_892 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_894: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_893, div_11);  mul_893 = None
        sum_316: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_894, [-1], True)
        neg_19: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_11);  div_11 = None
        fma_18: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_19, sum_316, mul_894);  neg_19 = sum_316 = mul_894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_46: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_18);  fma_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_1057: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_46, [192, 512, 512]);  where_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_122: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_883, view_1057);  permute_883 = None
        bmm_123: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_1057, permute_884);  view_1057 = permute_884 = None
        div_107: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_122, full_default_1);  bmm_122 = None
        permute_885: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_107, [0, 2, 1]);  div_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1058: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_120, [8, 24, 512, 64]);  bmm_120 = None
        permute_886: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1058, [0, 2, 1, 3]);  view_1058 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_226: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_886, memory_format = torch.contiguous_format);  permute_886 = None
        view_1059: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_226, [8, 512, 1536]);  clone_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_1060: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_1059, [4096, 1536]);  view_1059 = None
        permute_59: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        permute_887: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        mm_226: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1060, permute_887);  permute_887 = None
        permute_888: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1060, [1, 0])
        mm_227: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_888, view_110);  permute_888 = None
        sum_317: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1060, [0], True);  view_1060 = None
        view_1061: "f32[1536]" = torch.ops.aten.reshape.default(sum_317, [1536]);  sum_317 = None
        view_1062: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_226, [8, 512, 1536]);  mm_226 = None
        add_288: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_888, view_1062);  mul_888 = view_1062 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1063: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_885, [8, 24, 512, 64]);  permute_885 = None
        permute_891: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1063, [0, 2, 1, 3]);  view_1063 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_1064: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_891, [8, 512, 1536]);  permute_891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_227: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_1064, memory_format = torch.contiguous_format);  view_1064 = None
        view_1065: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_227, [4096, 1536]);  clone_227 = None
        permute_57: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
        permute_892: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_228: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1065, permute_892);  permute_892 = None
        permute_893: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1065, [1, 0])
        mm_229: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_893, view_110);  permute_893 = None
        sum_318: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1065, [0], True);  view_1065 = None
        view_1066: "f32[1536]" = torch.ops.aten.reshape.default(sum_318, [1536]);  sum_318 = None
        view_1067: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_228, [8, 512, 1536]);  mm_228 = None
        add_289: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_288, view_1067);  add_288 = view_1067 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1068: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_123, [8, 24, 512, 64]);  bmm_123 = None
        permute_896: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1068, [0, 2, 1, 3]);  view_1068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_228: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_896, memory_format = torch.contiguous_format);  permute_896 = None
        view_1069: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_228, [8, 512, 1536]);  clone_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_1070: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_1069, [4096, 1536]);  view_1069 = None
        permute_55: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        permute_897: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_230: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1070, permute_897);  permute_897 = None
        permute_898: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1070, [1, 0])
        mm_231: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_898, view_110);  permute_898 = view_110 = None
        sum_319: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1070, [0], True);  view_1070 = None
        view_1071: "f32[1536]" = torch.ops.aten.reshape.default(sum_319, [1536]);  sum_319 = None
        view_1072: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_230, [8, 512, 1536]);  mm_230 = None
        add_290: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_289, view_1072);  add_289 = view_1072 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_896: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_290, primals_85);  primals_85 = None
        mul_897: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_896, 1536)
        sum_320: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_896, [2], True)
        mul_898: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_896, mul_74);  mul_896 = None
        sum_321: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_898, [2], True);  mul_898 = None
        mul_899: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_74, sum_321);  sum_321 = None
        sub_195: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_897, sum_320);  mul_897 = sum_320 = None
        sub_196: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_195, mul_899);  sub_195 = mul_899 = None
        mul_900: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_108, sub_196);  div_108 = sub_196 = None
        mul_901: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_290, mul_74);  mul_74 = None
        sum_322: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_901, [0, 1]);  mul_901 = None
        sum_323: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_290, [0, 1]);  add_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_82: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_902: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_82, 1.1111111111111112);  convert_element_type_82 = None
        mul_903: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_900, mul_902);  mul_902 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_1073: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_903, [4096, 1536]);  mul_903 = None
        permute_54: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
        permute_901: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_232: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_1073, permute_901);  permute_901 = None
        permute_902: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1073, [1, 0])
        mm_233: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_902, view_108);  permute_902 = view_108 = None
        sum_324: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1073, [0], True);  view_1073 = None
        view_1074: "f32[1536]" = torch.ops.aten.reshape.default(sum_324, [1536]);  sum_324 = None
        view_1075: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_232, [8, 512, 6144]);  mm_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_107: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_28, [8, 512, 6144]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_70: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
        erf_4: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_70);  mul_70 = None
        add_34: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_905: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
        mul_906: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_107, view_107)
        mul_907: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_906, -0.5);  mul_906 = None
        exp_46: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_907);  mul_907 = None
        mul_908: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
        mul_909: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_107, mul_908);  view_107 = mul_908 = None
        add_292: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_905, mul_909);  mul_905 = mul_909 = None
        mul_910: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_1075, add_292);  view_1075 = add_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_1076: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_910, [4096, 6144]);  mul_910 = None
        permute_53: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
        permute_905: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_234: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1076, permute_905);  permute_905 = None
        permute_906: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_1076, [1, 0])
        mm_235: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_906, view_106);  permute_906 = view_106 = None
        sum_325: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1076, [0], True);  view_1076 = None
        view_1077: "f32[6144]" = torch.ops.aten.reshape.default(sum_325, [6144]);  sum_325 = None
        view_1078: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_234, [8, 512, 1536]);  mm_234 = None
        add_293: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_900, view_1078);  mul_900 = view_1078 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_912: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_293, primals_79);  primals_79 = None
        mul_913: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_912, 1536)
        sum_326: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_912, [2], True)
        mul_914: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_912, mul_67);  mul_912 = None
        sum_327: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_914, [2], True);  mul_914 = None
        mul_915: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_67, sum_327);  sum_327 = None
        sub_198: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_913, sum_326);  mul_913 = sum_326 = None
        sub_199: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_198, mul_915);  sub_198 = mul_915 = None
        mul_916: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_109, sub_199);  div_109 = sub_199 = None
        mul_917: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_293, mul_67);  mul_67 = None
        sum_328: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_917, [0, 1]);  mul_917 = None
        sum_329: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_293, [0, 1]);  add_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_83: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_918: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_83, 1.1111111111111112);  convert_element_type_83 = None
        mul_919: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_916, mul_918);  mul_918 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_1079: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_919, [4096, 1536]);  mul_919 = None
        permute_52: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        permute_909: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_236: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1079, permute_909);  permute_909 = None
        permute_910: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1079, [1, 0])
        mm_237: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_910, view_104);  permute_910 = view_104 = None
        sum_330: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1079, [0], True);  view_1079 = None
        view_1080: "f32[1536]" = torch.ops.aten.reshape.default(sum_330, [1536]);  sum_330 = None
        view_1081: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_236, [8, 512, 1536]);  mm_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1082: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_1081, [8, 512, 24, 64]);  view_1081 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_913: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_1082, [0, 2, 1, 3]);  view_1082 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_231: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_913, memory_format = torch.contiguous_format);  permute_913 = None
        view_1083: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_231, [192, 512, 64]);  clone_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_124: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_914, view_1083);  permute_914 = None
        bmm_125: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_1083, permute_915);  view_1083 = permute_915 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_1084: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_125, [8, 24, 512, 512]);  bmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_84: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_13, torch.float32);  gt_13 = None
        mul_920: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_84, 1.1111111111111112);  convert_element_type_84 = None
        mul_921: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_1084, mul_920);  view_1084 = mul_920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_922: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_921, div_9);  mul_921 = None
        sum_331: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_922, [-1], True)
        neg_20: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_9);  div_9 = None
        fma_19: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_20, sum_331, mul_922);  neg_20 = sum_331 = mul_922 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_47: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_19);  fma_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_1085: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_47, [192, 512, 512]);  where_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_126: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_916, view_1085);  permute_916 = None
        bmm_127: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_1085, permute_917);  view_1085 = permute_917 = None
        div_110: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_126, full_default_1);  bmm_126 = None
        permute_918: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_110, [0, 2, 1]);  div_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1086: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_124, [8, 24, 512, 64]);  bmm_124 = None
        permute_919: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1086, [0, 2, 1, 3]);  view_1086 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_233: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_919, memory_format = torch.contiguous_format);  permute_919 = None
        view_1087: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_233, [8, 512, 1536]);  clone_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_1088: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_1087, [4096, 1536]);  view_1087 = None
        permute_48: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        permute_920: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_238: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1088, permute_920);  permute_920 = None
        permute_921: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1088, [1, 0])
        mm_239: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_921, view_88);  permute_921 = None
        sum_332: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1088, [0], True);  view_1088 = None
        view_1089: "f32[1536]" = torch.ops.aten.reshape.default(sum_332, [1536]);  sum_332 = None
        view_1090: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_238, [8, 512, 1536]);  mm_238 = None
        add_294: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_916, view_1090);  mul_916 = view_1090 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1091: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_918, [8, 24, 512, 64]);  permute_918 = None
        permute_924: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1091, [0, 2, 1, 3]);  view_1091 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_1092: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_924, [8, 512, 1536]);  permute_924 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_234: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_1092, memory_format = torch.contiguous_format);  view_1092 = None
        view_1093: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_234, [4096, 1536]);  clone_234 = None
        permute_46: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        permute_925: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_240: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1093, permute_925);  permute_925 = None
        permute_926: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1093, [1, 0])
        mm_241: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_926, view_88);  permute_926 = None
        sum_333: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1093, [0], True);  view_1093 = None
        view_1094: "f32[1536]" = torch.ops.aten.reshape.default(sum_333, [1536]);  sum_333 = None
        view_1095: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_240, [8, 512, 1536]);  mm_240 = None
        add_295: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_294, view_1095);  add_294 = view_1095 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1096: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_127, [8, 24, 512, 64]);  bmm_127 = None
        permute_929: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1096, [0, 2, 1, 3]);  view_1096 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_235: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_929, memory_format = torch.contiguous_format);  permute_929 = None
        view_1097: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_235, [8, 512, 1536]);  clone_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_1098: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_1097, [4096, 1536]);  view_1097 = None
        permute_44: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
        permute_930: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_242: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1098, permute_930);  permute_930 = None
        permute_931: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1098, [1, 0])
        mm_243: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_931, view_88);  permute_931 = view_88 = None
        sum_334: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1098, [0], True);  view_1098 = None
        view_1099: "f32[1536]" = torch.ops.aten.reshape.default(sum_334, [1536]);  sum_334 = None
        view_1100: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_242, [8, 512, 1536]);  mm_242 = None
        add_296: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_295, view_1100);  add_295 = view_1100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_924: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_296, primals_69);  primals_69 = None
        mul_925: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_924, 1536)
        sum_335: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_924, [2], True)
        mul_926: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_924, mul_60);  mul_924 = None
        sum_336: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_926, [2], True);  mul_926 = None
        mul_927: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_60, sum_336);  sum_336 = None
        sub_201: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_925, sum_335);  mul_925 = sum_335 = None
        sub_202: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_201, mul_927);  sub_201 = mul_927 = None
        mul_928: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_111, sub_202);  div_111 = sub_202 = None
        mul_929: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_296, mul_60);  mul_60 = None
        sum_337: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_929, [0, 1]);  mul_929 = None
        sum_338: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_296, [0, 1]);  add_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_85: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_930: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_85, 1.1111111111111112);  convert_element_type_85 = None
        mul_931: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_928, mul_930);  mul_930 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_1101: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_931, [4096, 1536]);  mul_931 = None
        permute_43: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        permute_934: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_244: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_1101, permute_934);  permute_934 = None
        permute_935: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1101, [1, 0])
        mm_245: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_935, view_86);  permute_935 = view_86 = None
        sum_339: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1101, [0], True);  view_1101 = None
        view_1102: "f32[1536]" = torch.ops.aten.reshape.default(sum_339, [1536]);  sum_339 = None
        view_1103: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_244, [8, 512, 6144]);  mm_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_85: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_22, [8, 512, 6144]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_56: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
        erf_3: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
        add_27: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_933: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_27, 0.5);  add_27 = None
        mul_934: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_85, view_85)
        mul_935: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_934, -0.5);  mul_934 = None
        exp_47: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_935);  mul_935 = None
        mul_936: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
        mul_937: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_85, mul_936);  view_85 = mul_936 = None
        add_298: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_933, mul_937);  mul_933 = mul_937 = None
        mul_938: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_1103, add_298);  view_1103 = add_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_1104: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_938, [4096, 6144]);  mul_938 = None
        permute_42: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
        permute_938: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_246: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1104, permute_938);  permute_938 = None
        permute_939: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_1104, [1, 0])
        mm_247: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_939, view_84);  permute_939 = view_84 = None
        sum_340: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1104, [0], True);  view_1104 = None
        view_1105: "f32[6144]" = torch.ops.aten.reshape.default(sum_340, [6144]);  sum_340 = None
        view_1106: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_246, [8, 512, 1536]);  mm_246 = None
        add_299: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_928, view_1106);  mul_928 = view_1106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_940: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_299, primals_63);  primals_63 = None
        mul_941: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_940, 1536)
        sum_341: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_940, [2], True)
        mul_942: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_940, mul_53);  mul_940 = None
        sum_342: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_942, [2], True);  mul_942 = None
        mul_943: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_53, sum_342);  sum_342 = None
        sub_204: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_941, sum_341);  mul_941 = sum_341 = None
        sub_205: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_204, mul_943);  sub_204 = mul_943 = None
        mul_944: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_112, sub_205);  div_112 = sub_205 = None
        mul_945: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_299, mul_53);  mul_53 = None
        sum_343: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_945, [0, 1]);  mul_945 = None
        sum_344: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_299, [0, 1]);  add_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_86: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_946: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_86, 1.1111111111111112);  convert_element_type_86 = None
        mul_947: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_944, mul_946);  mul_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_1107: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_947, [4096, 1536]);  mul_947 = None
        permute_41: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        permute_942: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_248: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1107, permute_942);  permute_942 = None
        permute_943: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1107, [1, 0])
        mm_249: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_943, view_82);  permute_943 = view_82 = None
        sum_345: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1107, [0], True);  view_1107 = None
        view_1108: "f32[1536]" = torch.ops.aten.reshape.default(sum_345, [1536]);  sum_345 = None
        view_1109: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_248, [8, 512, 1536]);  mm_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1110: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_1109, [8, 512, 24, 64]);  view_1109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_946: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_1110, [0, 2, 1, 3]);  view_1110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_238: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_946, memory_format = torch.contiguous_format);  permute_946 = None
        view_1111: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_238, [192, 512, 64]);  clone_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_128: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_947, view_1111);  permute_947 = None
        bmm_129: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_1111, permute_948);  view_1111 = permute_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_1112: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_129, [8, 24, 512, 512]);  bmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_87: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_948: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_87, 1.1111111111111112);  convert_element_type_87 = None
        mul_949: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_1112, mul_948);  view_1112 = mul_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_950: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_949, div_7);  mul_949 = None
        sum_346: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_950, [-1], True)
        neg_21: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_20: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_21, sum_346, mul_950);  neg_21 = sum_346 = mul_950 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_48: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_20);  fma_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_1113: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_48, [192, 512, 512]);  where_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_130: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_949, view_1113);  permute_949 = None
        bmm_131: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_1113, permute_950);  view_1113 = permute_950 = None
        div_113: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_130, full_default_1);  bmm_130 = None
        permute_951: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_113, [0, 2, 1]);  div_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1114: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_128, [8, 24, 512, 64]);  bmm_128 = None
        permute_952: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1114, [0, 2, 1, 3]);  view_1114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_240: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_952, memory_format = torch.contiguous_format);  permute_952 = None
        view_1115: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_240, [8, 512, 1536]);  clone_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_1116: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_1115, [4096, 1536]);  view_1115 = None
        permute_37: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        permute_953: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        mm_250: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1116, permute_953);  permute_953 = None
        permute_954: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1116, [1, 0])
        mm_251: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_954, view_66);  permute_954 = None
        sum_347: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1116, [0], True);  view_1116 = None
        view_1117: "f32[1536]" = torch.ops.aten.reshape.default(sum_347, [1536]);  sum_347 = None
        view_1118: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_250, [8, 512, 1536]);  mm_250 = None
        add_300: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_944, view_1118);  mul_944 = view_1118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1119: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_951, [8, 24, 512, 64]);  permute_951 = None
        permute_957: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1119, [0, 2, 1, 3]);  view_1119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_1120: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_957, [8, 512, 1536]);  permute_957 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_241: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_1120, memory_format = torch.contiguous_format);  view_1120 = None
        view_1121: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_241, [4096, 1536]);  clone_241 = None
        permute_35: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        permute_958: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_252: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1121, permute_958);  permute_958 = None
        permute_959: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1121, [1, 0])
        mm_253: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_959, view_66);  permute_959 = None
        sum_348: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1121, [0], True);  view_1121 = None
        view_1122: "f32[1536]" = torch.ops.aten.reshape.default(sum_348, [1536]);  sum_348 = None
        view_1123: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_252, [8, 512, 1536]);  mm_252 = None
        add_301: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_300, view_1123);  add_300 = view_1123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1124: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_131, [8, 24, 512, 64]);  bmm_131 = None
        permute_962: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1124, [0, 2, 1, 3]);  view_1124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_242: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_962, memory_format = torch.contiguous_format);  permute_962 = None
        view_1125: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_242, [8, 512, 1536]);  clone_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_1126: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_1125, [4096, 1536]);  view_1125 = None
        permute_33: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        permute_963: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_254: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1126, permute_963);  permute_963 = None
        permute_964: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1126, [1, 0])
        mm_255: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_964, view_66);  permute_964 = view_66 = None
        sum_349: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1126, [0], True);  view_1126 = None
        view_1127: "f32[1536]" = torch.ops.aten.reshape.default(sum_349, [1536]);  sum_349 = None
        view_1128: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_254, [8, 512, 1536]);  mm_254 = None
        add_302: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_301, view_1128);  add_301 = view_1128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_952: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_302, primals_53);  primals_53 = None
        mul_953: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_952, 1536)
        sum_350: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_952, [2], True)
        mul_954: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_952, mul_46);  mul_952 = None
        sum_351: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_954, [2], True);  mul_954 = None
        mul_955: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_46, sum_351);  sum_351 = None
        sub_207: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_953, sum_350);  mul_953 = sum_350 = None
        sub_208: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_207, mul_955);  sub_207 = mul_955 = None
        mul_956: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_114, sub_208);  div_114 = sub_208 = None
        mul_957: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_302, mul_46);  mul_46 = None
        sum_352: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_957, [0, 1]);  mul_957 = None
        sum_353: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_302, [0, 1]);  add_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_88: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_958: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_88, 1.1111111111111112);  convert_element_type_88 = None
        mul_959: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_956, mul_958);  mul_958 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_1129: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_959, [4096, 1536]);  mul_959 = None
        permute_32: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        permute_967: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_256: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_1129, permute_967);  permute_967 = None
        permute_968: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1129, [1, 0])
        mm_257: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_968, view_64);  permute_968 = view_64 = None
        sum_354: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1129, [0], True);  view_1129 = None
        view_1130: "f32[1536]" = torch.ops.aten.reshape.default(sum_354, [1536]);  sum_354 = None
        view_1131: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_256, [8, 512, 6144]);  mm_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_63: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_16, [8, 512, 6144]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_42: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
        erf_2: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
        add_20: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_961: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_20, 0.5);  add_20 = None
        mul_962: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_63, view_63)
        mul_963: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_962, -0.5);  mul_962 = None
        exp_48: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_963);  mul_963 = None
        mul_964: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
        mul_965: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_63, mul_964);  view_63 = mul_964 = None
        add_304: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_961, mul_965);  mul_961 = mul_965 = None
        mul_966: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_1131, add_304);  view_1131 = add_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_1132: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_966, [4096, 6144]);  mul_966 = None
        permute_31: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        permute_971: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_258: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1132, permute_971);  permute_971 = None
        permute_972: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_1132, [1, 0])
        mm_259: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_972, view_62);  permute_972 = view_62 = None
        sum_355: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1132, [0], True);  view_1132 = None
        view_1133: "f32[6144]" = torch.ops.aten.reshape.default(sum_355, [6144]);  sum_355 = None
        view_1134: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_258, [8, 512, 1536]);  mm_258 = None
        add_305: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_956, view_1134);  mul_956 = view_1134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_968: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_305, primals_47);  primals_47 = None
        mul_969: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_968, 1536)
        sum_356: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_968, [2], True)
        mul_970: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_968, mul_39);  mul_968 = None
        sum_357: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_970, [2], True);  mul_970 = None
        mul_971: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_39, sum_357);  sum_357 = None
        sub_210: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_969, sum_356);  mul_969 = sum_356 = None
        sub_211: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_210, mul_971);  sub_210 = mul_971 = None
        mul_972: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_115, sub_211);  div_115 = sub_211 = None
        mul_973: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_305, mul_39);  mul_39 = None
        sum_358: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_973, [0, 1]);  mul_973 = None
        sum_359: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_305, [0, 1]);  add_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_89: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_974: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_89, 1.1111111111111112);  convert_element_type_89 = None
        mul_975: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_972, mul_974);  mul_974 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_1135: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_975, [4096, 1536]);  mul_975 = None
        permute_30: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        permute_975: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_260: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1135, permute_975);  permute_975 = None
        permute_976: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1135, [1, 0])
        mm_261: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_976, view_60);  permute_976 = view_60 = None
        sum_360: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1135, [0], True);  view_1135 = None
        view_1136: "f32[1536]" = torch.ops.aten.reshape.default(sum_360, [1536]);  sum_360 = None
        view_1137: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_260, [8, 512, 1536]);  mm_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1138: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_1137, [8, 512, 24, 64]);  view_1137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_979: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_1138, [0, 2, 1, 3]);  view_1138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_245: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_979, memory_format = torch.contiguous_format);  permute_979 = None
        view_1139: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_245, [192, 512, 64]);  clone_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_132: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_980, view_1139);  permute_980 = None
        bmm_133: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_1139, permute_981);  view_1139 = permute_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_1140: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_133, [8, 24, 512, 512]);  bmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_90: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_976: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_90, 1.1111111111111112);  convert_element_type_90 = None
        mul_977: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_1140, mul_976);  view_1140 = mul_976 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_978: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_977, div_5);  mul_977 = None
        sum_361: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_978, [-1], True)
        neg_22: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_21: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_22, sum_361, mul_978);  neg_22 = sum_361 = mul_978 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_49: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_21);  fma_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_1141: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_49, [192, 512, 512]);  where_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_134: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_982, view_1141);  permute_982 = None
        bmm_135: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_1141, permute_983);  view_1141 = permute_983 = None
        div_116: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_134, full_default_1);  bmm_134 = None
        permute_984: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_116, [0, 2, 1]);  div_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1142: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_132, [8, 24, 512, 64]);  bmm_132 = None
        permute_985: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1142, [0, 2, 1, 3]);  view_1142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_247: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_985, memory_format = torch.contiguous_format);  permute_985 = None
        view_1143: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_247, [8, 512, 1536]);  clone_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_1144: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_1143, [4096, 1536]);  view_1143 = None
        permute_26: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_986: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        mm_262: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1144, permute_986);  permute_986 = None
        permute_987: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1144, [1, 0])
        mm_263: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_987, view_44);  permute_987 = None
        sum_362: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1144, [0], True);  view_1144 = None
        view_1145: "f32[1536]" = torch.ops.aten.reshape.default(sum_362, [1536]);  sum_362 = None
        view_1146: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_262, [8, 512, 1536]);  mm_262 = None
        add_306: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_972, view_1146);  mul_972 = view_1146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1147: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_984, [8, 24, 512, 64]);  permute_984 = None
        permute_990: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1147, [0, 2, 1, 3]);  view_1147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_1148: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_990, [8, 512, 1536]);  permute_990 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_248: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_1148, memory_format = torch.contiguous_format);  view_1148 = None
        view_1149: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_248, [4096, 1536]);  clone_248 = None
        permute_24: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
        permute_991: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_264: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1149, permute_991);  permute_991 = None
        permute_992: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1149, [1, 0])
        mm_265: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_992, view_44);  permute_992 = None
        sum_363: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1149, [0], True);  view_1149 = None
        view_1150: "f32[1536]" = torch.ops.aten.reshape.default(sum_363, [1536]);  sum_363 = None
        view_1151: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_264, [8, 512, 1536]);  mm_264 = None
        add_307: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_306, view_1151);  add_306 = view_1151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1152: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_135, [8, 24, 512, 64]);  bmm_135 = None
        permute_995: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1152, [0, 2, 1, 3]);  view_1152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_249: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_995, memory_format = torch.contiguous_format);  permute_995 = None
        view_1153: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_249, [8, 512, 1536]);  clone_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_1154: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_1153, [4096, 1536]);  view_1153 = None
        permute_22: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        permute_996: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_266: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1154, permute_996);  permute_996 = None
        permute_997: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1154, [1, 0])
        mm_267: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_997, view_44);  permute_997 = view_44 = None
        sum_364: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1154, [0], True);  view_1154 = None
        view_1155: "f32[1536]" = torch.ops.aten.reshape.default(sum_364, [1536]);  sum_364 = None
        view_1156: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_266, [8, 512, 1536]);  mm_266 = None
        add_308: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_307, view_1156);  add_307 = view_1156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_980: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_308, primals_37);  primals_37 = None
        mul_981: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_980, 1536)
        sum_365: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_980, [2], True)
        mul_982: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_980, mul_32);  mul_980 = None
        sum_366: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_982, [2], True);  mul_982 = None
        mul_983: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_32, sum_366);  sum_366 = None
        sub_213: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_981, sum_365);  mul_981 = sum_365 = None
        sub_214: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_213, mul_983);  sub_213 = mul_983 = None
        mul_984: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_117, sub_214);  div_117 = sub_214 = None
        mul_985: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_308, mul_32);  mul_32 = None
        sum_367: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_985, [0, 1]);  mul_985 = None
        sum_368: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_308, [0, 1]);  add_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_91: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_986: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_91, 1.1111111111111112);  convert_element_type_91 = None
        mul_987: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_984, mul_986);  mul_986 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_1157: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_987, [4096, 1536]);  mul_987 = None
        permute_21: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
        permute_1000: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_268: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_1157, permute_1000);  permute_1000 = None
        permute_1001: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1157, [1, 0])
        mm_269: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_1001, view_42);  permute_1001 = view_42 = None
        sum_369: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1157, [0], True);  view_1157 = None
        view_1158: "f32[1536]" = torch.ops.aten.reshape.default(sum_369, [1536]);  sum_369 = None
        view_1159: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_268, [8, 512, 6144]);  mm_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_41: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_10, [8, 512, 6144]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_28: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
        erf_1: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_13: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_989: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
        mul_990: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_41, view_41)
        mul_991: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_990, -0.5);  mul_990 = None
        exp_49: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_991);  mul_991 = None
        mul_992: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_49, 0.3989422804014327);  exp_49 = None
        mul_993: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_41, mul_992);  view_41 = mul_992 = None
        add_310: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_989, mul_993);  mul_989 = mul_993 = None
        mul_994: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_1159, add_310);  view_1159 = add_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_1160: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_994, [4096, 6144]);  mul_994 = None
        permute_20: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
        permute_1004: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_270: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1160, permute_1004);  permute_1004 = None
        permute_1005: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_1160, [1, 0])
        mm_271: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_1005, view_40);  permute_1005 = view_40 = None
        sum_370: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1160, [0], True);  view_1160 = None
        view_1161: "f32[6144]" = torch.ops.aten.reshape.default(sum_370, [6144]);  sum_370 = None
        view_1162: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_270, [8, 512, 1536]);  mm_270 = None
        add_311: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_984, view_1162);  mul_984 = view_1162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_996: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_311, primals_31);  primals_31 = None
        mul_997: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_996, 1536)
        sum_371: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_996, [2], True)
        mul_998: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_996, mul_25);  mul_996 = None
        sum_372: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_998, [2], True);  mul_998 = None
        mul_999: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_25, sum_372);  sum_372 = None
        sub_216: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_997, sum_371);  mul_997 = sum_371 = None
        sub_217: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_216, mul_999);  sub_216 = mul_999 = None
        mul_1000: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_118, sub_217);  div_118 = sub_217 = None
        mul_1001: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_311, mul_25);  mul_25 = None
        sum_373: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1001, [0, 1]);  mul_1001 = None
        sum_374: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_311, [0, 1]);  add_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_92: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_1002: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_92, 1.1111111111111112);  convert_element_type_92 = None
        mul_1003: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_1000, mul_1002);  mul_1002 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_1163: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_1003, [4096, 1536]);  mul_1003 = None
        permute_19: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_1008: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_272: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1163, permute_1008);  permute_1008 = None
        permute_1009: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1163, [1, 0])
        mm_273: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1009, view_38);  permute_1009 = view_38 = None
        sum_375: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1163, [0], True);  view_1163 = None
        view_1164: "f32[1536]" = torch.ops.aten.reshape.default(sum_375, [1536]);  sum_375 = None
        view_1165: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_272, [8, 512, 1536]);  mm_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1166: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_1165, [8, 512, 24, 64]);  view_1165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_1012: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_1166, [0, 2, 1, 3]);  view_1166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_252: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_1012, memory_format = torch.contiguous_format);  permute_1012 = None
        view_1167: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_252, [192, 512, 64]);  clone_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_136: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_1013, view_1167);  permute_1013 = None
        bmm_137: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_1167, permute_1014);  view_1167 = permute_1014 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_1168: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_137, [8, 24, 512, 512]);  bmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_93: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_1004: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_93, 1.1111111111111112);  convert_element_type_93 = None
        mul_1005: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_1168, mul_1004);  view_1168 = mul_1004 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        mul_1006: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_1005, div_3);  mul_1005 = None
        sum_376: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1006, [-1], True)
        neg_23: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_22: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_23, sum_376, mul_1006);  neg_23 = sum_376 = mul_1006 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_50: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_22);  fma_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_1169: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_50, [192, 512, 512]);  where_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_138: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_1015, view_1169);  permute_1015 = None
        bmm_139: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_1169, permute_1016);  view_1169 = permute_1016 = None
        div_119: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_138, full_default_1);  bmm_138 = None
        permute_1017: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_119, [0, 2, 1]);  div_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1170: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_136, [8, 24, 512, 64]);  bmm_136 = None
        permute_1018: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1170, [0, 2, 1, 3]);  view_1170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_254: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_1018, memory_format = torch.contiguous_format);  permute_1018 = None
        view_1171: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_254, [8, 512, 1536]);  clone_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_1172: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_1171, [4096, 1536]);  view_1171 = None
        permute_15: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_1019: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        mm_274: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1172, permute_1019);  permute_1019 = None
        permute_1020: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1172, [1, 0])
        mm_275: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1020, view_22);  permute_1020 = None
        sum_377: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1172, [0], True);  view_1172 = None
        view_1173: "f32[1536]" = torch.ops.aten.reshape.default(sum_377, [1536]);  sum_377 = None
        view_1174: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_274, [8, 512, 1536]);  mm_274 = None
        add_312: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_1000, view_1174);  mul_1000 = view_1174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1175: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_1017, [8, 24, 512, 64]);  permute_1017 = None
        permute_1023: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1175, [0, 2, 1, 3]);  view_1175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_1176: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_1023, [8, 512, 1536]);  permute_1023 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_255: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_1176, memory_format = torch.contiguous_format);  view_1176 = None
        view_1177: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_255, [4096, 1536]);  clone_255 = None
        permute_13: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_1024: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_276: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1177, permute_1024);  permute_1024 = None
        permute_1025: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1177, [1, 0])
        mm_277: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1025, view_22);  permute_1025 = None
        sum_378: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1177, [0], True);  view_1177 = None
        view_1178: "f32[1536]" = torch.ops.aten.reshape.default(sum_378, [1536]);  sum_378 = None
        view_1179: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_276, [8, 512, 1536]);  mm_276 = None
        add_313: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_312, view_1179);  add_312 = view_1179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1180: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_139, [8, 24, 512, 64]);  bmm_139 = None
        permute_1028: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1180, [0, 2, 1, 3]);  view_1180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_256: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_1028, memory_format = torch.contiguous_format);  permute_1028 = None
        view_1181: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_256, [8, 512, 1536]);  clone_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_1182: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_1181, [4096, 1536]);  view_1181 = None
        permute_11: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_1029: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_278: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1182, permute_1029);  permute_1029 = None
        permute_1030: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1182, [1, 0])
        mm_279: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1030, view_22);  permute_1030 = view_22 = None
        sum_379: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1182, [0], True);  view_1182 = None
        view_1183: "f32[1536]" = torch.ops.aten.reshape.default(sum_379, [1536]);  sum_379 = None
        view_1184: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_278, [8, 512, 1536]);  mm_278 = None
        add_314: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_313, view_1184);  add_313 = view_1184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_1008: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_314, primals_21);  primals_21 = None
        mul_1009: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_1008, 1536)
        sum_380: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1008, [2], True)
        mul_1010: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_1008, mul_18);  mul_1008 = None
        sum_381: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1010, [2], True);  mul_1010 = None
        mul_1011: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_18, sum_381);  sum_381 = None
        sub_219: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_1009, sum_380);  mul_1009 = sum_380 = None
        sub_220: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_219, mul_1011);  sub_219 = mul_1011 = None
        mul_1012: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_120, sub_220);  div_120 = sub_220 = None
        mul_1013: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_314, mul_18);  mul_18 = None
        sum_382: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1013, [0, 1]);  mul_1013 = None
        sum_383: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_314, [0, 1]);  add_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_94: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_1014: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 1.1111111111111112);  convert_element_type_94 = None
        mul_1015: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_1012, mul_1014);  mul_1014 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_1185: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_1015, [4096, 1536]);  mul_1015 = None
        permute_10: "f32[6144, 1536]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_1033: "f32[1536, 6144]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_280: "f32[4096, 6144]" = torch.ops.aten.mm.default(view_1185, permute_1033);  permute_1033 = None
        permute_1034: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1185, [1, 0])
        mm_281: "f32[1536, 6144]" = torch.ops.aten.mm.default(permute_1034, view_20);  permute_1034 = view_20 = None
        sum_384: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1185, [0], True);  view_1185 = None
        view_1186: "f32[1536]" = torch.ops.aten.reshape.default(sum_384, [1536]);  sum_384 = None
        view_1187: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(mm_280, [8, 512, 6144]);  mm_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_19: "f32[8, 512, 6144]" = torch.ops.aten.reshape.default(addmm_4, [8, 512, 6144]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_14: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
        erf: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_14);  mul_14 = None
        add_6: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_1017: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
        mul_1018: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_19, view_19)
        mul_1019: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_1018, -0.5);  mul_1018 = None
        exp_50: "f32[8, 512, 6144]" = torch.ops.aten.exp.default(mul_1019);  mul_1019 = None
        mul_1020: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(exp_50, 0.3989422804014327);  exp_50 = None
        mul_1021: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_19, mul_1020);  view_19 = mul_1020 = None
        add_316: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(mul_1017, mul_1021);  mul_1017 = mul_1021 = None
        mul_1022: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(view_1187, add_316);  view_1187 = add_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_1188: "f32[4096, 6144]" = torch.ops.aten.reshape.default(mul_1022, [4096, 6144]);  mul_1022 = None
        permute_9: "f32[1536, 6144]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_1037: "f32[6144, 1536]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_282: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1188, permute_1037);  permute_1037 = None
        permute_1038: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_1188, [1, 0])
        mm_283: "f32[6144, 1536]" = torch.ops.aten.mm.default(permute_1038, view_18);  permute_1038 = view_18 = None
        sum_385: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1188, [0], True);  view_1188 = None
        view_1189: "f32[6144]" = torch.ops.aten.reshape.default(sum_385, [6144]);  sum_385 = None
        view_1190: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_282, [8, 512, 1536]);  mm_282 = None
        add_317: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_1012, view_1190);  mul_1012 = view_1190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_1024: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_317, primals_15);  primals_15 = None
        mul_1025: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_1024, 1536)
        sum_386: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1024, [2], True)
        mul_1026: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_1024, mul_11);  mul_1024 = None
        sum_387: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1026, [2], True);  mul_1026 = None
        mul_1027: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_11, sum_387);  sum_387 = None
        sub_222: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_1025, sum_386);  mul_1025 = sum_386 = None
        sub_223: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_222, mul_1027);  sub_222 = mul_1027 = None
        mul_1028: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_121, sub_223);  div_121 = sub_223 = None
        mul_1029: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_317, mul_11);  mul_11 = None
        sum_388: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1029, [0, 1]);  mul_1029 = None
        sum_389: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_317, [0, 1]);  add_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_95: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_1030: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_95, 1.1111111111111112);  convert_element_type_95 = None
        mul_1031: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_1028, mul_1030);  mul_1030 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_1191: "f32[4096, 1536]" = torch.ops.aten.reshape.default(mul_1031, [4096, 1536]);  mul_1031 = None
        permute_8: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_1041: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_284: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1191, permute_1041);  permute_1041 = None
        permute_1042: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1191, [1, 0])
        mm_285: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1042, view_16);  permute_1042 = view_16 = None
        sum_390: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1191, [0], True);  view_1191 = None
        view_1192: "f32[1536]" = torch.ops.aten.reshape.default(sum_390, [1536]);  sum_390 = None
        view_1193: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_284, [8, 512, 1536]);  mm_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1194: "f32[8, 512, 24, 64]" = torch.ops.aten.reshape.default(view_1193, [8, 512, 24, 64]);  view_1193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_1045: "f32[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_1194, [0, 2, 1, 3]);  view_1194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_259: "f32[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_1045, memory_format = torch.contiguous_format);  permute_1045 = None
        view_1195: "f32[192, 512, 64]" = torch.ops.aten.reshape.default(clone_259, [192, 512, 64]);  clone_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_140: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(permute_1046, view_1195);  permute_1046 = None
        bmm_141: "f32[192, 512, 512]" = torch.ops.aten.bmm.default(view_1195, permute_1047);  view_1195 = permute_1047 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_1196: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm_141, [8, 24, 512, 512]);  bmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_96: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_1032: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_96, 1.1111111111111112);  convert_element_type_96 = None
        mul_1033: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(view_1196, mul_1032);  view_1196 = mul_1032 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_12: "f32[8, 24, 512, 512]" = torch.ops.aten.reshape.default(bmm, [-1, 24, 512, 512]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_3, view_12);  full_default_3 = view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        sub_1: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
        exp: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div_1: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_1034: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_1033, div_1);  mul_1033 = None
        sum_391: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1034, [-1], True)
        neg_24: "f32[8, 24, 512, 512]" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_23: "f32[8, 24, 512, 512]" = torch.ops.prims.fma.default(neg_24, sum_391, mul_1034);  neg_24 = sum_391 = mul_1034 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_51: "f32[8, 24, 512, 512]" = torch.ops.aten.where.self(full_default_2, full_default_74, fma_23);  full_default_2 = fma_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_1197: "f32[192, 512, 512]" = torch.ops.aten.reshape.default(where_51, [192, 512, 512]);  where_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_142: "f32[192, 64, 512]" = torch.ops.aten.bmm.default(permute_1048, view_1197);  permute_1048 = None
        bmm_143: "f32[192, 512, 64]" = torch.ops.aten.bmm.default(view_1197, permute_1049);  view_1197 = permute_1049 = None
        div_122: "f32[192, 64, 512]" = torch.ops.aten.div.Tensor(bmm_142, full_default_1);  bmm_142 = full_default_1 = None
        permute_1050: "f32[192, 512, 64]" = torch.ops.aten.permute.default(div_122, [0, 2, 1]);  div_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1198: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_140, [8, 24, 512, 64]);  bmm_140 = None
        permute_1051: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1198, [0, 2, 1, 3]);  view_1198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_261: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_1051, memory_format = torch.contiguous_format);  permute_1051 = None
        view_1199: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_261, [8, 512, 1536]);  clone_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_1200: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_1199, [4096, 1536]);  view_1199 = None
        permute_4: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_1052: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_286: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1200, permute_1052);  permute_1052 = None
        permute_1053: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1200, [1, 0])
        mm_287: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1053, view);  permute_1053 = None
        sum_392: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1200, [0], True);  view_1200 = None
        view_1201: "f32[1536]" = torch.ops.aten.reshape.default(sum_392, [1536]);  sum_392 = None
        view_1202: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_286, [8, 512, 1536]);  mm_286 = None
        add_318: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_1028, view_1202);  mul_1028 = view_1202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1203: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(permute_1050, [8, 24, 512, 64]);  permute_1050 = None
        permute_1056: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1203, [0, 2, 1, 3]);  view_1203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_1204: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(permute_1056, [8, 512, 1536]);  permute_1056 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_262: "f32[8, 512, 1536]" = torch.ops.aten.clone.default(view_1204, memory_format = torch.contiguous_format);  view_1204 = None
        view_1205: "f32[4096, 1536]" = torch.ops.aten.reshape.default(clone_262, [4096, 1536]);  clone_262 = None
        permute_2: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_1057: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_288: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1205, permute_1057);  permute_1057 = None
        permute_1058: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1205, [1, 0])
        mm_289: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1058, view);  permute_1058 = None
        sum_393: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1205, [0], True);  view_1205 = None
        view_1206: "f32[1536]" = torch.ops.aten.reshape.default(sum_393, [1536]);  sum_393 = None
        view_1207: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_288, [8, 512, 1536]);  mm_288 = None
        add_319: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_318, view_1207);  add_318 = view_1207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1208: "f32[8, 24, 512, 64]" = torch.ops.aten.reshape.default(bmm_143, [8, 24, 512, 64]);  bmm_143 = None
        permute_1061: "f32[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_1208, [0, 2, 1, 3]);  view_1208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_263: "f32[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_1061, memory_format = torch.contiguous_format);  permute_1061 = None
        view_1209: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(clone_263, [8, 512, 1536]);  clone_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_1210: "f32[4096, 1536]" = torch.ops.aten.reshape.default(view_1209, [4096, 1536]);  view_1209 = None
        permute: "f32[1536, 1536]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_1062: "f32[1536, 1536]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_290: "f32[4096, 1536]" = torch.ops.aten.mm.default(view_1210, permute_1062);  permute_1062 = None
        permute_1063: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1210, [1, 0])
        mm_291: "f32[1536, 1536]" = torch.ops.aten.mm.default(permute_1063, view);  permute_1063 = view = None
        sum_394: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True);  view_1210 = None
        view_1211: "f32[1536]" = torch.ops.aten.reshape.default(sum_394, [1536]);  sum_394 = None
        view_1212: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_290, [8, 512, 1536]);  mm_290 = None
        add_320: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_319, view_1212);  add_319 = view_1212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:563 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_97: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_1035: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_97, 1.1111111111111112);  convert_element_type_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:561 in forward, code: embeddings = embeddings * mask
        mul_1036: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_320, mul_1035);  add_320 = mul_1035 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:552 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_1039: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_1036, primals_5);  primals_5 = None
        mul_1040: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_1039, 1536)
        sum_395: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1039, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:544 in forward, code: embeddings = embeddings + position_embeddings
        add: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:552 in forward, code: embeddings = self.LayerNorm(embeddings)
        sub: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1041: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_1039, mul);  mul_1039 = None
        sum_396: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1041, [2], True);  mul_1041 = None
        mul_1042: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul, sum_396);  sum_396 = None
        sub_225: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_1040, sum_395);  mul_1040 = sum_395 = None
        sub_226: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_225, mul_1042);  sub_225 = mul_1042 = None
        div_123: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1536);  rsqrt = None
        mul_1043: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_123, sub_226);  div_123 = sub_226 = None
        mul_1044: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_1036, mul);  mul = None
        sum_397: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1044, [0, 1]);  mul_1044 = None
        sum_398: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1036, [0, 1]);  mul_1036 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:544 in forward, code: embeddings = embeddings + position_embeddings
        sum_399: "f32[1, 512, 1536]" = torch.ops.aten.sum.dim_IntList(mul_1043, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:538 in forward, code: position_embeddings = self.position_embeddings(position_ids.long())
        eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_2, -1)
        unsqueeze_6: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        where_52: "f32[1, 512, 1536]" = torch.ops.aten.where.self(unsqueeze_6, full_default_74, sum_399);  unsqueeze_6 = sum_399 = None
        full_default_103: "f32[512, 1536]" = torch.ops.aten.full.default([512, 1536], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[512, 1536]" = torch.ops.aten.index_put.default(full_default_103, [primals_2], where_52, True);  full_default_103 = primals_2 = where_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:535 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_1: "b8[8, 512]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_7: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_53: "f32[8, 512, 1536]" = torch.ops.aten.where.self(unsqueeze_7, full_default_74, mul_1043);  unsqueeze_7 = full_default_74 = mul_1043 = None
        full_default_105: "f32[128100, 1536]" = torch.ops.aten.full.default([128100, 1536], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[128100, 1536]" = torch.ops.aten.index_put.default(full_default_105, [primals_1], where_53, True);  full_default_105 = primals_1 = where_53 = None
        add_321: "f32[128100, 1536]" = torch.ops.aten.add.Tensor(mm_1, index_put_1);  mm_1 = index_put_1 = None
        return (None, None, add_321, index_put, sum_397, sum_398, mm_291, view_1211, mm_289, view_1206, mm_287, view_1201, mm_285, view_1192, sum_388, sum_389, mm_283, view_1189, mm_281, view_1186, sum_382, sum_383, mm_279, view_1183, mm_277, view_1178, mm_275, view_1173, mm_273, view_1164, sum_373, sum_374, mm_271, view_1161, mm_269, view_1158, sum_367, sum_368, mm_267, view_1155, mm_265, view_1150, mm_263, view_1145, mm_261, view_1136, sum_358, sum_359, mm_259, view_1133, mm_257, view_1130, sum_352, sum_353, mm_255, view_1127, mm_253, view_1122, mm_251, view_1117, mm_249, view_1108, sum_343, sum_344, mm_247, view_1105, mm_245, view_1102, sum_337, sum_338, mm_243, view_1099, mm_241, view_1094, mm_239, view_1089, mm_237, view_1080, sum_328, sum_329, mm_235, view_1077, mm_233, view_1074, sum_322, sum_323, mm_231, view_1071, mm_229, view_1066, mm_227, view_1061, mm_225, view_1052, sum_313, sum_314, mm_223, view_1049, mm_221, view_1046, sum_307, sum_308, mm_219, view_1043, mm_217, view_1038, mm_215, view_1033, mm_213, view_1024, sum_298, sum_299, mm_211, view_1021, mm_209, view_1018, sum_292, sum_293, mm_207, view_1015, mm_205, view_1010, mm_203, view_1005, mm_201, view_996, sum_283, sum_284, mm_199, view_993, mm_197, view_990, sum_277, sum_278, mm_195, view_987, mm_193, view_982, mm_191, view_977, mm_189, view_968, sum_268, sum_269, mm_187, view_965, mm_185, view_962, sum_262, sum_263, mm_183, view_959, mm_181, view_954, mm_179, view_949, mm_177, view_940, sum_253, sum_254, mm_175, view_937, mm_173, view_934, sum_247, sum_248, mm_171, view_931, mm_169, view_926, mm_167, view_921, mm_165, view_912, sum_238, sum_239, mm_163, view_909, mm_161, view_906, sum_232, sum_233, mm_159, view_903, mm_157, view_898, mm_155, view_893, mm_153, view_884, sum_223, sum_224, mm_151, view_881, mm_149, view_878, sum_217, sum_218, mm_147, view_875, mm_145, view_870, mm_143, view_865, mm_141, view_856, sum_208, sum_209, mm_139, view_853, mm_137, view_850, sum_202, sum_203, mm_135, view_847, mm_133, view_842, mm_131, view_837, mm_129, view_828, sum_193, sum_194, mm_127, view_825, mm_125, view_822, sum_187, sum_188, mm_123, view_819, mm_121, view_814, mm_119, view_809, mm_117, view_800, sum_178, sum_179, mm_115, view_797, mm_113, view_794, sum_172, sum_173, mm_111, view_791, mm_109, view_786, mm_107, view_781, mm_105, view_772, sum_163, sum_164, mm_103, view_769, mm_101, view_766, sum_157, sum_158, mm_99, view_763, mm_97, view_758, mm_95, view_753, mm_93, view_744, sum_148, sum_149, mm_91, view_741, mm_89, view_738, sum_142, sum_143, mm_87, view_735, mm_85, view_730, mm_83, view_725, mm_81, view_716, sum_133, sum_134, mm_79, view_713, mm_77, view_710, sum_127, sum_128, mm_75, view_707, mm_73, view_702, mm_71, view_697, mm_69, view_688, sum_118, sum_119, mm_67, view_685, mm_65, view_682, sum_112, sum_113, mm_63, view_679, mm_61, view_674, mm_59, view_669, mm_57, view_660, sum_103, sum_104, mm_55, view_657, mm_53, view_654, sum_97, sum_98, mm_51, view_651, mm_49, view_646, mm_47, view_641, mm_45, view_632, sum_88, sum_89, mm_43, view_629, mm_41, view_626, sum_82, sum_83, mm_39, view_623, mm_37, view_618, mm_35, view_613, mm_33, view_604, sum_73, sum_74, mm_31, view_601, mm_29, view_598, sum_67, sum_68, mm_27, view_595, mm_25, view_590, mm_23, view_585, mm_21, view_576, sum_58, sum_59, mm_19, view_573, mm_17, view_570, sum_52, sum_53, mm_15, view_567, mm_13, view_562, mm_11, view_557, mm_9, view_548, sum_43, sum_44, mm_7, view_545, mm_5, view_542, sum_37, sum_38, mm_3, view_539, sum_32, sum_33, view_536, None)
