class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[1, 128][128, 1]cuda:0", primals_3: "f32[4096][1]cuda:0", primals_14: "f32[4096][1]cuda:0", primals_25: "f32[4096][1]cuda:0", primals_36: "f32[4096][1]cuda:0", primals_47: "f32[4096][1]cuda:0", primals_58: "f32[4096][1]cuda:0", primals_69: "f32[4096][1]cuda:0", primals_80: "f32[4096][1]cuda:0", primals_91: "f32[4096][1]cuda:0", primals_102: "f32[4096][1]cuda:0", primals_113: "f32[4096][1]cuda:0", primals_124: "f32[4096][1]cuda:0", primals_135: "f32[4096][1]cuda:0", primals_146: "f32[4096][1]cuda:0", primals_157: "f32[4096][1]cuda:0", primals_168: "f32[4096][1]cuda:0", primals_179: "f32[4096][1]cuda:0", primals_190: "f32[4096][1]cuda:0", primals_201: "f32[4096][1]cuda:0", primals_212: "f32[4096][1]cuda:0", primals_223: "f32[4096][1]cuda:0", primals_234: "f32[4096][1]cuda:0", primals_245: "f32[4096][1]cuda:0", primals_256: "f32[4096][1]cuda:0", primals_267: "f32[4096][1]cuda:0", primals_278: "f32[4096][1]cuda:0", primals_289: "f32[4096][1]cuda:0", primals_300: "f32[4096][1]cuda:0", primals_311: "f32[4096][1]cuda:0", embedding: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", getitem_1: "f32[1, 128, 1][128, 1, 1]cuda:0", rsqrt: "f32[1, 128, 1][128, 1, 1]cuda:0", view: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_12: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_14: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_1: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_22: "bf16[128, 4096][4096, 1]cuda:0", addmm: "bf16[128, 16384][16384, 1]cuda:0", view_26: "bf16[128, 16384][16384, 1]cuda:0", add_10: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0", getitem_5: "f32[1, 128, 1][128, 1, 1]cuda:0", rsqrt_1: "f32[1, 128, 1][128, 1, 1]cuda:0", view_28: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_25: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_27: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_3: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_50: "bf16[128, 4096][4096, 1]cuda:0", addmm_2: "bf16[128, 16384][16384, 1]cuda:0", view_54: "bf16[128, 16384][16384, 1]cuda:0", mul_20: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_56: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_38: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_40: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_5: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_78: "bf16[128, 4096][4096, 1]cuda:0", addmm_4: "bf16[128, 16384][16384, 1]cuda:0", view_82: "bf16[128, 16384][16384, 1]cuda:0", mul_30: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_84: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_51: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_53: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_7: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_106: "bf16[128, 4096][4096, 1]cuda:0", addmm_6: "bf16[128, 16384][16384, 1]cuda:0", view_110: "bf16[128, 16384][16384, 1]cuda:0", mul_40: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_112: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_64: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_66: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_9: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_134: "bf16[128, 4096][4096, 1]cuda:0", addmm_8: "bf16[128, 16384][16384, 1]cuda:0", view_138: "bf16[128, 16384][16384, 1]cuda:0", mul_50: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_140: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_77: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_79: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_11: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_162: "bf16[128, 4096][4096, 1]cuda:0", addmm_10: "bf16[128, 16384][16384, 1]cuda:0", view_166: "bf16[128, 16384][16384, 1]cuda:0", mul_60: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_168: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_90: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_92: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_13: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_190: "bf16[128, 4096][4096, 1]cuda:0", addmm_12: "bf16[128, 16384][16384, 1]cuda:0", view_194: "bf16[128, 16384][16384, 1]cuda:0", mul_70: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_196: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_103: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_105: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_15: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_218: "bf16[128, 4096][4096, 1]cuda:0", addmm_14: "bf16[128, 16384][16384, 1]cuda:0", view_222: "bf16[128, 16384][16384, 1]cuda:0", mul_80: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_224: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_116: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_118: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_17: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_246: "bf16[128, 4096][4096, 1]cuda:0", addmm_16: "bf16[128, 16384][16384, 1]cuda:0", view_250: "bf16[128, 16384][16384, 1]cuda:0", mul_90: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_252: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_129: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_131: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_19: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_274: "bf16[128, 4096][4096, 1]cuda:0", addmm_18: "bf16[128, 16384][16384, 1]cuda:0", view_278: "bf16[128, 16384][16384, 1]cuda:0", mul_100: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_280: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_142: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_144: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_21: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_302: "bf16[128, 4096][4096, 1]cuda:0", addmm_20: "bf16[128, 16384][16384, 1]cuda:0", view_306: "bf16[128, 16384][16384, 1]cuda:0", mul_110: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_308: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_155: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_157: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_23: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_330: "bf16[128, 4096][4096, 1]cuda:0", addmm_22: "bf16[128, 16384][16384, 1]cuda:0", view_334: "bf16[128, 16384][16384, 1]cuda:0", mul_120: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_336: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_168: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_170: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_25: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_358: "bf16[128, 4096][4096, 1]cuda:0", addmm_24: "bf16[128, 16384][16384, 1]cuda:0", view_362: "bf16[128, 16384][16384, 1]cuda:0", mul_130: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_364: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_181: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_183: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_27: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_386: "bf16[128, 4096][4096, 1]cuda:0", addmm_26: "bf16[128, 16384][16384, 1]cuda:0", view_390: "bf16[128, 16384][16384, 1]cuda:0", mul_140: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_392: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_194: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_196: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_29: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_414: "bf16[128, 4096][4096, 1]cuda:0", addmm_28: "bf16[128, 16384][16384, 1]cuda:0", view_418: "bf16[128, 16384][16384, 1]cuda:0", mul_150: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_420: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_207: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_209: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_31: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_442: "bf16[128, 4096][4096, 1]cuda:0", addmm_30: "bf16[128, 16384][16384, 1]cuda:0", view_446: "bf16[128, 16384][16384, 1]cuda:0", mul_160: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_448: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_220: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_222: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_33: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_470: "bf16[128, 4096][4096, 1]cuda:0", addmm_32: "bf16[128, 16384][16384, 1]cuda:0", view_474: "bf16[128, 16384][16384, 1]cuda:0", mul_170: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_476: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_233: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_235: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_35: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_498: "bf16[128, 4096][4096, 1]cuda:0", addmm_34: "bf16[128, 16384][16384, 1]cuda:0", view_502: "bf16[128, 16384][16384, 1]cuda:0", mul_180: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_504: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_246: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_248: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_37: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_526: "bf16[128, 4096][4096, 1]cuda:0", addmm_36: "bf16[128, 16384][16384, 1]cuda:0", view_530: "bf16[128, 16384][16384, 1]cuda:0", mul_190: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_532: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_259: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_261: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_39: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_554: "bf16[128, 4096][4096, 1]cuda:0", addmm_38: "bf16[128, 16384][16384, 1]cuda:0", view_558: "bf16[128, 16384][16384, 1]cuda:0", mul_200: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_560: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_272: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_274: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_41: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_582: "bf16[128, 4096][4096, 1]cuda:0", addmm_40: "bf16[128, 16384][16384, 1]cuda:0", view_586: "bf16[128, 16384][16384, 1]cuda:0", mul_210: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_588: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_285: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_287: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_43: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_610: "bf16[128, 4096][4096, 1]cuda:0", addmm_42: "bf16[128, 16384][16384, 1]cuda:0", view_614: "bf16[128, 16384][16384, 1]cuda:0", mul_220: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_616: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_298: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_300: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_45: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_638: "bf16[128, 4096][4096, 1]cuda:0", addmm_44: "bf16[128, 16384][16384, 1]cuda:0", view_642: "bf16[128, 16384][16384, 1]cuda:0", mul_230: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_644: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_311: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_313: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_47: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_666: "bf16[128, 4096][4096, 1]cuda:0", addmm_46: "bf16[128, 16384][16384, 1]cuda:0", view_670: "bf16[128, 16384][16384, 1]cuda:0", mul_240: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_672: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_324: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_326: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_49: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_694: "bf16[128, 4096][4096, 1]cuda:0", addmm_48: "bf16[128, 16384][16384, 1]cuda:0", view_698: "bf16[128, 16384][16384, 1]cuda:0", mul_250: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_700: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_337: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_339: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_51: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_722: "bf16[128, 4096][4096, 1]cuda:0", addmm_50: "bf16[128, 16384][16384, 1]cuda:0", view_726: "bf16[128, 16384][16384, 1]cuda:0", mul_260: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_728: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_350: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_352: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_53: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_750: "bf16[128, 4096][4096, 1]cuda:0", addmm_52: "bf16[128, 16384][16384, 1]cuda:0", view_754: "bf16[128, 16384][16384, 1]cuda:0", mul_270: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_756: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_363: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_365: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_55: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_778: "bf16[128, 4096][4096, 1]cuda:0", addmm_54: "bf16[128, 16384][16384, 1]cuda:0", view_782: "bf16[128, 16384][16384, 1]cuda:0", mul_280: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_785: "bf16[128, 4096][4096, 1]cuda:0", view_786: "bf16[1, 128, 50400][6451200, 50400, 1]cuda:0", constant_pad_nd: "i64[1, 129][129, 1]cuda:0", amax_28: "f32[128, 1][1, 1]cuda:0", log: "f32[128, 1][1, 1]cuda:0", convert_element_type_1071: "f32[][]cuda:0", permute_309: "bf16[50400, 4096][4096, 1]cuda:0", div_58: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_313: "bf16[4096, 16384][16384, 1]cuda:0", permute_317: "bf16[16384, 4096][4096, 1]cuda:0", permute_323: "bf16[4096, 4096][4096, 1]cuda:0", permute_326: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_327: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_328: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_329: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_336: "bf16[4096, 4096][4096, 1]cuda:0", permute_340: "bf16[4096, 4096][4096, 1]cuda:0", permute_344: "bf16[4096, 4096][4096, 1]cuda:0", div_60: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_346: "bf16[4096, 16384][16384, 1]cuda:0", permute_350: "bf16[16384, 4096][4096, 1]cuda:0", permute_356: "bf16[4096, 4096][4096, 1]cuda:0", permute_359: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_360: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_361: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_362: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_369: "bf16[4096, 4096][4096, 1]cuda:0", permute_373: "bf16[4096, 4096][4096, 1]cuda:0", permute_377: "bf16[4096, 4096][4096, 1]cuda:0", div_62: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_379: "bf16[4096, 16384][16384, 1]cuda:0", permute_383: "bf16[16384, 4096][4096, 1]cuda:0", permute_389: "bf16[4096, 4096][4096, 1]cuda:0", permute_392: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_393: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_394: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_395: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_402: "bf16[4096, 4096][4096, 1]cuda:0", permute_406: "bf16[4096, 4096][4096, 1]cuda:0", permute_410: "bf16[4096, 4096][4096, 1]cuda:0", div_64: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_412: "bf16[4096, 16384][16384, 1]cuda:0", permute_416: "bf16[16384, 4096][4096, 1]cuda:0", permute_422: "bf16[4096, 4096][4096, 1]cuda:0", permute_425: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_426: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_427: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_428: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_435: "bf16[4096, 4096][4096, 1]cuda:0", permute_439: "bf16[4096, 4096][4096, 1]cuda:0", permute_443: "bf16[4096, 4096][4096, 1]cuda:0", div_66: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_445: "bf16[4096, 16384][16384, 1]cuda:0", permute_449: "bf16[16384, 4096][4096, 1]cuda:0", permute_455: "bf16[4096, 4096][4096, 1]cuda:0", permute_458: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_459: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_460: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_461: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_468: "bf16[4096, 4096][4096, 1]cuda:0", permute_472: "bf16[4096, 4096][4096, 1]cuda:0", permute_476: "bf16[4096, 4096][4096, 1]cuda:0", div_68: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_478: "bf16[4096, 16384][16384, 1]cuda:0", permute_482: "bf16[16384, 4096][4096, 1]cuda:0", permute_488: "bf16[4096, 4096][4096, 1]cuda:0", permute_491: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_492: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_493: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_494: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_501: "bf16[4096, 4096][4096, 1]cuda:0", permute_505: "bf16[4096, 4096][4096, 1]cuda:0", permute_509: "bf16[4096, 4096][4096, 1]cuda:0", div_70: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_511: "bf16[4096, 16384][16384, 1]cuda:0", permute_515: "bf16[16384, 4096][4096, 1]cuda:0", permute_521: "bf16[4096, 4096][4096, 1]cuda:0", permute_524: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_525: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_526: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_527: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_534: "bf16[4096, 4096][4096, 1]cuda:0", permute_538: "bf16[4096, 4096][4096, 1]cuda:0", permute_542: "bf16[4096, 4096][4096, 1]cuda:0", div_72: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_544: "bf16[4096, 16384][16384, 1]cuda:0", permute_548: "bf16[16384, 4096][4096, 1]cuda:0", permute_554: "bf16[4096, 4096][4096, 1]cuda:0", permute_557: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_558: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_559: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_560: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_567: "bf16[4096, 4096][4096, 1]cuda:0", permute_571: "bf16[4096, 4096][4096, 1]cuda:0", permute_575: "bf16[4096, 4096][4096, 1]cuda:0", div_74: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_577: "bf16[4096, 16384][16384, 1]cuda:0", permute_581: "bf16[16384, 4096][4096, 1]cuda:0", permute_587: "bf16[4096, 4096][4096, 1]cuda:0", permute_590: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_591: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_592: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_593: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_600: "bf16[4096, 4096][4096, 1]cuda:0", permute_604: "bf16[4096, 4096][4096, 1]cuda:0", permute_608: "bf16[4096, 4096][4096, 1]cuda:0", div_76: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_610: "bf16[4096, 16384][16384, 1]cuda:0", permute_614: "bf16[16384, 4096][4096, 1]cuda:0", permute_620: "bf16[4096, 4096][4096, 1]cuda:0", permute_623: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_624: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_625: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_626: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_633: "bf16[4096, 4096][4096, 1]cuda:0", permute_637: "bf16[4096, 4096][4096, 1]cuda:0", permute_641: "bf16[4096, 4096][4096, 1]cuda:0", div_78: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_643: "bf16[4096, 16384][16384, 1]cuda:0", permute_647: "bf16[16384, 4096][4096, 1]cuda:0", permute_653: "bf16[4096, 4096][4096, 1]cuda:0", permute_656: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_657: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_658: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_659: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_666: "bf16[4096, 4096][4096, 1]cuda:0", permute_670: "bf16[4096, 4096][4096, 1]cuda:0", permute_674: "bf16[4096, 4096][4096, 1]cuda:0", div_80: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_676: "bf16[4096, 16384][16384, 1]cuda:0", permute_680: "bf16[16384, 4096][4096, 1]cuda:0", permute_686: "bf16[4096, 4096][4096, 1]cuda:0", permute_689: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_690: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_691: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_692: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_699: "bf16[4096, 4096][4096, 1]cuda:0", permute_703: "bf16[4096, 4096][4096, 1]cuda:0", permute_707: "bf16[4096, 4096][4096, 1]cuda:0", div_82: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_709: "bf16[4096, 16384][16384, 1]cuda:0", permute_713: "bf16[16384, 4096][4096, 1]cuda:0", permute_719: "bf16[4096, 4096][4096, 1]cuda:0", permute_722: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_723: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_724: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_725: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_732: "bf16[4096, 4096][4096, 1]cuda:0", permute_736: "bf16[4096, 4096][4096, 1]cuda:0", permute_740: "bf16[4096, 4096][4096, 1]cuda:0", div_84: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_742: "bf16[4096, 16384][16384, 1]cuda:0", permute_746: "bf16[16384, 4096][4096, 1]cuda:0", permute_752: "bf16[4096, 4096][4096, 1]cuda:0", permute_755: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_756: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_757: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_758: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_765: "bf16[4096, 4096][4096, 1]cuda:0", permute_769: "bf16[4096, 4096][4096, 1]cuda:0", permute_773: "bf16[4096, 4096][4096, 1]cuda:0", div_86: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_775: "bf16[4096, 16384][16384, 1]cuda:0", permute_779: "bf16[16384, 4096][4096, 1]cuda:0", permute_785: "bf16[4096, 4096][4096, 1]cuda:0", permute_788: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_789: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_790: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_791: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_798: "bf16[4096, 4096][4096, 1]cuda:0", permute_802: "bf16[4096, 4096][4096, 1]cuda:0", permute_806: "bf16[4096, 4096][4096, 1]cuda:0", div_88: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_808: "bf16[4096, 16384][16384, 1]cuda:0", permute_812: "bf16[16384, 4096][4096, 1]cuda:0", permute_818: "bf16[4096, 4096][4096, 1]cuda:0", permute_821: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_822: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_823: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_824: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_831: "bf16[4096, 4096][4096, 1]cuda:0", permute_835: "bf16[4096, 4096][4096, 1]cuda:0", permute_839: "bf16[4096, 4096][4096, 1]cuda:0", div_90: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_841: "bf16[4096, 16384][16384, 1]cuda:0", permute_845: "bf16[16384, 4096][4096, 1]cuda:0", permute_851: "bf16[4096, 4096][4096, 1]cuda:0", permute_854: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_855: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_856: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_857: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_864: "bf16[4096, 4096][4096, 1]cuda:0", permute_868: "bf16[4096, 4096][4096, 1]cuda:0", permute_872: "bf16[4096, 4096][4096, 1]cuda:0", div_92: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_874: "bf16[4096, 16384][16384, 1]cuda:0", permute_878: "bf16[16384, 4096][4096, 1]cuda:0", permute_884: "bf16[4096, 4096][4096, 1]cuda:0", permute_887: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_888: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_889: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_890: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_897: "bf16[4096, 4096][4096, 1]cuda:0", permute_901: "bf16[4096, 4096][4096, 1]cuda:0", permute_905: "bf16[4096, 4096][4096, 1]cuda:0", div_94: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_907: "bf16[4096, 16384][16384, 1]cuda:0", permute_911: "bf16[16384, 4096][4096, 1]cuda:0", permute_917: "bf16[4096, 4096][4096, 1]cuda:0", permute_920: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_921: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_922: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_923: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_930: "bf16[4096, 4096][4096, 1]cuda:0", permute_934: "bf16[4096, 4096][4096, 1]cuda:0", permute_938: "bf16[4096, 4096][4096, 1]cuda:0", div_96: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_940: "bf16[4096, 16384][16384, 1]cuda:0", permute_944: "bf16[16384, 4096][4096, 1]cuda:0", permute_950: "bf16[4096, 4096][4096, 1]cuda:0", permute_953: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_954: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_955: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_956: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_963: "bf16[4096, 4096][4096, 1]cuda:0", permute_967: "bf16[4096, 4096][4096, 1]cuda:0", permute_971: "bf16[4096, 4096][4096, 1]cuda:0", div_98: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_973: "bf16[4096, 16384][16384, 1]cuda:0", permute_977: "bf16[16384, 4096][4096, 1]cuda:0", permute_983: "bf16[4096, 4096][4096, 1]cuda:0", permute_986: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_987: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_988: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_989: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_996: "bf16[4096, 4096][4096, 1]cuda:0", permute_1000: "bf16[4096, 4096][4096, 1]cuda:0", permute_1004: "bf16[4096, 4096][4096, 1]cuda:0", div_100: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_1006: "bf16[4096, 16384][16384, 1]cuda:0", permute_1010: "bf16[16384, 4096][4096, 1]cuda:0", permute_1016: "bf16[4096, 4096][4096, 1]cuda:0", permute_1019: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_1020: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1021: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1022: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_1029: "bf16[4096, 4096][4096, 1]cuda:0", permute_1033: "bf16[4096, 4096][4096, 1]cuda:0", permute_1037: "bf16[4096, 4096][4096, 1]cuda:0", div_102: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_1039: "bf16[4096, 16384][16384, 1]cuda:0", permute_1043: "bf16[16384, 4096][4096, 1]cuda:0", permute_1049: "bf16[4096, 4096][4096, 1]cuda:0", permute_1052: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_1053: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1054: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1055: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_1062: "bf16[4096, 4096][4096, 1]cuda:0", permute_1066: "bf16[4096, 4096][4096, 1]cuda:0", permute_1070: "bf16[4096, 4096][4096, 1]cuda:0", div_104: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_1072: "bf16[4096, 16384][16384, 1]cuda:0", permute_1076: "bf16[16384, 4096][4096, 1]cuda:0", permute_1082: "bf16[4096, 4096][4096, 1]cuda:0", permute_1085: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_1086: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1087: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1088: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_1095: "bf16[4096, 4096][4096, 1]cuda:0", permute_1099: "bf16[4096, 4096][4096, 1]cuda:0", permute_1103: "bf16[4096, 4096][4096, 1]cuda:0", div_106: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_1105: "bf16[4096, 16384][16384, 1]cuda:0", permute_1109: "bf16[16384, 4096][4096, 1]cuda:0", permute_1115: "bf16[4096, 4096][4096, 1]cuda:0", permute_1118: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_1119: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1120: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1121: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_1128: "bf16[4096, 4096][4096, 1]cuda:0", permute_1132: "bf16[4096, 4096][4096, 1]cuda:0", permute_1136: "bf16[4096, 4096][4096, 1]cuda:0", div_108: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_1138: "bf16[4096, 16384][16384, 1]cuda:0", permute_1142: "bf16[16384, 4096][4096, 1]cuda:0", permute_1148: "bf16[4096, 4096][4096, 1]cuda:0", permute_1151: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_1152: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1153: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1154: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_1161: "bf16[4096, 4096][4096, 1]cuda:0", permute_1165: "bf16[4096, 4096][4096, 1]cuda:0", permute_1169: "bf16[4096, 4096][4096, 1]cuda:0", div_110: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_1171: "bf16[4096, 16384][16384, 1]cuda:0", permute_1175: "bf16[16384, 4096][4096, 1]cuda:0", permute_1181: "bf16[4096, 4096][4096, 1]cuda:0", permute_1184: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_1185: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1186: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1187: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_1194: "bf16[4096, 4096][4096, 1]cuda:0", permute_1198: "bf16[4096, 4096][4096, 1]cuda:0", permute_1202: "bf16[4096, 4096][4096, 1]cuda:0", permute_1204: "bf16[4096, 16384][16384, 1]cuda:0", permute_1208: "bf16[16384, 4096][4096, 1]cuda:0", permute_1214: "bf16[4096, 4096][4096, 1]cuda:0", permute_1217: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_1218: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1219: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1220: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_1227: "bf16[4096, 4096][4096, 1]cuda:0", permute_1231: "bf16[4096, 4096][4096, 1]cuda:0", permute_1235: "bf16[4096, 4096][4096, 1]cuda:0", tangents_1: "f32[][]cuda:0", tangents_2: "bf16[1, 128, 50400][6451200, 50400, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        div_57: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_1071);  tangents_1 = convert_element_type_1071 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_228: "i64[1, 128][129, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_788: "i64[128][1]cuda:0" = torch.ops.aten.reshape.default(slice_228, [-1]);  slice_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        unsqueeze_375: "i64[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_788, 1);  view_788 = None
        ne_4: "b8[128, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_375, -100)
        full_default_2: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "i64[128, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_4, unsqueeze_375, full_default_2);  unsqueeze_375 = full_default_2 = None

        # No stacktrace found for following nodes
        iota_default: "i64[50400][1]cuda:0" = torch.ops.prims.iota.default(50400, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 50400][50400, 1]cuda:0" = torch.ops.aten.reshape.default(iota_default, [1, 50400]);  iota_default = None
        expand_default: "i64[128, 50400][1, 0]cuda:0" = torch.ops.aten.expand.default(where_3, [128, 50400]);  where_3 = None
        eq_tensor: "b8[128, 50400][50400, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_self: "f32[128, 50400][50400, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_4: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_4, div_57, full_default);  ne_4 = div_57 = full_default = None
        mul_282: "f32[128, 50400][50400, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self, where_4);  where_self = where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_1070: "f32[1, 128, 50400][6451200, 50400, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_786, torch.float32);  view_786 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_787: "f32[128, 50400][50400, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1070, [-1, 50400]);  convert_element_type_1070 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        sub_59: "f32[128, 50400][50400, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_787, amax_28);  view_787 = amax_28 = None
        sub_60: "f32[128, 50400][50400, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_59, log);  sub_59 = log = None
        exp_29: "f32[128, 50400][50400, 1]cuda:0" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        sum_32: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_282, [1], True)
        mul_283: "f32[128, 50400][50400, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_29, sum_32);  exp_29 = sum_32 = None
        sub_61: "f32[128, 50400][50400, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_789: "f32[1, 128, 50400][6451200, 50400, 1]cuda:0" = torch.ops.aten.reshape.default(sub_61, [1, 128, 50400]);  sub_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_1072: "bf16[1, 128, 50400][6451200, 50400, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_789, torch.bfloat16);  view_789 = None
        add_257: "bf16[1, 128, 50400][6451200, 50400, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, convert_element_type_1072);  tangents_2 = convert_element_type_1072 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:623 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_790: "bf16[128, 50400][50400, 1]cuda:0" = torch.ops.aten.reshape.default(add_257, [128, 50400]);  add_257 = None
        mm_112: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_790, permute_309);  permute_309 = None
        permute_310: "bf16[50400, 128][1, 50400]cuda:0" = torch.ops.aten.permute.default(view_790, [1, 0])
        mm_113: "bf16[50400, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_310, view_785);  permute_310 = view_785 = None
        sum_33: "f32[1, 50400][50400, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_790, [0], True, dtype = torch.float32);  view_790 = None
        view_791: "f32[50400][1]cuda:0" = torch.ops.aten.reshape.default(sum_33, [50400]);  sum_33 = None
        convert_element_type_1077: "bf16[50400][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_791, torch.bfloat16);  view_791 = None
        view_792: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_112, [1, 128, 4096]);  mm_112 = None
        convert_element_type_1078: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_792, torch.float32);  view_792 = None
        convert_element_type_1079: "f32[50400, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_1080: "f32[50400][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1077, torch.float32);  convert_element_type_1077 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:542 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_285: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1078, primals_311);  primals_311 = None
        mul_286: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_285, 4096)
        sum_34: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_285, [2], True)
        mul_287: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_285, mul_280);  mul_285 = None
        sum_35: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True);  mul_287 = None
        mul_288: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_280, sum_35);  sum_35 = None
        sub_63: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_286, sum_34);  mul_286 = sum_34 = None
        sub_64: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_63, mul_288);  sub_63 = mul_288 = None
        mul_289: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_58, sub_64);  div_58 = sub_64 = None
        mul_290: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1078, mul_280);  mul_280 = None
        sum_36: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_290, [0, 1]);  mul_290 = None
        sum_37: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1078, [0, 1]);  convert_element_type_1078 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1081: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_289, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_794: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1081, [128, 4096]);  convert_element_type_1081 = None
        mm_114: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_794, permute_313);  permute_313 = None
        permute_314: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_794, [1, 0])
        mm_115: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_314, view_782);  view_782 = None
        sum_38: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_794, [0], True, dtype = torch.float32)
        view_795: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_38, [4096]);  sum_38 = None
        convert_element_type_1086: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_795, torch.bfloat16);  view_795 = None
        view_796: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_114, [1, 128, 16384]);  mm_114 = None
        convert_element_type_1087: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_796, torch.float32);  view_796 = None
        convert_element_type_1088: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_1089: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1086, torch.float32);  convert_element_type_1086 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_781: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [1, 128, 16384]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_276: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_781, 0.5)
        mul_291: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1087, mul_276);  mul_276 = None
        convert_element_type_1057: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_781, torch.float32)
        pow_28: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1057, 3.0)
        mul_277: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_28, 0.044715);  pow_28 = None
        add_251: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_781, mul_277);  view_781 = mul_277 = None
        mul_278: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_251, 0.7978845608028654);  add_251 = None
        tanh_27: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_278);  mul_278 = None
        add_252: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_27, 1.0)
        mul_292: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1087, add_252);  convert_element_type_1087 = add_252 = None
        convert_element_type_1090: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_292, torch.bfloat16);  mul_292 = None
        mul_293: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_27, tanh_27);  tanh_27 = None
        sub_65: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_293);  mul_293 = None
        mul_294: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_291, sub_65);  mul_291 = sub_65 = None
        mul_295: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_294, 0.7978845608028654);  mul_294 = None
        convert_element_type_1091: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_295, torch.bfloat16)
        mul_296: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_295, 0.044715);  mul_295 = None
        pow_29: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1057, 2.0);  convert_element_type_1057 = None
        mul_297: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_29, 3.0);  pow_29 = None
        mul_298: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
        convert_element_type_1092: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_298, torch.bfloat16);  mul_298 = None
        add_258: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1091, convert_element_type_1092);  convert_element_type_1091 = convert_element_type_1092 = None
        mul_299: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1090, 0.5);  convert_element_type_1090 = None
        add_259: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_258, mul_299);  add_258 = mul_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_797: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_259, [128, 16384]);  add_259 = None
        mm_116: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_797, permute_317);  permute_317 = None
        permute_318: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_797, [1, 0])
        mm_117: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_318, view_756);  permute_318 = None
        sum_39: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_797, [0], True, dtype = torch.float32);  view_797 = None
        view_798: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_39, [16384]);  sum_39 = None
        convert_element_type_1097: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_798, torch.bfloat16);  view_798 = None
        view_799: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_116, [1, 128, 4096]);  mm_116 = None
        convert_element_type_1098: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_799, torch.float32);  view_799 = None
        convert_element_type_1099: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_1100: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1097, torch.float32);  convert_element_type_1097 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_118: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_314, view_778);  permute_314 = view_778 = None
        mm_119: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_794, permute_323);  view_794 = permute_323 = None
        view_801: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_119, [1, 128, 4096]);  mm_119 = None
        convert_element_type_1105: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_118, torch.float32);  mm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_802: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_801, [1, 128, 16, 256]);  view_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_325: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_802, [0, 2, 1, 3]);  view_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_803: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_325, [16, 128, 256]);  permute_325 = None
        bmm_56: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_326, view_803);  permute_326 = None
        bmm_57: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_803, permute_327);  view_803 = permute_327 = None
        view_804: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_56, [1, 16, 128, 256]);  bmm_56 = None
        view_805: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_57, [1, 16, 128, 128]);  bmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1110: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_805, torch.float32);  view_805 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_300: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1110, div_55);  convert_element_type_1110 = None
        sum_40: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_300, [-1], True)
        neg_57: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_55);  div_55 = None
        fma: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_57, sum_40, mul_300);  neg_57 = sum_40 = mul_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1111: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_59: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1111, 16.0);  convert_element_type_1111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_806: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_59, [16, 128, 128]);  div_59 = None
        bmm_58: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_328, view_806);  permute_328 = None
        bmm_59: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_806, permute_329);  view_806 = permute_329 = None
        view_807: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_58, [1, 16, 256, 128]);  bmm_58 = None
        view_808: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_59, [1, 16, 128, 256]);  bmm_59 = None
        convert_element_type_1117: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_807, torch.float32);  view_807 = None
        permute_330: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1117, [0, 1, 3, 2]);  convert_element_type_1117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1118: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_330, torch.bfloat16);  permute_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_27: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_808, torch.bfloat16);  view_808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_331: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_27, [0, 2, 1, 3]);  convert_element_type_default_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_332: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1118, [0, 2, 1, 3]);  convert_element_type_1118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_229: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_331, 3, 0, 64)
        slice_230: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_331, 3, 64, 256);  permute_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_231: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_332, 3, 0, 64)
        slice_232: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_332, 3, 64, 256);  permute_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_217: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_363, [1, 128, 1, 32, 2]);  unsqueeze_363 = None
        clone_217: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_217, memory_format = torch.contiguous_format);  expand_217 = None
        view_765: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_217, [1, 128, 1, 64]);  clone_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_301: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_229, view_765)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_809: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_301, [1, 128, 16, 32, 2]);  mul_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_809, -1, 0)
        select_1: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_809, -1, 1);  view_809 = None
        neg_58: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select);  select = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        full_default_7: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.full.default([1, 128, 16, 64], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_58, 3, 1, 9223372036854775807, 2);  neg_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_1: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_1, 3, 0, 9223372036854775807, 2);  select_1 = None
        add_260: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter, slice_scatter_1);  slice_scatter = slice_scatter_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_218: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_365, [1, 128, 1, 32, 2]);  unsqueeze_365 = None
        clone_218: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_218, memory_format = torch.contiguous_format);  expand_218 = None
        view_766: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_218, [1, 128, 1, 64]);  clone_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_302: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_229, view_766);  slice_229 = None
        add_261: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_260, mul_302);  add_260 = mul_302 = None
        mul_303: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_231, view_765);  view_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_810: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_303, [1, 128, 16, 32, 2]);  mul_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_2: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_810, -1, 0)
        select_3: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_810, -1, 1);  view_810 = None
        neg_59: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_2);  select_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_2: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_59, 3, 1, 9223372036854775807, 2);  neg_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_3: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_3, 3, 0, 9223372036854775807, 2);  select_3 = None
        add_262: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_2, slice_scatter_3);  slice_scatter_2 = slice_scatter_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_304: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_231, view_766);  slice_231 = view_766 = None
        add_263: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_262, mul_304);  add_262 = mul_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        full_default_11: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.full.default([1, 128, 16, 256], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_4: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_230, 3, 64, 9223372036854775807);  slice_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_5: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_261, 3, 0, 64);  add_261 = None
        add_264: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_4, slice_scatter_5);  slice_scatter_4 = slice_scatter_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_6: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_232, 3, 64, 9223372036854775807);  slice_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_7: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_263, 3, 0, 64);  add_263 = None
        add_265: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_6, slice_scatter_7);  slice_scatter_6 = slice_scatter_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_333: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_804, [0, 2, 1, 3]);  view_804 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_225: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_333, memory_format = torch.contiguous_format);  permute_333 = None
        view_811: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_225, [1, 128, 4096]);  clone_225 = None
        view_812: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_265, [1, 128, 4096]);  add_265 = None
        view_813: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_264, [1, 128, 4096]);  add_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_814: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_811, [128, 4096]);  view_811 = None
        permute_334: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_814, [1, 0])
        mm_120: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_334, view_756);  permute_334 = None
        mm_121: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_814, permute_336);  view_814 = permute_336 = None
        view_815: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_121, [1, 128, 4096]);  mm_121 = None
        convert_element_type_1124: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_815, torch.float32);  view_815 = None
        add_266: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1098, convert_element_type_1124);  convert_element_type_1098 = convert_element_type_1124 = None
        convert_element_type_1125: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_120, torch.float32);  mm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_816: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_812, [128, 4096]);  view_812 = None
        permute_338: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_816, [1, 0])
        mm_122: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_338, view_756);  permute_338 = None
        mm_123: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_816, permute_340);  view_816 = permute_340 = None
        view_817: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_123, [1, 128, 4096]);  mm_123 = None
        convert_element_type_1130: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_817, torch.float32);  view_817 = None
        add_267: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_266, convert_element_type_1130);  add_266 = convert_element_type_1130 = None
        convert_element_type_1131: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_122, torch.float32);  mm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_818: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_813, [128, 4096]);  view_813 = None
        permute_342: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_818, [1, 0])
        mm_124: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_342, view_756);  permute_342 = view_756 = None
        mm_125: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_818, permute_344);  view_818 = permute_344 = None
        view_819: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_125, [1, 128, 4096]);  mm_125 = None
        convert_element_type_1136: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_819, torch.float32);  view_819 = None
        add_268: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_267, convert_element_type_1136);  add_267 = convert_element_type_1136 = None
        convert_element_type_1137: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_124, torch.float32);  mm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_306: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_268, primals_300);  primals_300 = None
        mul_307: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_306, 4096)
        sum_41: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_306, [2], True)
        mul_308: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_306, mul_270);  mul_306 = None
        sum_42: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_308, [2], True);  mul_308 = None
        mul_309: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_270, sum_42);  sum_42 = None
        sub_67: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_307, sum_41);  mul_307 = sum_41 = None
        sub_68: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_67, mul_309);  sub_67 = mul_309 = None
        mul_310: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_60, sub_68);  div_60 = sub_68 = None
        mul_311: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_268, mul_270);  mul_270 = None
        sum_43: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_311, [0, 1]);  mul_311 = None
        sum_44: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_268, [0, 1]);  add_268 = None
        add_269: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_289, mul_310);  mul_289 = mul_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1138: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_269, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_820: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1138, [128, 4096]);  convert_element_type_1138 = None
        mm_126: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_820, permute_346);  permute_346 = None
        permute_347: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_820, [1, 0])
        mm_127: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_347, view_754);  view_754 = None
        sum_45: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_820, [0], True, dtype = torch.float32)
        view_821: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_45, [4096]);  sum_45 = None
        convert_element_type_1143: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_821, torch.bfloat16);  view_821 = None
        view_822: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_126, [1, 128, 16384]);  mm_126 = None
        convert_element_type_1144: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_822, torch.float32);  view_822 = None
        convert_element_type_1145: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_1146: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1143, torch.float32);  convert_element_type_1143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_753: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [1, 128, 16384]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_266: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_753, 0.5)
        mul_312: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1144, mul_266);  mul_266 = None
        convert_element_type_1019: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_753, torch.float32)
        pow_27: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1019, 3.0)
        mul_267: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
        add_242: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_753, mul_267);  view_753 = mul_267 = None
        mul_268: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_242, 0.7978845608028654);  add_242 = None
        tanh_26: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_268);  mul_268 = None
        add_243: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_26, 1.0)
        mul_313: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1144, add_243);  convert_element_type_1144 = add_243 = None
        convert_element_type_1147: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_313, torch.bfloat16);  mul_313 = None
        mul_314: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_26, tanh_26);  tanh_26 = None
        sub_69: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_314);  mul_314 = None
        mul_315: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_312, sub_69);  mul_312 = sub_69 = None
        mul_316: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_315, 0.7978845608028654);  mul_315 = None
        convert_element_type_1148: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_316, torch.bfloat16)
        mul_317: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_316, 0.044715);  mul_316 = None
        pow_30: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1019, 2.0);  convert_element_type_1019 = None
        mul_318: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_30, 3.0);  pow_30 = None
        mul_319: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_317, mul_318);  mul_317 = mul_318 = None
        convert_element_type_1149: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_319, torch.bfloat16);  mul_319 = None
        add_270: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1148, convert_element_type_1149);  convert_element_type_1148 = convert_element_type_1149 = None
        mul_320: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1147, 0.5);  convert_element_type_1147 = None
        add_271: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_270, mul_320);  add_270 = mul_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_823: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_271, [128, 16384]);  add_271 = None
        mm_128: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_823, permute_350);  permute_350 = None
        permute_351: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_823, [1, 0])
        mm_129: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_351, view_728);  permute_351 = None
        sum_46: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_823, [0], True, dtype = torch.float32);  view_823 = None
        view_824: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_46, [16384]);  sum_46 = None
        convert_element_type_1154: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_824, torch.bfloat16);  view_824 = None
        view_825: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_128, [1, 128, 4096]);  mm_128 = None
        convert_element_type_1155: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_825, torch.float32);  view_825 = None
        convert_element_type_1156: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        convert_element_type_1157: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1154, torch.float32);  convert_element_type_1154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_130: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_347, view_750);  permute_347 = view_750 = None
        mm_131: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_820, permute_356);  view_820 = permute_356 = None
        view_827: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_131, [1, 128, 4096]);  mm_131 = None
        convert_element_type_1162: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_130, torch.float32);  mm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_828: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_827, [1, 128, 16, 256]);  view_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_358: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_828, [0, 2, 1, 3]);  view_828 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_829: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_358, [16, 128, 256]);  permute_358 = None
        bmm_60: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_359, view_829);  permute_359 = None
        bmm_61: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_829, permute_360);  view_829 = permute_360 = None
        view_830: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_60, [1, 16, 128, 256]);  bmm_60 = None
        view_831: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_61, [1, 16, 128, 128]);  bmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1167: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_831, torch.float32);  view_831 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_321: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1167, div_53);  convert_element_type_1167 = None
        sum_47: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_321, [-1], True)
        neg_60: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_53);  div_53 = None
        fma_1: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_60, sum_47, mul_321);  neg_60 = sum_47 = mul_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1168: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_61: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1168, 16.0);  convert_element_type_1168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_832: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_61, [16, 128, 128]);  div_61 = None
        bmm_62: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_361, view_832);  permute_361 = None
        bmm_63: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_832, permute_362);  view_832 = permute_362 = None
        view_833: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_62, [1, 16, 256, 128]);  bmm_62 = None
        view_834: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_63, [1, 16, 128, 256]);  bmm_63 = None
        convert_element_type_1174: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_833, torch.float32);  view_833 = None
        permute_363: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1174, [0, 1, 3, 2]);  convert_element_type_1174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1175: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_363, torch.bfloat16);  permute_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_26: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_834, torch.bfloat16);  view_834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_364: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_26, [0, 2, 1, 3]);  convert_element_type_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_365: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1175, [0, 2, 1, 3]);  convert_element_type_1175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_233: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_364, 3, 0, 64)
        slice_234: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_364, 3, 64, 256);  permute_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_235: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_365, 3, 0, 64)
        slice_236: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_365, 3, 64, 256);  permute_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_209: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_350, [1, 128, 1, 32, 2]);  unsqueeze_350 = None
        clone_209: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_209, memory_format = torch.contiguous_format);  expand_209 = None
        view_737: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_209, [1, 128, 1, 64]);  clone_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_322: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_233, view_737)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_835: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_322, [1, 128, 16, 32, 2]);  mul_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_4: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_835, -1, 0)
        select_5: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_835, -1, 1);  view_835 = None
        neg_61: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_4);  select_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_8: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_61, 3, 1, 9223372036854775807, 2);  neg_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_9: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_5, 3, 0, 9223372036854775807, 2);  select_5 = None
        add_272: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_8, slice_scatter_9);  slice_scatter_8 = slice_scatter_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_210: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_352, [1, 128, 1, 32, 2]);  unsqueeze_352 = None
        clone_210: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_210, memory_format = torch.contiguous_format);  expand_210 = None
        view_738: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_210, [1, 128, 1, 64]);  clone_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_323: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_233, view_738);  slice_233 = None
        add_273: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_272, mul_323);  add_272 = mul_323 = None
        mul_324: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_235, view_737);  view_737 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_836: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_324, [1, 128, 16, 32, 2]);  mul_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_6: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_836, -1, 0)
        select_7: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_836, -1, 1);  view_836 = None
        neg_62: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_6);  select_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_10: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_62, 3, 1, 9223372036854775807, 2);  neg_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_11: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_7, 3, 0, 9223372036854775807, 2);  select_7 = None
        add_274: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_10, slice_scatter_11);  slice_scatter_10 = slice_scatter_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_325: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_235, view_738);  slice_235 = view_738 = None
        add_275: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_274, mul_325);  add_274 = mul_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_12: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_234, 3, 64, 9223372036854775807);  slice_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_13: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_273, 3, 0, 64);  add_273 = None
        add_276: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_12, slice_scatter_13);  slice_scatter_12 = slice_scatter_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_14: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_236, 3, 64, 9223372036854775807);  slice_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_15: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_275, 3, 0, 64);  add_275 = None
        add_277: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_14, slice_scatter_15);  slice_scatter_14 = slice_scatter_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_366: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_830, [0, 2, 1, 3]);  view_830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_226: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_366, memory_format = torch.contiguous_format);  permute_366 = None
        view_837: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_226, [1, 128, 4096]);  clone_226 = None
        view_838: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_277, [1, 128, 4096]);  add_277 = None
        view_839: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_276, [1, 128, 4096]);  add_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_840: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_837, [128, 4096]);  view_837 = None
        permute_367: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_840, [1, 0])
        mm_132: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_367, view_728);  permute_367 = None
        mm_133: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_840, permute_369);  view_840 = permute_369 = None
        view_841: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_133, [1, 128, 4096]);  mm_133 = None
        convert_element_type_1181: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_841, torch.float32);  view_841 = None
        add_278: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1155, convert_element_type_1181);  convert_element_type_1155 = convert_element_type_1181 = None
        convert_element_type_1182: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_132, torch.float32);  mm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_842: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_838, [128, 4096]);  view_838 = None
        permute_371: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_842, [1, 0])
        mm_134: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_371, view_728);  permute_371 = None
        mm_135: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_842, permute_373);  view_842 = permute_373 = None
        view_843: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_135, [1, 128, 4096]);  mm_135 = None
        convert_element_type_1187: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_843, torch.float32);  view_843 = None
        add_279: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_278, convert_element_type_1187);  add_278 = convert_element_type_1187 = None
        convert_element_type_1188: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_134, torch.float32);  mm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_844: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_839, [128, 4096]);  view_839 = None
        permute_375: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_844, [1, 0])
        mm_136: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_375, view_728);  permute_375 = view_728 = None
        mm_137: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_844, permute_377);  view_844 = permute_377 = None
        view_845: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_137, [1, 128, 4096]);  mm_137 = None
        convert_element_type_1193: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_845, torch.float32);  view_845 = None
        add_280: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_279, convert_element_type_1193);  add_279 = convert_element_type_1193 = None
        convert_element_type_1194: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_136, torch.float32);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_327: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_280, primals_289);  primals_289 = None
        mul_328: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, 4096)
        sum_48: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True)
        mul_329: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, mul_260);  mul_327 = None
        sum_49: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True);  mul_329 = None
        mul_330: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_260, sum_49);  sum_49 = None
        sub_71: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_328, sum_48);  mul_328 = sum_48 = None
        sub_72: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_71, mul_330);  sub_71 = mul_330 = None
        mul_331: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_62, sub_72);  div_62 = sub_72 = None
        mul_332: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_280, mul_260);  mul_260 = None
        sum_50: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1]);  mul_332 = None
        sum_51: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_280, [0, 1]);  add_280 = None
        add_281: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_269, mul_331);  add_269 = mul_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1195: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_281, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_846: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1195, [128, 4096]);  convert_element_type_1195 = None
        mm_138: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_846, permute_379);  permute_379 = None
        permute_380: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_846, [1, 0])
        mm_139: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_380, view_726);  view_726 = None
        sum_52: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_846, [0], True, dtype = torch.float32)
        view_847: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_52, [4096]);  sum_52 = None
        convert_element_type_1200: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_847, torch.bfloat16);  view_847 = None
        view_848: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_138, [1, 128, 16384]);  mm_138 = None
        convert_element_type_1201: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_848, torch.float32);  view_848 = None
        convert_element_type_1202: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_1203: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1200, torch.float32);  convert_element_type_1200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_725: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [1, 128, 16384]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_256: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_725, 0.5)
        mul_333: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1201, mul_256);  mul_256 = None
        convert_element_type_981: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_725, torch.float32)
        pow_26: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_981, 3.0)
        mul_257: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_26, 0.044715);  pow_26 = None
        add_233: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_725, mul_257);  view_725 = mul_257 = None
        mul_258: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_233, 0.7978845608028654);  add_233 = None
        tanh_25: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_258);  mul_258 = None
        add_234: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_25, 1.0)
        mul_334: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1201, add_234);  convert_element_type_1201 = add_234 = None
        convert_element_type_1204: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_334, torch.bfloat16);  mul_334 = None
        mul_335: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_25, tanh_25);  tanh_25 = None
        sub_73: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_335);  mul_335 = None
        mul_336: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_333, sub_73);  mul_333 = sub_73 = None
        mul_337: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_336, 0.7978845608028654);  mul_336 = None
        convert_element_type_1205: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_337, torch.bfloat16)
        mul_338: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_337, 0.044715);  mul_337 = None
        pow_31: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_981, 2.0);  convert_element_type_981 = None
        mul_339: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_31, 3.0);  pow_31 = None
        mul_340: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_338, mul_339);  mul_338 = mul_339 = None
        convert_element_type_1206: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_340, torch.bfloat16);  mul_340 = None
        add_282: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1205, convert_element_type_1206);  convert_element_type_1205 = convert_element_type_1206 = None
        mul_341: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1204, 0.5);  convert_element_type_1204 = None
        add_283: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_282, mul_341);  add_282 = mul_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_849: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_283, [128, 16384]);  add_283 = None
        mm_140: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_849, permute_383);  permute_383 = None
        permute_384: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_849, [1, 0])
        mm_141: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_384, view_700);  permute_384 = None
        sum_53: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_849, [0], True, dtype = torch.float32);  view_849 = None
        view_850: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_53, [16384]);  sum_53 = None
        convert_element_type_1211: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_850, torch.bfloat16);  view_850 = None
        view_851: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_140, [1, 128, 4096]);  mm_140 = None
        convert_element_type_1212: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_851, torch.float32);  view_851 = None
        convert_element_type_1213: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_1214: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1211, torch.float32);  convert_element_type_1211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_142: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_380, view_722);  permute_380 = view_722 = None
        mm_143: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_846, permute_389);  view_846 = permute_389 = None
        view_853: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_143, [1, 128, 4096]);  mm_143 = None
        convert_element_type_1219: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_142, torch.float32);  mm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_854: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_853, [1, 128, 16, 256]);  view_853 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_391: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_854, [0, 2, 1, 3]);  view_854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_855: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_391, [16, 128, 256]);  permute_391 = None
        bmm_64: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_392, view_855);  permute_392 = None
        bmm_65: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_855, permute_393);  view_855 = permute_393 = None
        view_856: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_64, [1, 16, 128, 256]);  bmm_64 = None
        view_857: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_65, [1, 16, 128, 128]);  bmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1224: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_857, torch.float32);  view_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_342: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1224, div_51);  convert_element_type_1224 = None
        sum_54: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_342, [-1], True)
        neg_63: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_51);  div_51 = None
        fma_2: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_63, sum_54, mul_342);  neg_63 = sum_54 = mul_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1225: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_63: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1225, 16.0);  convert_element_type_1225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_858: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_63, [16, 128, 128]);  div_63 = None
        bmm_66: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_394, view_858);  permute_394 = None
        bmm_67: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_858, permute_395);  view_858 = permute_395 = None
        view_859: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_66, [1, 16, 256, 128]);  bmm_66 = None
        view_860: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_67, [1, 16, 128, 256]);  bmm_67 = None
        convert_element_type_1231: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_859, torch.float32);  view_859 = None
        permute_396: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1231, [0, 1, 3, 2]);  convert_element_type_1231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1232: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_396, torch.bfloat16);  permute_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_25: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_860, torch.bfloat16);  view_860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_397: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_25, [0, 2, 1, 3]);  convert_element_type_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_398: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1232, [0, 2, 1, 3]);  convert_element_type_1232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_237: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_397, 3, 0, 64)
        slice_238: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_397, 3, 64, 256);  permute_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_239: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_398, 3, 0, 64)
        slice_240: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_398, 3, 64, 256);  permute_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_201: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_337, [1, 128, 1, 32, 2]);  unsqueeze_337 = None
        clone_201: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_201, memory_format = torch.contiguous_format);  expand_201 = None
        view_709: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_201, [1, 128, 1, 64]);  clone_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_343: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_237, view_709)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_861: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_343, [1, 128, 16, 32, 2]);  mul_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_8: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_861, -1, 0)
        select_9: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_861, -1, 1);  view_861 = None
        neg_64: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_8);  select_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_16: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_64, 3, 1, 9223372036854775807, 2);  neg_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_17: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_9, 3, 0, 9223372036854775807, 2);  select_9 = None
        add_284: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_16, slice_scatter_17);  slice_scatter_16 = slice_scatter_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_202: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_339, [1, 128, 1, 32, 2]);  unsqueeze_339 = None
        clone_202: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_202, memory_format = torch.contiguous_format);  expand_202 = None
        view_710: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_202, [1, 128, 1, 64]);  clone_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_344: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_237, view_710);  slice_237 = None
        add_285: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_284, mul_344);  add_284 = mul_344 = None
        mul_345: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_239, view_709);  view_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_862: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_345, [1, 128, 16, 32, 2]);  mul_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_10: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_862, -1, 0)
        select_11: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_862, -1, 1);  view_862 = None
        neg_65: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_10);  select_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_18: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_65, 3, 1, 9223372036854775807, 2);  neg_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_19: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_11, 3, 0, 9223372036854775807, 2);  select_11 = None
        add_286: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_18, slice_scatter_19);  slice_scatter_18 = slice_scatter_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_346: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_239, view_710);  slice_239 = view_710 = None
        add_287: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_286, mul_346);  add_286 = mul_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_20: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_238, 3, 64, 9223372036854775807);  slice_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_21: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_285, 3, 0, 64);  add_285 = None
        add_288: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_20, slice_scatter_21);  slice_scatter_20 = slice_scatter_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_22: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_240, 3, 64, 9223372036854775807);  slice_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_23: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_287, 3, 0, 64);  add_287 = None
        add_289: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_22, slice_scatter_23);  slice_scatter_22 = slice_scatter_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_399: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_856, [0, 2, 1, 3]);  view_856 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_227: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_399, memory_format = torch.contiguous_format);  permute_399 = None
        view_863: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_227, [1, 128, 4096]);  clone_227 = None
        view_864: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_289, [1, 128, 4096]);  add_289 = None
        view_865: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_288, [1, 128, 4096]);  add_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_866: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_863, [128, 4096]);  view_863 = None
        permute_400: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_866, [1, 0])
        mm_144: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_400, view_700);  permute_400 = None
        mm_145: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_866, permute_402);  view_866 = permute_402 = None
        view_867: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_145, [1, 128, 4096]);  mm_145 = None
        convert_element_type_1238: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_867, torch.float32);  view_867 = None
        add_290: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1212, convert_element_type_1238);  convert_element_type_1212 = convert_element_type_1238 = None
        convert_element_type_1239: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_144, torch.float32);  mm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_868: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_864, [128, 4096]);  view_864 = None
        permute_404: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_868, [1, 0])
        mm_146: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_404, view_700);  permute_404 = None
        mm_147: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_868, permute_406);  view_868 = permute_406 = None
        view_869: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_147, [1, 128, 4096]);  mm_147 = None
        convert_element_type_1244: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_869, torch.float32);  view_869 = None
        add_291: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_290, convert_element_type_1244);  add_290 = convert_element_type_1244 = None
        convert_element_type_1245: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_146, torch.float32);  mm_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_870: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_865, [128, 4096]);  view_865 = None
        permute_408: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_870, [1, 0])
        mm_148: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_408, view_700);  permute_408 = view_700 = None
        mm_149: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_870, permute_410);  view_870 = permute_410 = None
        view_871: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_149, [1, 128, 4096]);  mm_149 = None
        convert_element_type_1250: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_871, torch.float32);  view_871 = None
        add_292: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_291, convert_element_type_1250);  add_291 = convert_element_type_1250 = None
        convert_element_type_1251: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_148, torch.float32);  mm_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_348: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_292, primals_278);  primals_278 = None
        mul_349: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_348, 4096)
        sum_55: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_348, [2], True)
        mul_350: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_348, mul_250);  mul_348 = None
        sum_56: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_350, [2], True);  mul_350 = None
        mul_351: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_250, sum_56);  sum_56 = None
        sub_75: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_349, sum_55);  mul_349 = sum_55 = None
        sub_76: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_75, mul_351);  sub_75 = mul_351 = None
        mul_352: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_64, sub_76);  div_64 = sub_76 = None
        mul_353: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_292, mul_250);  mul_250 = None
        sum_57: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_353, [0, 1]);  mul_353 = None
        sum_58: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_292, [0, 1]);  add_292 = None
        add_293: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_281, mul_352);  add_281 = mul_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1252: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_293, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_872: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1252, [128, 4096]);  convert_element_type_1252 = None
        mm_150: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_872, permute_412);  permute_412 = None
        permute_413: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_872, [1, 0])
        mm_151: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_413, view_698);  view_698 = None
        sum_59: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_872, [0], True, dtype = torch.float32)
        view_873: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_59, [4096]);  sum_59 = None
        convert_element_type_1257: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_873, torch.bfloat16);  view_873 = None
        view_874: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_150, [1, 128, 16384]);  mm_150 = None
        convert_element_type_1258: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_874, torch.float32);  view_874 = None
        convert_element_type_1259: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_151, torch.float32);  mm_151 = None
        convert_element_type_1260: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1257, torch.float32);  convert_element_type_1257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_697: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [1, 128, 16384]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_246: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_697, 0.5)
        mul_354: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1258, mul_246);  mul_246 = None
        convert_element_type_943: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_697, torch.float32)
        pow_25: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_943, 3.0)
        mul_247: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
        add_224: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_697, mul_247);  view_697 = mul_247 = None
        mul_248: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_224, 0.7978845608028654);  add_224 = None
        tanh_24: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_248);  mul_248 = None
        add_225: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_24, 1.0)
        mul_355: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1258, add_225);  convert_element_type_1258 = add_225 = None
        convert_element_type_1261: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_355, torch.bfloat16);  mul_355 = None
        mul_356: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_24, tanh_24);  tanh_24 = None
        sub_77: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_356);  mul_356 = None
        mul_357: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_354, sub_77);  mul_354 = sub_77 = None
        mul_358: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_357, 0.7978845608028654);  mul_357 = None
        convert_element_type_1262: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_358, torch.bfloat16)
        mul_359: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_358, 0.044715);  mul_358 = None
        pow_32: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_943, 2.0);  convert_element_type_943 = None
        mul_360: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_32, 3.0);  pow_32 = None
        mul_361: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_359, mul_360);  mul_359 = mul_360 = None
        convert_element_type_1263: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_361, torch.bfloat16);  mul_361 = None
        add_294: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1262, convert_element_type_1263);  convert_element_type_1262 = convert_element_type_1263 = None
        mul_362: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1261, 0.5);  convert_element_type_1261 = None
        add_295: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_294, mul_362);  add_294 = mul_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_875: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_295, [128, 16384]);  add_295 = None
        mm_152: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_875, permute_416);  permute_416 = None
        permute_417: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_875, [1, 0])
        mm_153: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_417, view_672);  permute_417 = None
        sum_60: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_875, [0], True, dtype = torch.float32);  view_875 = None
        view_876: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_60, [16384]);  sum_60 = None
        convert_element_type_1268: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_876, torch.bfloat16);  view_876 = None
        view_877: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_152, [1, 128, 4096]);  mm_152 = None
        convert_element_type_1269: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_877, torch.float32);  view_877 = None
        convert_element_type_1270: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_153, torch.float32);  mm_153 = None
        convert_element_type_1271: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1268, torch.float32);  convert_element_type_1268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_154: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_413, view_694);  permute_413 = view_694 = None
        mm_155: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_872, permute_422);  view_872 = permute_422 = None
        view_879: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_155, [1, 128, 4096]);  mm_155 = None
        convert_element_type_1276: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_154, torch.float32);  mm_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_880: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_879, [1, 128, 16, 256]);  view_879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_424: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_880, [0, 2, 1, 3]);  view_880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_881: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_424, [16, 128, 256]);  permute_424 = None
        bmm_68: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_425, view_881);  permute_425 = None
        bmm_69: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_881, permute_426);  view_881 = permute_426 = None
        view_882: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_68, [1, 16, 128, 256]);  bmm_68 = None
        view_883: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_69, [1, 16, 128, 128]);  bmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1281: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_883, torch.float32);  view_883 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_363: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1281, div_49);  convert_element_type_1281 = None
        sum_61: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_363, [-1], True)
        neg_66: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_49);  div_49 = None
        fma_3: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_66, sum_61, mul_363);  neg_66 = sum_61 = mul_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1282: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_65: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1282, 16.0);  convert_element_type_1282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_884: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_65, [16, 128, 128]);  div_65 = None
        bmm_70: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_427, view_884);  permute_427 = None
        bmm_71: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_884, permute_428);  view_884 = permute_428 = None
        view_885: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_70, [1, 16, 256, 128]);  bmm_70 = None
        view_886: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_71, [1, 16, 128, 256]);  bmm_71 = None
        convert_element_type_1288: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_885, torch.float32);  view_885 = None
        permute_429: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1288, [0, 1, 3, 2]);  convert_element_type_1288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1289: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_429, torch.bfloat16);  permute_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_24: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_886, torch.bfloat16);  view_886 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_430: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_24, [0, 2, 1, 3]);  convert_element_type_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_431: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1289, [0, 2, 1, 3]);  convert_element_type_1289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_241: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_430, 3, 0, 64)
        slice_242: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_430, 3, 64, 256);  permute_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_243: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_431, 3, 0, 64)
        slice_244: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_431, 3, 64, 256);  permute_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_193: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_324, [1, 128, 1, 32, 2]);  unsqueeze_324 = None
        clone_193: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_193, memory_format = torch.contiguous_format);  expand_193 = None
        view_681: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_193, [1, 128, 1, 64]);  clone_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_364: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_241, view_681)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_887: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_364, [1, 128, 16, 32, 2]);  mul_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_12: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_887, -1, 0)
        select_13: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_887, -1, 1);  view_887 = None
        neg_67: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_12);  select_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_24: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_67, 3, 1, 9223372036854775807, 2);  neg_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_25: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_13, 3, 0, 9223372036854775807, 2);  select_13 = None
        add_296: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_24, slice_scatter_25);  slice_scatter_24 = slice_scatter_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_194: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_326, [1, 128, 1, 32, 2]);  unsqueeze_326 = None
        clone_194: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_194, memory_format = torch.contiguous_format);  expand_194 = None
        view_682: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_194, [1, 128, 1, 64]);  clone_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_365: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_241, view_682);  slice_241 = None
        add_297: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_296, mul_365);  add_296 = mul_365 = None
        mul_366: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_243, view_681);  view_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_888: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_366, [1, 128, 16, 32, 2]);  mul_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_14: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_888, -1, 0)
        select_15: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_888, -1, 1);  view_888 = None
        neg_68: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_14);  select_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_26: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_68, 3, 1, 9223372036854775807, 2);  neg_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_27: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_15, 3, 0, 9223372036854775807, 2);  select_15 = None
        add_298: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_26, slice_scatter_27);  slice_scatter_26 = slice_scatter_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_367: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_243, view_682);  slice_243 = view_682 = None
        add_299: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_298, mul_367);  add_298 = mul_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_28: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_242, 3, 64, 9223372036854775807);  slice_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_29: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_297, 3, 0, 64);  add_297 = None
        add_300: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_28, slice_scatter_29);  slice_scatter_28 = slice_scatter_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_30: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_244, 3, 64, 9223372036854775807);  slice_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_31: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_299, 3, 0, 64);  add_299 = None
        add_301: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_30, slice_scatter_31);  slice_scatter_30 = slice_scatter_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_432: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_882, [0, 2, 1, 3]);  view_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_228: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_432, memory_format = torch.contiguous_format);  permute_432 = None
        view_889: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_228, [1, 128, 4096]);  clone_228 = None
        view_890: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_301, [1, 128, 4096]);  add_301 = None
        view_891: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_300, [1, 128, 4096]);  add_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_892: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_889, [128, 4096]);  view_889 = None
        permute_433: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_892, [1, 0])
        mm_156: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_433, view_672);  permute_433 = None
        mm_157: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_892, permute_435);  view_892 = permute_435 = None
        view_893: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_157, [1, 128, 4096]);  mm_157 = None
        convert_element_type_1295: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_893, torch.float32);  view_893 = None
        add_302: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1269, convert_element_type_1295);  convert_element_type_1269 = convert_element_type_1295 = None
        convert_element_type_1296: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_156, torch.float32);  mm_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_894: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_890, [128, 4096]);  view_890 = None
        permute_437: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_894, [1, 0])
        mm_158: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_437, view_672);  permute_437 = None
        mm_159: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_894, permute_439);  view_894 = permute_439 = None
        view_895: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_159, [1, 128, 4096]);  mm_159 = None
        convert_element_type_1301: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_895, torch.float32);  view_895 = None
        add_303: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_302, convert_element_type_1301);  add_302 = convert_element_type_1301 = None
        convert_element_type_1302: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_158, torch.float32);  mm_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_896: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_891, [128, 4096]);  view_891 = None
        permute_441: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_896, [1, 0])
        mm_160: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_441, view_672);  permute_441 = view_672 = None
        mm_161: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_896, permute_443);  view_896 = permute_443 = None
        view_897: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_161, [1, 128, 4096]);  mm_161 = None
        convert_element_type_1307: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_897, torch.float32);  view_897 = None
        add_304: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_303, convert_element_type_1307);  add_303 = convert_element_type_1307 = None
        convert_element_type_1308: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_160, torch.float32);  mm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_369: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_304, primals_267);  primals_267 = None
        mul_370: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_369, 4096)
        sum_62: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True)
        mul_371: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_369, mul_240);  mul_369 = None
        sum_63: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True);  mul_371 = None
        mul_372: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_240, sum_63);  sum_63 = None
        sub_79: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_370, sum_62);  mul_370 = sum_62 = None
        sub_80: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_79, mul_372);  sub_79 = mul_372 = None
        mul_373: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_66, sub_80);  div_66 = sub_80 = None
        mul_374: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_304, mul_240);  mul_240 = None
        sum_64: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 1]);  mul_374 = None
        sum_65: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_304, [0, 1]);  add_304 = None
        add_305: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_293, mul_373);  add_293 = mul_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1309: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_305, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_898: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1309, [128, 4096]);  convert_element_type_1309 = None
        mm_162: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_898, permute_445);  permute_445 = None
        permute_446: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_898, [1, 0])
        mm_163: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_446, view_670);  view_670 = None
        sum_66: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_898, [0], True, dtype = torch.float32)
        view_899: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_66, [4096]);  sum_66 = None
        convert_element_type_1314: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_899, torch.bfloat16);  view_899 = None
        view_900: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_162, [1, 128, 16384]);  mm_162 = None
        convert_element_type_1315: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_900, torch.float32);  view_900 = None
        convert_element_type_1316: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_163, torch.float32);  mm_163 = None
        convert_element_type_1317: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1314, torch.float32);  convert_element_type_1314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_669: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [1, 128, 16384]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_236: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_669, 0.5)
        mul_375: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1315, mul_236);  mul_236 = None
        convert_element_type_905: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_669, torch.float32)
        pow_24: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_905, 3.0)
        mul_237: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_215: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_669, mul_237);  view_669 = mul_237 = None
        mul_238: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_215, 0.7978845608028654);  add_215 = None
        tanh_23: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_238);  mul_238 = None
        add_216: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_23, 1.0)
        mul_376: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1315, add_216);  convert_element_type_1315 = add_216 = None
        convert_element_type_1318: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_376, torch.bfloat16);  mul_376 = None
        mul_377: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_23, tanh_23);  tanh_23 = None
        sub_81: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_377);  mul_377 = None
        mul_378: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_375, sub_81);  mul_375 = sub_81 = None
        mul_379: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_378, 0.7978845608028654);  mul_378 = None
        convert_element_type_1319: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_379, torch.bfloat16)
        mul_380: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_379, 0.044715);  mul_379 = None
        pow_33: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_905, 2.0);  convert_element_type_905 = None
        mul_381: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_33, 3.0);  pow_33 = None
        mul_382: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_380, mul_381);  mul_380 = mul_381 = None
        convert_element_type_1320: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_382, torch.bfloat16);  mul_382 = None
        add_306: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1319, convert_element_type_1320);  convert_element_type_1319 = convert_element_type_1320 = None
        mul_383: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1318, 0.5);  convert_element_type_1318 = None
        add_307: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_306, mul_383);  add_306 = mul_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_901: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_307, [128, 16384]);  add_307 = None
        mm_164: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_901, permute_449);  permute_449 = None
        permute_450: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_901, [1, 0])
        mm_165: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_450, view_644);  permute_450 = None
        sum_67: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_901, [0], True, dtype = torch.float32);  view_901 = None
        view_902: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_67, [16384]);  sum_67 = None
        convert_element_type_1325: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_902, torch.bfloat16);  view_902 = None
        view_903: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_164, [1, 128, 4096]);  mm_164 = None
        convert_element_type_1326: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_903, torch.float32);  view_903 = None
        convert_element_type_1327: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_165, torch.float32);  mm_165 = None
        convert_element_type_1328: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1325, torch.float32);  convert_element_type_1325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_166: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_446, view_666);  permute_446 = view_666 = None
        mm_167: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_898, permute_455);  view_898 = permute_455 = None
        view_905: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_167, [1, 128, 4096]);  mm_167 = None
        convert_element_type_1333: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_166, torch.float32);  mm_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_906: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_905, [1, 128, 16, 256]);  view_905 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_457: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_906, [0, 2, 1, 3]);  view_906 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_907: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_457, [16, 128, 256]);  permute_457 = None
        bmm_72: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_458, view_907);  permute_458 = None
        bmm_73: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_907, permute_459);  view_907 = permute_459 = None
        view_908: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_72, [1, 16, 128, 256]);  bmm_72 = None
        view_909: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_73, [1, 16, 128, 128]);  bmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1338: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_909, torch.float32);  view_909 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_384: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1338, div_47);  convert_element_type_1338 = None
        sum_68: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_384, [-1], True)
        neg_69: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_47);  div_47 = None
        fma_4: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_69, sum_68, mul_384);  neg_69 = sum_68 = mul_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1339: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_67: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1339, 16.0);  convert_element_type_1339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_910: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_67, [16, 128, 128]);  div_67 = None
        bmm_74: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_460, view_910);  permute_460 = None
        bmm_75: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_910, permute_461);  view_910 = permute_461 = None
        view_911: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_74, [1, 16, 256, 128]);  bmm_74 = None
        view_912: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_75, [1, 16, 128, 256]);  bmm_75 = None
        convert_element_type_1345: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_911, torch.float32);  view_911 = None
        permute_462: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1345, [0, 1, 3, 2]);  convert_element_type_1345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1346: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_462, torch.bfloat16);  permute_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_23: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_912, torch.bfloat16);  view_912 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_463: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_23, [0, 2, 1, 3]);  convert_element_type_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_464: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1346, [0, 2, 1, 3]);  convert_element_type_1346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_245: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_463, 3, 0, 64)
        slice_246: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_463, 3, 64, 256);  permute_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_247: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_464, 3, 0, 64)
        slice_248: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_464, 3, 64, 256);  permute_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_185: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_311, [1, 128, 1, 32, 2]);  unsqueeze_311 = None
        clone_185: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
        view_653: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_185, [1, 128, 1, 64]);  clone_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_385: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_245, view_653)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_913: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_385, [1, 128, 16, 32, 2]);  mul_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_16: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_913, -1, 0)
        select_17: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_913, -1, 1);  view_913 = None
        neg_70: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_16);  select_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_32: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_70, 3, 1, 9223372036854775807, 2);  neg_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_33: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_17, 3, 0, 9223372036854775807, 2);  select_17 = None
        add_308: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_32, slice_scatter_33);  slice_scatter_32 = slice_scatter_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_186: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_313, [1, 128, 1, 32, 2]);  unsqueeze_313 = None
        clone_186: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_186, memory_format = torch.contiguous_format);  expand_186 = None
        view_654: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_186, [1, 128, 1, 64]);  clone_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_386: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_245, view_654);  slice_245 = None
        add_309: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_308, mul_386);  add_308 = mul_386 = None
        mul_387: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_247, view_653);  view_653 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_914: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_387, [1, 128, 16, 32, 2]);  mul_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_18: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_914, -1, 0)
        select_19: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_914, -1, 1);  view_914 = None
        neg_71: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_18);  select_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_34: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_71, 3, 1, 9223372036854775807, 2);  neg_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_35: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_19, 3, 0, 9223372036854775807, 2);  select_19 = None
        add_310: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_34, slice_scatter_35);  slice_scatter_34 = slice_scatter_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_388: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_247, view_654);  slice_247 = view_654 = None
        add_311: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_310, mul_388);  add_310 = mul_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_36: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_246, 3, 64, 9223372036854775807);  slice_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_37: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_309, 3, 0, 64);  add_309 = None
        add_312: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_36, slice_scatter_37);  slice_scatter_36 = slice_scatter_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_38: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_248, 3, 64, 9223372036854775807);  slice_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_39: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_311, 3, 0, 64);  add_311 = None
        add_313: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_38, slice_scatter_39);  slice_scatter_38 = slice_scatter_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_465: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_908, [0, 2, 1, 3]);  view_908 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_229: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_465, memory_format = torch.contiguous_format);  permute_465 = None
        view_915: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_229, [1, 128, 4096]);  clone_229 = None
        view_916: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_313, [1, 128, 4096]);  add_313 = None
        view_917: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_312, [1, 128, 4096]);  add_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_918: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_915, [128, 4096]);  view_915 = None
        permute_466: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_918, [1, 0])
        mm_168: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_466, view_644);  permute_466 = None
        mm_169: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_918, permute_468);  view_918 = permute_468 = None
        view_919: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_169, [1, 128, 4096]);  mm_169 = None
        convert_element_type_1352: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_919, torch.float32);  view_919 = None
        add_314: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1326, convert_element_type_1352);  convert_element_type_1326 = convert_element_type_1352 = None
        convert_element_type_1353: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_168, torch.float32);  mm_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_920: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_916, [128, 4096]);  view_916 = None
        permute_470: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_920, [1, 0])
        mm_170: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_470, view_644);  permute_470 = None
        mm_171: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_920, permute_472);  view_920 = permute_472 = None
        view_921: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_171, [1, 128, 4096]);  mm_171 = None
        convert_element_type_1358: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_921, torch.float32);  view_921 = None
        add_315: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_314, convert_element_type_1358);  add_314 = convert_element_type_1358 = None
        convert_element_type_1359: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_170, torch.float32);  mm_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_922: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_917, [128, 4096]);  view_917 = None
        permute_474: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_922, [1, 0])
        mm_172: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_474, view_644);  permute_474 = view_644 = None
        mm_173: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_922, permute_476);  view_922 = permute_476 = None
        view_923: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_173, [1, 128, 4096]);  mm_173 = None
        convert_element_type_1364: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_923, torch.float32);  view_923 = None
        add_316: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_315, convert_element_type_1364);  add_315 = convert_element_type_1364 = None
        convert_element_type_1365: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_172, torch.float32);  mm_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_390: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_316, primals_256);  primals_256 = None
        mul_391: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_390, 4096)
        sum_69: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_390, [2], True)
        mul_392: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_390, mul_230);  mul_390 = None
        sum_70: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_392, [2], True);  mul_392 = None
        mul_393: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, sum_70);  sum_70 = None
        sub_83: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_391, sum_69);  mul_391 = sum_69 = None
        sub_84: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_83, mul_393);  sub_83 = mul_393 = None
        mul_394: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_68, sub_84);  div_68 = sub_84 = None
        mul_395: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_316, mul_230);  mul_230 = None
        sum_71: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_395, [0, 1]);  mul_395 = None
        sum_72: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_316, [0, 1]);  add_316 = None
        add_317: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_305, mul_394);  add_305 = mul_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1366: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_317, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_924: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1366, [128, 4096]);  convert_element_type_1366 = None
        mm_174: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_924, permute_478);  permute_478 = None
        permute_479: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_924, [1, 0])
        mm_175: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_479, view_642);  view_642 = None
        sum_73: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_924, [0], True, dtype = torch.float32)
        view_925: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_73, [4096]);  sum_73 = None
        convert_element_type_1371: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_925, torch.bfloat16);  view_925 = None
        view_926: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_174, [1, 128, 16384]);  mm_174 = None
        convert_element_type_1372: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_926, torch.float32);  view_926 = None
        convert_element_type_1373: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_175, torch.float32);  mm_175 = None
        convert_element_type_1374: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1371, torch.float32);  convert_element_type_1371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_641: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [1, 128, 16384]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_226: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_641, 0.5)
        mul_396: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1372, mul_226);  mul_226 = None
        convert_element_type_867: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_641, torch.float32)
        pow_23: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_867, 3.0)
        mul_227: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_206: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_641, mul_227);  view_641 = mul_227 = None
        mul_228: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_206, 0.7978845608028654);  add_206 = None
        tanh_22: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_228);  mul_228 = None
        add_207: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_22, 1.0)
        mul_397: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1372, add_207);  convert_element_type_1372 = add_207 = None
        convert_element_type_1375: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_397, torch.bfloat16);  mul_397 = None
        mul_398: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_22, tanh_22);  tanh_22 = None
        sub_85: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_398);  mul_398 = None
        mul_399: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_396, sub_85);  mul_396 = sub_85 = None
        mul_400: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_399, 0.7978845608028654);  mul_399 = None
        convert_element_type_1376: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_400, torch.bfloat16)
        mul_401: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_400, 0.044715);  mul_400 = None
        pow_34: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_867, 2.0);  convert_element_type_867 = None
        mul_402: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_34, 3.0);  pow_34 = None
        mul_403: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_401, mul_402);  mul_401 = mul_402 = None
        convert_element_type_1377: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_403, torch.bfloat16);  mul_403 = None
        add_318: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1376, convert_element_type_1377);  convert_element_type_1376 = convert_element_type_1377 = None
        mul_404: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1375, 0.5);  convert_element_type_1375 = None
        add_319: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_318, mul_404);  add_318 = mul_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_927: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_319, [128, 16384]);  add_319 = None
        mm_176: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_927, permute_482);  permute_482 = None
        permute_483: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_927, [1, 0])
        mm_177: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_483, view_616);  permute_483 = None
        sum_74: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_927, [0], True, dtype = torch.float32);  view_927 = None
        view_928: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_74, [16384]);  sum_74 = None
        convert_element_type_1382: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_928, torch.bfloat16);  view_928 = None
        view_929: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_176, [1, 128, 4096]);  mm_176 = None
        convert_element_type_1383: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_929, torch.float32);  view_929 = None
        convert_element_type_1384: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_177, torch.float32);  mm_177 = None
        convert_element_type_1385: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1382, torch.float32);  convert_element_type_1382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_178: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_479, view_638);  permute_479 = view_638 = None
        mm_179: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_924, permute_488);  view_924 = permute_488 = None
        view_931: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_179, [1, 128, 4096]);  mm_179 = None
        convert_element_type_1390: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_178, torch.float32);  mm_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_932: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_931, [1, 128, 16, 256]);  view_931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_490: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_932, [0, 2, 1, 3]);  view_932 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_933: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_490, [16, 128, 256]);  permute_490 = None
        bmm_76: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_491, view_933);  permute_491 = None
        bmm_77: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_933, permute_492);  view_933 = permute_492 = None
        view_934: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_76, [1, 16, 128, 256]);  bmm_76 = None
        view_935: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_77, [1, 16, 128, 128]);  bmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1395: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_935, torch.float32);  view_935 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_405: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1395, div_45);  convert_element_type_1395 = None
        sum_75: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_405, [-1], True)
        neg_72: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_45);  div_45 = None
        fma_5: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_72, sum_75, mul_405);  neg_72 = sum_75 = mul_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1396: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_69: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1396, 16.0);  convert_element_type_1396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_936: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_69, [16, 128, 128]);  div_69 = None
        bmm_78: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_493, view_936);  permute_493 = None
        bmm_79: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_936, permute_494);  view_936 = permute_494 = None
        view_937: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_78, [1, 16, 256, 128]);  bmm_78 = None
        view_938: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_79, [1, 16, 128, 256]);  bmm_79 = None
        convert_element_type_1402: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_937, torch.float32);  view_937 = None
        permute_495: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1402, [0, 1, 3, 2]);  convert_element_type_1402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1403: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_495, torch.bfloat16);  permute_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_22: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_938, torch.bfloat16);  view_938 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_496: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_22, [0, 2, 1, 3]);  convert_element_type_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_497: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1403, [0, 2, 1, 3]);  convert_element_type_1403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_249: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_496, 3, 0, 64)
        slice_250: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_496, 3, 64, 256);  permute_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_251: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_497, 3, 0, 64)
        slice_252: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_497, 3, 64, 256);  permute_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_177: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_298, [1, 128, 1, 32, 2]);  unsqueeze_298 = None
        clone_177: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
        view_625: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_177, [1, 128, 1, 64]);  clone_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_406: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_249, view_625)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_939: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_406, [1, 128, 16, 32, 2]);  mul_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_20: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_939, -1, 0)
        select_21: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_939, -1, 1);  view_939 = None
        neg_73: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_20);  select_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_40: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_73, 3, 1, 9223372036854775807, 2);  neg_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_41: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_21, 3, 0, 9223372036854775807, 2);  select_21 = None
        add_320: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_40, slice_scatter_41);  slice_scatter_40 = slice_scatter_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_178: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_300, [1, 128, 1, 32, 2]);  unsqueeze_300 = None
        clone_178: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_178, memory_format = torch.contiguous_format);  expand_178 = None
        view_626: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_178, [1, 128, 1, 64]);  clone_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_407: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_249, view_626);  slice_249 = None
        add_321: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_320, mul_407);  add_320 = mul_407 = None
        mul_408: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_251, view_625);  view_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_940: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_408, [1, 128, 16, 32, 2]);  mul_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_22: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_940, -1, 0)
        select_23: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_940, -1, 1);  view_940 = None
        neg_74: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_22);  select_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_42: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_74, 3, 1, 9223372036854775807, 2);  neg_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_43: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_23, 3, 0, 9223372036854775807, 2);  select_23 = None
        add_322: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_42, slice_scatter_43);  slice_scatter_42 = slice_scatter_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_409: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_251, view_626);  slice_251 = view_626 = None
        add_323: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_322, mul_409);  add_322 = mul_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_44: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_250, 3, 64, 9223372036854775807);  slice_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_45: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_321, 3, 0, 64);  add_321 = None
        add_324: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_44, slice_scatter_45);  slice_scatter_44 = slice_scatter_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_46: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_252, 3, 64, 9223372036854775807);  slice_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_47: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_323, 3, 0, 64);  add_323 = None
        add_325: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_46, slice_scatter_47);  slice_scatter_46 = slice_scatter_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_498: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_934, [0, 2, 1, 3]);  view_934 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_230: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_498, memory_format = torch.contiguous_format);  permute_498 = None
        view_941: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_230, [1, 128, 4096]);  clone_230 = None
        view_942: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_325, [1, 128, 4096]);  add_325 = None
        view_943: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_324, [1, 128, 4096]);  add_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_944: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_941, [128, 4096]);  view_941 = None
        permute_499: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_944, [1, 0])
        mm_180: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_499, view_616);  permute_499 = None
        mm_181: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_944, permute_501);  view_944 = permute_501 = None
        view_945: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_181, [1, 128, 4096]);  mm_181 = None
        convert_element_type_1409: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_945, torch.float32);  view_945 = None
        add_326: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1383, convert_element_type_1409);  convert_element_type_1383 = convert_element_type_1409 = None
        convert_element_type_1410: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_180, torch.float32);  mm_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_946: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_942, [128, 4096]);  view_942 = None
        permute_503: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_946, [1, 0])
        mm_182: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_503, view_616);  permute_503 = None
        mm_183: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_946, permute_505);  view_946 = permute_505 = None
        view_947: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_183, [1, 128, 4096]);  mm_183 = None
        convert_element_type_1415: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_947, torch.float32);  view_947 = None
        add_327: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_326, convert_element_type_1415);  add_326 = convert_element_type_1415 = None
        convert_element_type_1416: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_182, torch.float32);  mm_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_948: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_943, [128, 4096]);  view_943 = None
        permute_507: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_948, [1, 0])
        mm_184: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_507, view_616);  permute_507 = view_616 = None
        mm_185: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_948, permute_509);  view_948 = permute_509 = None
        view_949: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_185, [1, 128, 4096]);  mm_185 = None
        convert_element_type_1421: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_949, torch.float32);  view_949 = None
        add_328: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_327, convert_element_type_1421);  add_327 = convert_element_type_1421 = None
        convert_element_type_1422: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_184, torch.float32);  mm_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_411: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_328, primals_245);  primals_245 = None
        mul_412: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_411, 4096)
        sum_76: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True)
        mul_413: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_411, mul_220);  mul_411 = None
        sum_77: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
        mul_414: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_220, sum_77);  sum_77 = None
        sub_87: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_412, sum_76);  mul_412 = sum_76 = None
        sub_88: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_87, mul_414);  sub_87 = mul_414 = None
        mul_415: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_70, sub_88);  div_70 = sub_88 = None
        mul_416: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_328, mul_220);  mul_220 = None
        sum_78: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1]);  mul_416 = None
        sum_79: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_328, [0, 1]);  add_328 = None
        add_329: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_317, mul_415);  add_317 = mul_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1423: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_329, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_950: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1423, [128, 4096]);  convert_element_type_1423 = None
        mm_186: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_950, permute_511);  permute_511 = None
        permute_512: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_950, [1, 0])
        mm_187: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_512, view_614);  view_614 = None
        sum_80: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_950, [0], True, dtype = torch.float32)
        view_951: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_80, [4096]);  sum_80 = None
        convert_element_type_1428: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_951, torch.bfloat16);  view_951 = None
        view_952: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_186, [1, 128, 16384]);  mm_186 = None
        convert_element_type_1429: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_952, torch.float32);  view_952 = None
        convert_element_type_1430: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_187, torch.float32);  mm_187 = None
        convert_element_type_1431: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1428, torch.float32);  convert_element_type_1428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_613: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [1, 128, 16384]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_216: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_613, 0.5)
        mul_417: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1429, mul_216);  mul_216 = None
        convert_element_type_829: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_613, torch.float32)
        pow_22: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_829, 3.0)
        mul_217: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_197: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_613, mul_217);  view_613 = mul_217 = None
        mul_218: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_197, 0.7978845608028654);  add_197 = None
        tanh_21: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_218);  mul_218 = None
        add_198: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_21, 1.0)
        mul_418: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1429, add_198);  convert_element_type_1429 = add_198 = None
        convert_element_type_1432: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_418, torch.bfloat16);  mul_418 = None
        mul_419: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_21, tanh_21);  tanh_21 = None
        sub_89: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_419);  mul_419 = None
        mul_420: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_417, sub_89);  mul_417 = sub_89 = None
        mul_421: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_420, 0.7978845608028654);  mul_420 = None
        convert_element_type_1433: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_421, torch.bfloat16)
        mul_422: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_421, 0.044715);  mul_421 = None
        pow_35: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_829, 2.0);  convert_element_type_829 = None
        mul_423: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_35, 3.0);  pow_35 = None
        mul_424: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_422, mul_423);  mul_422 = mul_423 = None
        convert_element_type_1434: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_424, torch.bfloat16);  mul_424 = None
        add_330: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1433, convert_element_type_1434);  convert_element_type_1433 = convert_element_type_1434 = None
        mul_425: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1432, 0.5);  convert_element_type_1432 = None
        add_331: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_330, mul_425);  add_330 = mul_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_953: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_331, [128, 16384]);  add_331 = None
        mm_188: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_953, permute_515);  permute_515 = None
        permute_516: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_953, [1, 0])
        mm_189: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_516, view_588);  permute_516 = None
        sum_81: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_953, [0], True, dtype = torch.float32);  view_953 = None
        view_954: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_81, [16384]);  sum_81 = None
        convert_element_type_1439: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_954, torch.bfloat16);  view_954 = None
        view_955: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_188, [1, 128, 4096]);  mm_188 = None
        convert_element_type_1440: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_955, torch.float32);  view_955 = None
        convert_element_type_1441: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_189, torch.float32);  mm_189 = None
        convert_element_type_1442: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1439, torch.float32);  convert_element_type_1439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_190: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_512, view_610);  permute_512 = view_610 = None
        mm_191: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_950, permute_521);  view_950 = permute_521 = None
        view_957: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_191, [1, 128, 4096]);  mm_191 = None
        convert_element_type_1447: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_190, torch.float32);  mm_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_958: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_957, [1, 128, 16, 256]);  view_957 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_523: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_958, [0, 2, 1, 3]);  view_958 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_959: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_523, [16, 128, 256]);  permute_523 = None
        bmm_80: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_524, view_959);  permute_524 = None
        bmm_81: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_959, permute_525);  view_959 = permute_525 = None
        view_960: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_80, [1, 16, 128, 256]);  bmm_80 = None
        view_961: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_81, [1, 16, 128, 128]);  bmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1452: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_961, torch.float32);  view_961 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_426: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1452, div_43);  convert_element_type_1452 = None
        sum_82: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_426, [-1], True)
        neg_75: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_43);  div_43 = None
        fma_6: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_75, sum_82, mul_426);  neg_75 = sum_82 = mul_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1453: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_71: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1453, 16.0);  convert_element_type_1453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_962: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_71, [16, 128, 128]);  div_71 = None
        bmm_82: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_526, view_962);  permute_526 = None
        bmm_83: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_962, permute_527);  view_962 = permute_527 = None
        view_963: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_82, [1, 16, 256, 128]);  bmm_82 = None
        view_964: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_83, [1, 16, 128, 256]);  bmm_83 = None
        convert_element_type_1459: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_963, torch.float32);  view_963 = None
        permute_528: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1459, [0, 1, 3, 2]);  convert_element_type_1459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1460: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_528, torch.bfloat16);  permute_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_21: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_964, torch.bfloat16);  view_964 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_529: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_21, [0, 2, 1, 3]);  convert_element_type_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_530: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1460, [0, 2, 1, 3]);  convert_element_type_1460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_253: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_529, 3, 0, 64)
        slice_254: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_529, 3, 64, 256);  permute_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_255: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_530, 3, 0, 64)
        slice_256: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_530, 3, 64, 256);  permute_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_169: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_285, [1, 128, 1, 32, 2]);  unsqueeze_285 = None
        clone_169: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
        view_597: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_169, [1, 128, 1, 64]);  clone_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_427: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_253, view_597)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_965: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_427, [1, 128, 16, 32, 2]);  mul_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_24: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_965, -1, 0)
        select_25: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_965, -1, 1);  view_965 = None
        neg_76: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_24);  select_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_48: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_76, 3, 1, 9223372036854775807, 2);  neg_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_49: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_25, 3, 0, 9223372036854775807, 2);  select_25 = None
        add_332: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_48, slice_scatter_49);  slice_scatter_48 = slice_scatter_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_170: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_287, [1, 128, 1, 32, 2]);  unsqueeze_287 = None
        clone_170: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_170, memory_format = torch.contiguous_format);  expand_170 = None
        view_598: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_170, [1, 128, 1, 64]);  clone_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_428: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_253, view_598);  slice_253 = None
        add_333: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_332, mul_428);  add_332 = mul_428 = None
        mul_429: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_255, view_597);  view_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_966: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_429, [1, 128, 16, 32, 2]);  mul_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_26: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_966, -1, 0)
        select_27: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_966, -1, 1);  view_966 = None
        neg_77: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_26);  select_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_50: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_77, 3, 1, 9223372036854775807, 2);  neg_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_51: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_27, 3, 0, 9223372036854775807, 2);  select_27 = None
        add_334: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_50, slice_scatter_51);  slice_scatter_50 = slice_scatter_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_430: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_255, view_598);  slice_255 = view_598 = None
        add_335: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_334, mul_430);  add_334 = mul_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_52: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_254, 3, 64, 9223372036854775807);  slice_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_53: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_333, 3, 0, 64);  add_333 = None
        add_336: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_52, slice_scatter_53);  slice_scatter_52 = slice_scatter_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_54: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_256, 3, 64, 9223372036854775807);  slice_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_55: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_335, 3, 0, 64);  add_335 = None
        add_337: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_54, slice_scatter_55);  slice_scatter_54 = slice_scatter_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_531: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_960, [0, 2, 1, 3]);  view_960 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_231: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_531, memory_format = torch.contiguous_format);  permute_531 = None
        view_967: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_231, [1, 128, 4096]);  clone_231 = None
        view_968: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_337, [1, 128, 4096]);  add_337 = None
        view_969: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_336, [1, 128, 4096]);  add_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_970: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_967, [128, 4096]);  view_967 = None
        permute_532: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_970, [1, 0])
        mm_192: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_532, view_588);  permute_532 = None
        mm_193: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_970, permute_534);  view_970 = permute_534 = None
        view_971: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_193, [1, 128, 4096]);  mm_193 = None
        convert_element_type_1466: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_971, torch.float32);  view_971 = None
        add_338: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1440, convert_element_type_1466);  convert_element_type_1440 = convert_element_type_1466 = None
        convert_element_type_1467: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_192, torch.float32);  mm_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_972: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_968, [128, 4096]);  view_968 = None
        permute_536: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_972, [1, 0])
        mm_194: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_536, view_588);  permute_536 = None
        mm_195: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_972, permute_538);  view_972 = permute_538 = None
        view_973: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_195, [1, 128, 4096]);  mm_195 = None
        convert_element_type_1472: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_973, torch.float32);  view_973 = None
        add_339: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_338, convert_element_type_1472);  add_338 = convert_element_type_1472 = None
        convert_element_type_1473: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_194, torch.float32);  mm_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_974: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_969, [128, 4096]);  view_969 = None
        permute_540: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_974, [1, 0])
        mm_196: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_540, view_588);  permute_540 = view_588 = None
        mm_197: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_974, permute_542);  view_974 = permute_542 = None
        view_975: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_197, [1, 128, 4096]);  mm_197 = None
        convert_element_type_1478: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_975, torch.float32);  view_975 = None
        add_340: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_339, convert_element_type_1478);  add_339 = convert_element_type_1478 = None
        convert_element_type_1479: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_196, torch.float32);  mm_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_432: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_340, primals_234);  primals_234 = None
        mul_433: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_432, 4096)
        sum_83: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_432, [2], True)
        mul_434: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_432, mul_210);  mul_432 = None
        sum_84: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_434, [2], True);  mul_434 = None
        mul_435: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, sum_84);  sum_84 = None
        sub_91: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_433, sum_83);  mul_433 = sum_83 = None
        sub_92: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_91, mul_435);  sub_91 = mul_435 = None
        mul_436: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_72, sub_92);  div_72 = sub_92 = None
        mul_437: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_340, mul_210);  mul_210 = None
        sum_85: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 1]);  mul_437 = None
        sum_86: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_340, [0, 1]);  add_340 = None
        add_341: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_329, mul_436);  add_329 = mul_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1480: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_341, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_976: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1480, [128, 4096]);  convert_element_type_1480 = None
        mm_198: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_976, permute_544);  permute_544 = None
        permute_545: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_976, [1, 0])
        mm_199: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_545, view_586);  view_586 = None
        sum_87: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_976, [0], True, dtype = torch.float32)
        view_977: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_87, [4096]);  sum_87 = None
        convert_element_type_1485: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_977, torch.bfloat16);  view_977 = None
        view_978: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_198, [1, 128, 16384]);  mm_198 = None
        convert_element_type_1486: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_978, torch.float32);  view_978 = None
        convert_element_type_1487: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_199, torch.float32);  mm_199 = None
        convert_element_type_1488: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1485, torch.float32);  convert_element_type_1485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_585: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [1, 128, 16384]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_206: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_585, 0.5)
        mul_438: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1486, mul_206);  mul_206 = None
        convert_element_type_791: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_585, torch.float32)
        pow_21: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_791, 3.0)
        mul_207: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_188: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_585, mul_207);  view_585 = mul_207 = None
        mul_208: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_188, 0.7978845608028654);  add_188 = None
        tanh_20: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_208);  mul_208 = None
        add_189: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_20, 1.0)
        mul_439: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1486, add_189);  convert_element_type_1486 = add_189 = None
        convert_element_type_1489: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_439, torch.bfloat16);  mul_439 = None
        mul_440: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_20, tanh_20);  tanh_20 = None
        sub_93: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_440);  mul_440 = None
        mul_441: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_438, sub_93);  mul_438 = sub_93 = None
        mul_442: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_441, 0.7978845608028654);  mul_441 = None
        convert_element_type_1490: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_442, torch.bfloat16)
        mul_443: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_442, 0.044715);  mul_442 = None
        pow_36: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_791, 2.0);  convert_element_type_791 = None
        mul_444: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_36, 3.0);  pow_36 = None
        mul_445: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_443, mul_444);  mul_443 = mul_444 = None
        convert_element_type_1491: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_445, torch.bfloat16);  mul_445 = None
        add_342: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1490, convert_element_type_1491);  convert_element_type_1490 = convert_element_type_1491 = None
        mul_446: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1489, 0.5);  convert_element_type_1489 = None
        add_343: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_342, mul_446);  add_342 = mul_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_979: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_343, [128, 16384]);  add_343 = None
        mm_200: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_979, permute_548);  permute_548 = None
        permute_549: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_979, [1, 0])
        mm_201: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_549, view_560);  permute_549 = None
        sum_88: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_979, [0], True, dtype = torch.float32);  view_979 = None
        view_980: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_88, [16384]);  sum_88 = None
        convert_element_type_1496: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_980, torch.bfloat16);  view_980 = None
        view_981: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_200, [1, 128, 4096]);  mm_200 = None
        convert_element_type_1497: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_981, torch.float32);  view_981 = None
        convert_element_type_1498: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_201, torch.float32);  mm_201 = None
        convert_element_type_1499: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1496, torch.float32);  convert_element_type_1496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_202: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_545, view_582);  permute_545 = view_582 = None
        mm_203: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_976, permute_554);  view_976 = permute_554 = None
        view_983: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_203, [1, 128, 4096]);  mm_203 = None
        convert_element_type_1504: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_202, torch.float32);  mm_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_984: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_983, [1, 128, 16, 256]);  view_983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_556: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_984, [0, 2, 1, 3]);  view_984 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_985: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_556, [16, 128, 256]);  permute_556 = None
        bmm_84: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_557, view_985);  permute_557 = None
        bmm_85: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_985, permute_558);  view_985 = permute_558 = None
        view_986: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_84, [1, 16, 128, 256]);  bmm_84 = None
        view_987: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_85, [1, 16, 128, 128]);  bmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1509: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_987, torch.float32);  view_987 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_447: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1509, div_41);  convert_element_type_1509 = None
        sum_89: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_447, [-1], True)
        neg_78: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_41);  div_41 = None
        fma_7: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_78, sum_89, mul_447);  neg_78 = sum_89 = mul_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1510: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_73: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1510, 16.0);  convert_element_type_1510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_988: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_73, [16, 128, 128]);  div_73 = None
        bmm_86: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_559, view_988);  permute_559 = None
        bmm_87: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_988, permute_560);  view_988 = permute_560 = None
        view_989: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_86, [1, 16, 256, 128]);  bmm_86 = None
        view_990: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_87, [1, 16, 128, 256]);  bmm_87 = None
        convert_element_type_1516: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_989, torch.float32);  view_989 = None
        permute_561: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1516, [0, 1, 3, 2]);  convert_element_type_1516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1517: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_561, torch.bfloat16);  permute_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_20: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_990, torch.bfloat16);  view_990 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_562: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_20, [0, 2, 1, 3]);  convert_element_type_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_563: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1517, [0, 2, 1, 3]);  convert_element_type_1517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_257: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_562, 3, 0, 64)
        slice_258: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_562, 3, 64, 256);  permute_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_259: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_563, 3, 0, 64)
        slice_260: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_563, 3, 64, 256);  permute_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_161: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_272, [1, 128, 1, 32, 2]);  unsqueeze_272 = None
        clone_161: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
        view_569: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_161, [1, 128, 1, 64]);  clone_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_448: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_257, view_569)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_991: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_448, [1, 128, 16, 32, 2]);  mul_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_28: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_991, -1, 0)
        select_29: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_991, -1, 1);  view_991 = None
        neg_79: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_28);  select_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_56: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_79, 3, 1, 9223372036854775807, 2);  neg_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_57: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_29, 3, 0, 9223372036854775807, 2);  select_29 = None
        add_344: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_56, slice_scatter_57);  slice_scatter_56 = slice_scatter_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_162: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_274, [1, 128, 1, 32, 2]);  unsqueeze_274 = None
        clone_162: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_162, memory_format = torch.contiguous_format);  expand_162 = None
        view_570: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_162, [1, 128, 1, 64]);  clone_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_449: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_257, view_570);  slice_257 = None
        add_345: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_344, mul_449);  add_344 = mul_449 = None
        mul_450: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_259, view_569);  view_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_992: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_450, [1, 128, 16, 32, 2]);  mul_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_30: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_992, -1, 0)
        select_31: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_992, -1, 1);  view_992 = None
        neg_80: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_30);  select_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_58: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_80, 3, 1, 9223372036854775807, 2);  neg_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_59: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_31, 3, 0, 9223372036854775807, 2);  select_31 = None
        add_346: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_58, slice_scatter_59);  slice_scatter_58 = slice_scatter_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_451: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_259, view_570);  slice_259 = view_570 = None
        add_347: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_346, mul_451);  add_346 = mul_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_60: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_258, 3, 64, 9223372036854775807);  slice_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_61: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_345, 3, 0, 64);  add_345 = None
        add_348: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_60, slice_scatter_61);  slice_scatter_60 = slice_scatter_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_62: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_260, 3, 64, 9223372036854775807);  slice_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_63: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_347, 3, 0, 64);  add_347 = None
        add_349: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_62, slice_scatter_63);  slice_scatter_62 = slice_scatter_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_564: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_986, [0, 2, 1, 3]);  view_986 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_232: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_564, memory_format = torch.contiguous_format);  permute_564 = None
        view_993: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_232, [1, 128, 4096]);  clone_232 = None
        view_994: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_349, [1, 128, 4096]);  add_349 = None
        view_995: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_348, [1, 128, 4096]);  add_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_996: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_993, [128, 4096]);  view_993 = None
        permute_565: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_996, [1, 0])
        mm_204: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_565, view_560);  permute_565 = None
        mm_205: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_996, permute_567);  view_996 = permute_567 = None
        view_997: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_205, [1, 128, 4096]);  mm_205 = None
        convert_element_type_1523: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_997, torch.float32);  view_997 = None
        add_350: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1497, convert_element_type_1523);  convert_element_type_1497 = convert_element_type_1523 = None
        convert_element_type_1524: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_204, torch.float32);  mm_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_998: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_994, [128, 4096]);  view_994 = None
        permute_569: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_998, [1, 0])
        mm_206: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_569, view_560);  permute_569 = None
        mm_207: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_998, permute_571);  view_998 = permute_571 = None
        view_999: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_207, [1, 128, 4096]);  mm_207 = None
        convert_element_type_1529: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_999, torch.float32);  view_999 = None
        add_351: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_350, convert_element_type_1529);  add_350 = convert_element_type_1529 = None
        convert_element_type_1530: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_206, torch.float32);  mm_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1000: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_995, [128, 4096]);  view_995 = None
        permute_573: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1000, [1, 0])
        mm_208: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_573, view_560);  permute_573 = view_560 = None
        mm_209: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1000, permute_575);  view_1000 = permute_575 = None
        view_1001: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_209, [1, 128, 4096]);  mm_209 = None
        convert_element_type_1535: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1001, torch.float32);  view_1001 = None
        add_352: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_351, convert_element_type_1535);  add_351 = convert_element_type_1535 = None
        convert_element_type_1536: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_208, torch.float32);  mm_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_453: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_352, primals_223);  primals_223 = None
        mul_454: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_453, 4096)
        sum_90: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True)
        mul_455: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_453, mul_200);  mul_453 = None
        sum_91: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_455, [2], True);  mul_455 = None
        mul_456: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_200, sum_91);  sum_91 = None
        sub_95: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_454, sum_90);  mul_454 = sum_90 = None
        sub_96: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_95, mul_456);  sub_95 = mul_456 = None
        mul_457: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_74, sub_96);  div_74 = sub_96 = None
        mul_458: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_352, mul_200);  mul_200 = None
        sum_92: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 1]);  mul_458 = None
        sum_93: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_352, [0, 1]);  add_352 = None
        add_353: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_341, mul_457);  add_341 = mul_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1537: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_353, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1002: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1537, [128, 4096]);  convert_element_type_1537 = None
        mm_210: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1002, permute_577);  permute_577 = None
        permute_578: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1002, [1, 0])
        mm_211: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_578, view_558);  view_558 = None
        sum_94: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1002, [0], True, dtype = torch.float32)
        view_1003: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_94, [4096]);  sum_94 = None
        convert_element_type_1542: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1003, torch.bfloat16);  view_1003 = None
        view_1004: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_210, [1, 128, 16384]);  mm_210 = None
        convert_element_type_1543: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1004, torch.float32);  view_1004 = None
        convert_element_type_1544: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_211, torch.float32);  mm_211 = None
        convert_element_type_1545: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1542, torch.float32);  convert_element_type_1542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_557: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [1, 128, 16384]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_196: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_557, 0.5)
        mul_459: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1543, mul_196);  mul_196 = None
        convert_element_type_753: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_557, torch.float32)
        pow_20: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_753, 3.0)
        mul_197: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_179: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_557, mul_197);  view_557 = mul_197 = None
        mul_198: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_179, 0.7978845608028654);  add_179 = None
        tanh_19: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_198);  mul_198 = None
        add_180: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_19, 1.0)
        mul_460: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1543, add_180);  convert_element_type_1543 = add_180 = None
        convert_element_type_1546: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_460, torch.bfloat16);  mul_460 = None
        mul_461: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_19, tanh_19);  tanh_19 = None
        sub_97: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_461);  mul_461 = None
        mul_462: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_459, sub_97);  mul_459 = sub_97 = None
        mul_463: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_462, 0.7978845608028654);  mul_462 = None
        convert_element_type_1547: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_463, torch.bfloat16)
        mul_464: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_463, 0.044715);  mul_463 = None
        pow_37: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_753, 2.0);  convert_element_type_753 = None
        mul_465: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_37, 3.0);  pow_37 = None
        mul_466: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_464, mul_465);  mul_464 = mul_465 = None
        convert_element_type_1548: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_466, torch.bfloat16);  mul_466 = None
        add_354: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1547, convert_element_type_1548);  convert_element_type_1547 = convert_element_type_1548 = None
        mul_467: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1546, 0.5);  convert_element_type_1546 = None
        add_355: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_354, mul_467);  add_354 = mul_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1005: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_355, [128, 16384]);  add_355 = None
        mm_212: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1005, permute_581);  permute_581 = None
        permute_582: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1005, [1, 0])
        mm_213: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_582, view_532);  permute_582 = None
        sum_95: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1005, [0], True, dtype = torch.float32);  view_1005 = None
        view_1006: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_95, [16384]);  sum_95 = None
        convert_element_type_1553: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1006, torch.bfloat16);  view_1006 = None
        view_1007: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_212, [1, 128, 4096]);  mm_212 = None
        convert_element_type_1554: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1007, torch.float32);  view_1007 = None
        convert_element_type_1555: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_213, torch.float32);  mm_213 = None
        convert_element_type_1556: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1553, torch.float32);  convert_element_type_1553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_214: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_578, view_554);  permute_578 = view_554 = None
        mm_215: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1002, permute_587);  view_1002 = permute_587 = None
        view_1009: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_215, [1, 128, 4096]);  mm_215 = None
        convert_element_type_1561: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_214, torch.float32);  mm_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1010: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1009, [1, 128, 16, 256]);  view_1009 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_589: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1010, [0, 2, 1, 3]);  view_1010 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1011: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_589, [16, 128, 256]);  permute_589 = None
        bmm_88: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_590, view_1011);  permute_590 = None
        bmm_89: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1011, permute_591);  view_1011 = permute_591 = None
        view_1012: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_88, [1, 16, 128, 256]);  bmm_88 = None
        view_1013: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_89, [1, 16, 128, 128]);  bmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1566: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1013, torch.float32);  view_1013 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_468: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1566, div_39);  convert_element_type_1566 = None
        sum_96: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_468, [-1], True)
        neg_81: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_39);  div_39 = None
        fma_8: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_81, sum_96, mul_468);  neg_81 = sum_96 = mul_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1567: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_75: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1567, 16.0);  convert_element_type_1567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1014: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_75, [16, 128, 128]);  div_75 = None
        bmm_90: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_592, view_1014);  permute_592 = None
        bmm_91: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1014, permute_593);  view_1014 = permute_593 = None
        view_1015: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_90, [1, 16, 256, 128]);  bmm_90 = None
        view_1016: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_91, [1, 16, 128, 256]);  bmm_91 = None
        convert_element_type_1573: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1015, torch.float32);  view_1015 = None
        permute_594: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1573, [0, 1, 3, 2]);  convert_element_type_1573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1574: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_594, torch.bfloat16);  permute_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_19: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1016, torch.bfloat16);  view_1016 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_595: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_19, [0, 2, 1, 3]);  convert_element_type_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_596: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1574, [0, 2, 1, 3]);  convert_element_type_1574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_261: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_595, 3, 0, 64)
        slice_262: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_595, 3, 64, 256);  permute_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_263: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_596, 3, 0, 64)
        slice_264: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_596, 3, 64, 256);  permute_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_153: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_259, [1, 128, 1, 32, 2]);  unsqueeze_259 = None
        clone_153: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
        view_541: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_153, [1, 128, 1, 64]);  clone_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_469: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_261, view_541)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1017: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_469, [1, 128, 16, 32, 2]);  mul_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_32: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1017, -1, 0)
        select_33: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1017, -1, 1);  view_1017 = None
        neg_82: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_32);  select_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_64: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_82, 3, 1, 9223372036854775807, 2);  neg_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_65: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_33, 3, 0, 9223372036854775807, 2);  select_33 = None
        add_356: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_64, slice_scatter_65);  slice_scatter_64 = slice_scatter_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_154: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_261, [1, 128, 1, 32, 2]);  unsqueeze_261 = None
        clone_154: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_154, memory_format = torch.contiguous_format);  expand_154 = None
        view_542: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_154, [1, 128, 1, 64]);  clone_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_470: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_261, view_542);  slice_261 = None
        add_357: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_356, mul_470);  add_356 = mul_470 = None
        mul_471: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_263, view_541);  view_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1018: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_471, [1, 128, 16, 32, 2]);  mul_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_34: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1018, -1, 0)
        select_35: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1018, -1, 1);  view_1018 = None
        neg_83: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_34);  select_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_66: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_83, 3, 1, 9223372036854775807, 2);  neg_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_67: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_35, 3, 0, 9223372036854775807, 2);  select_35 = None
        add_358: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_66, slice_scatter_67);  slice_scatter_66 = slice_scatter_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_472: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_263, view_542);  slice_263 = view_542 = None
        add_359: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_358, mul_472);  add_358 = mul_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_68: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_262, 3, 64, 9223372036854775807);  slice_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_69: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_357, 3, 0, 64);  add_357 = None
        add_360: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_68, slice_scatter_69);  slice_scatter_68 = slice_scatter_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_70: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_264, 3, 64, 9223372036854775807);  slice_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_71: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_359, 3, 0, 64);  add_359 = None
        add_361: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_70, slice_scatter_71);  slice_scatter_70 = slice_scatter_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_597: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1012, [0, 2, 1, 3]);  view_1012 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_233: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_597, memory_format = torch.contiguous_format);  permute_597 = None
        view_1019: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_233, [1, 128, 4096]);  clone_233 = None
        view_1020: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_361, [1, 128, 4096]);  add_361 = None
        view_1021: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_360, [1, 128, 4096]);  add_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1022: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1019, [128, 4096]);  view_1019 = None
        permute_598: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1022, [1, 0])
        mm_216: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_598, view_532);  permute_598 = None
        mm_217: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1022, permute_600);  view_1022 = permute_600 = None
        view_1023: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_217, [1, 128, 4096]);  mm_217 = None
        convert_element_type_1580: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1023, torch.float32);  view_1023 = None
        add_362: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1554, convert_element_type_1580);  convert_element_type_1554 = convert_element_type_1580 = None
        convert_element_type_1581: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_216, torch.float32);  mm_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1024: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1020, [128, 4096]);  view_1020 = None
        permute_602: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1024, [1, 0])
        mm_218: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_602, view_532);  permute_602 = None
        mm_219: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1024, permute_604);  view_1024 = permute_604 = None
        view_1025: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_219, [1, 128, 4096]);  mm_219 = None
        convert_element_type_1586: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1025, torch.float32);  view_1025 = None
        add_363: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_362, convert_element_type_1586);  add_362 = convert_element_type_1586 = None
        convert_element_type_1587: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_218, torch.float32);  mm_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1026: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1021, [128, 4096]);  view_1021 = None
        permute_606: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1026, [1, 0])
        mm_220: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_606, view_532);  permute_606 = view_532 = None
        mm_221: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1026, permute_608);  view_1026 = permute_608 = None
        view_1027: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_221, [1, 128, 4096]);  mm_221 = None
        convert_element_type_1592: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1027, torch.float32);  view_1027 = None
        add_364: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_363, convert_element_type_1592);  add_363 = convert_element_type_1592 = None
        convert_element_type_1593: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_220, torch.float32);  mm_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_474: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_364, primals_212);  primals_212 = None
        mul_475: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_474, 4096)
        sum_97: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_474, [2], True)
        mul_476: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_474, mul_190);  mul_474 = None
        sum_98: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True);  mul_476 = None
        mul_477: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_190, sum_98);  sum_98 = None
        sub_99: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_475, sum_97);  mul_475 = sum_97 = None
        sub_100: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_99, mul_477);  sub_99 = mul_477 = None
        mul_478: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_76, sub_100);  div_76 = sub_100 = None
        mul_479: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_364, mul_190);  mul_190 = None
        sum_99: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 1]);  mul_479 = None
        sum_100: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_364, [0, 1]);  add_364 = None
        add_365: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_353, mul_478);  add_353 = mul_478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1594: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_365, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1028: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1594, [128, 4096]);  convert_element_type_1594 = None
        mm_222: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1028, permute_610);  permute_610 = None
        permute_611: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1028, [1, 0])
        mm_223: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_611, view_530);  view_530 = None
        sum_101: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1028, [0], True, dtype = torch.float32)
        view_1029: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_101, [4096]);  sum_101 = None
        convert_element_type_1599: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1029, torch.bfloat16);  view_1029 = None
        view_1030: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_222, [1, 128, 16384]);  mm_222 = None
        convert_element_type_1600: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1030, torch.float32);  view_1030 = None
        convert_element_type_1601: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_223, torch.float32);  mm_223 = None
        convert_element_type_1602: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1599, torch.float32);  convert_element_type_1599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_529: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [1, 128, 16384]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_186: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_529, 0.5)
        mul_480: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1600, mul_186);  mul_186 = None
        convert_element_type_715: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_529, torch.float32)
        pow_19: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_715, 3.0)
        mul_187: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_170: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_529, mul_187);  view_529 = mul_187 = None
        mul_188: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_170, 0.7978845608028654);  add_170 = None
        tanh_18: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_188);  mul_188 = None
        add_171: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_18, 1.0)
        mul_481: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1600, add_171);  convert_element_type_1600 = add_171 = None
        convert_element_type_1603: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_481, torch.bfloat16);  mul_481 = None
        mul_482: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_18, tanh_18);  tanh_18 = None
        sub_101: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_482);  mul_482 = None
        mul_483: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_480, sub_101);  mul_480 = sub_101 = None
        mul_484: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_483, 0.7978845608028654);  mul_483 = None
        convert_element_type_1604: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_484, torch.bfloat16)
        mul_485: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_484, 0.044715);  mul_484 = None
        pow_38: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_715, 2.0);  convert_element_type_715 = None
        mul_486: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_38, 3.0);  pow_38 = None
        mul_487: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_485, mul_486);  mul_485 = mul_486 = None
        convert_element_type_1605: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_487, torch.bfloat16);  mul_487 = None
        add_366: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1604, convert_element_type_1605);  convert_element_type_1604 = convert_element_type_1605 = None
        mul_488: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1603, 0.5);  convert_element_type_1603 = None
        add_367: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_366, mul_488);  add_366 = mul_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1031: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_367, [128, 16384]);  add_367 = None
        mm_224: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1031, permute_614);  permute_614 = None
        permute_615: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1031, [1, 0])
        mm_225: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_615, view_504);  permute_615 = None
        sum_102: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1031, [0], True, dtype = torch.float32);  view_1031 = None
        view_1032: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_102, [16384]);  sum_102 = None
        convert_element_type_1610: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1032, torch.bfloat16);  view_1032 = None
        view_1033: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_224, [1, 128, 4096]);  mm_224 = None
        convert_element_type_1611: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1033, torch.float32);  view_1033 = None
        convert_element_type_1612: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_225, torch.float32);  mm_225 = None
        convert_element_type_1613: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1610, torch.float32);  convert_element_type_1610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_226: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_611, view_526);  permute_611 = view_526 = None
        mm_227: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1028, permute_620);  view_1028 = permute_620 = None
        view_1035: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_227, [1, 128, 4096]);  mm_227 = None
        convert_element_type_1618: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_226, torch.float32);  mm_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1036: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1035, [1, 128, 16, 256]);  view_1035 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_622: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1036, [0, 2, 1, 3]);  view_1036 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1037: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_622, [16, 128, 256]);  permute_622 = None
        bmm_92: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_623, view_1037);  permute_623 = None
        bmm_93: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1037, permute_624);  view_1037 = permute_624 = None
        view_1038: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_92, [1, 16, 128, 256]);  bmm_92 = None
        view_1039: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_93, [1, 16, 128, 128]);  bmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1623: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1039, torch.float32);  view_1039 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_489: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1623, div_37);  convert_element_type_1623 = None
        sum_103: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_489, [-1], True)
        neg_84: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_37);  div_37 = None
        fma_9: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_84, sum_103, mul_489);  neg_84 = sum_103 = mul_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1624: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_77: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1624, 16.0);  convert_element_type_1624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1040: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_77, [16, 128, 128]);  div_77 = None
        bmm_94: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_625, view_1040);  permute_625 = None
        bmm_95: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1040, permute_626);  view_1040 = permute_626 = None
        view_1041: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_94, [1, 16, 256, 128]);  bmm_94 = None
        view_1042: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_95, [1, 16, 128, 256]);  bmm_95 = None
        convert_element_type_1630: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1041, torch.float32);  view_1041 = None
        permute_627: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1630, [0, 1, 3, 2]);  convert_element_type_1630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1631: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_627, torch.bfloat16);  permute_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_18: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1042, torch.bfloat16);  view_1042 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_628: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_18, [0, 2, 1, 3]);  convert_element_type_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_629: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1631, [0, 2, 1, 3]);  convert_element_type_1631 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_265: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_628, 3, 0, 64)
        slice_266: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_628, 3, 64, 256);  permute_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_267: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_629, 3, 0, 64)
        slice_268: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_629, 3, 64, 256);  permute_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_145: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_246, [1, 128, 1, 32, 2]);  unsqueeze_246 = None
        clone_145: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
        view_513: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_145, [1, 128, 1, 64]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_490: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_265, view_513)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1043: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_490, [1, 128, 16, 32, 2]);  mul_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_36: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1043, -1, 0)
        select_37: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1043, -1, 1);  view_1043 = None
        neg_85: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_36);  select_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_72: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_85, 3, 1, 9223372036854775807, 2);  neg_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_73: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_37, 3, 0, 9223372036854775807, 2);  select_37 = None
        add_368: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_72, slice_scatter_73);  slice_scatter_72 = slice_scatter_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_146: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_248, [1, 128, 1, 32, 2]);  unsqueeze_248 = None
        clone_146: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_146, memory_format = torch.contiguous_format);  expand_146 = None
        view_514: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_146, [1, 128, 1, 64]);  clone_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_491: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_265, view_514);  slice_265 = None
        add_369: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_368, mul_491);  add_368 = mul_491 = None
        mul_492: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_267, view_513);  view_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1044: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_492, [1, 128, 16, 32, 2]);  mul_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_38: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1044, -1, 0)
        select_39: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1044, -1, 1);  view_1044 = None
        neg_86: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_38);  select_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_74: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_86, 3, 1, 9223372036854775807, 2);  neg_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_75: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_39, 3, 0, 9223372036854775807, 2);  select_39 = None
        add_370: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_74, slice_scatter_75);  slice_scatter_74 = slice_scatter_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_493: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_267, view_514);  slice_267 = view_514 = None
        add_371: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_370, mul_493);  add_370 = mul_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_76: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_266, 3, 64, 9223372036854775807);  slice_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_77: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_369, 3, 0, 64);  add_369 = None
        add_372: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_76, slice_scatter_77);  slice_scatter_76 = slice_scatter_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_78: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_268, 3, 64, 9223372036854775807);  slice_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_79: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_371, 3, 0, 64);  add_371 = None
        add_373: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_78, slice_scatter_79);  slice_scatter_78 = slice_scatter_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_630: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1038, [0, 2, 1, 3]);  view_1038 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_234: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_630, memory_format = torch.contiguous_format);  permute_630 = None
        view_1045: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_234, [1, 128, 4096]);  clone_234 = None
        view_1046: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_373, [1, 128, 4096]);  add_373 = None
        view_1047: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_372, [1, 128, 4096]);  add_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1048: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1045, [128, 4096]);  view_1045 = None
        permute_631: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1048, [1, 0])
        mm_228: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_631, view_504);  permute_631 = None
        mm_229: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1048, permute_633);  view_1048 = permute_633 = None
        view_1049: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_229, [1, 128, 4096]);  mm_229 = None
        convert_element_type_1637: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1049, torch.float32);  view_1049 = None
        add_374: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1611, convert_element_type_1637);  convert_element_type_1611 = convert_element_type_1637 = None
        convert_element_type_1638: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_228, torch.float32);  mm_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1050: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1046, [128, 4096]);  view_1046 = None
        permute_635: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1050, [1, 0])
        mm_230: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_635, view_504);  permute_635 = None
        mm_231: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1050, permute_637);  view_1050 = permute_637 = None
        view_1051: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_231, [1, 128, 4096]);  mm_231 = None
        convert_element_type_1643: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1051, torch.float32);  view_1051 = None
        add_375: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_374, convert_element_type_1643);  add_374 = convert_element_type_1643 = None
        convert_element_type_1644: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_230, torch.float32);  mm_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1052: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1047, [128, 4096]);  view_1047 = None
        permute_639: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1052, [1, 0])
        mm_232: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_639, view_504);  permute_639 = view_504 = None
        mm_233: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1052, permute_641);  view_1052 = permute_641 = None
        view_1053: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_233, [1, 128, 4096]);  mm_233 = None
        convert_element_type_1649: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1053, torch.float32);  view_1053 = None
        add_376: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_375, convert_element_type_1649);  add_375 = convert_element_type_1649 = None
        convert_element_type_1650: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_232, torch.float32);  mm_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_495: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_376, primals_201);  primals_201 = None
        mul_496: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_495, 4096)
        sum_104: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_495, [2], True)
        mul_497: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_495, mul_180);  mul_495 = None
        sum_105: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True);  mul_497 = None
        mul_498: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_180, sum_105);  sum_105 = None
        sub_103: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_496, sum_104);  mul_496 = sum_104 = None
        sub_104: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_103, mul_498);  sub_103 = mul_498 = None
        mul_499: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_78, sub_104);  div_78 = sub_104 = None
        mul_500: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_376, mul_180);  mul_180 = None
        sum_106: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 1]);  mul_500 = None
        sum_107: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_376, [0, 1]);  add_376 = None
        add_377: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_365, mul_499);  add_365 = mul_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1651: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_377, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1054: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1651, [128, 4096]);  convert_element_type_1651 = None
        mm_234: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1054, permute_643);  permute_643 = None
        permute_644: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1054, [1, 0])
        mm_235: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_644, view_502);  view_502 = None
        sum_108: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1054, [0], True, dtype = torch.float32)
        view_1055: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_108, [4096]);  sum_108 = None
        convert_element_type_1656: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1055, torch.bfloat16);  view_1055 = None
        view_1056: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_234, [1, 128, 16384]);  mm_234 = None
        convert_element_type_1657: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1056, torch.float32);  view_1056 = None
        convert_element_type_1658: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_235, torch.float32);  mm_235 = None
        convert_element_type_1659: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1656, torch.float32);  convert_element_type_1656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_501: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [1, 128, 16384]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_176: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_501, 0.5)
        mul_501: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1657, mul_176);  mul_176 = None
        convert_element_type_677: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_501, torch.float32)
        pow_18: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_677, 3.0)
        mul_177: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_161: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_501, mul_177);  view_501 = mul_177 = None
        mul_178: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_161, 0.7978845608028654);  add_161 = None
        tanh_17: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_178);  mul_178 = None
        add_162: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_17, 1.0)
        mul_502: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1657, add_162);  convert_element_type_1657 = add_162 = None
        convert_element_type_1660: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_502, torch.bfloat16);  mul_502 = None
        mul_503: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_17, tanh_17);  tanh_17 = None
        sub_105: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_503);  mul_503 = None
        mul_504: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_501, sub_105);  mul_501 = sub_105 = None
        mul_505: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_504, 0.7978845608028654);  mul_504 = None
        convert_element_type_1661: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_505, torch.bfloat16)
        mul_506: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_505, 0.044715);  mul_505 = None
        pow_39: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_677, 2.0);  convert_element_type_677 = None
        mul_507: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_39, 3.0);  pow_39 = None
        mul_508: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_506, mul_507);  mul_506 = mul_507 = None
        convert_element_type_1662: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_508, torch.bfloat16);  mul_508 = None
        add_378: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1661, convert_element_type_1662);  convert_element_type_1661 = convert_element_type_1662 = None
        mul_509: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1660, 0.5);  convert_element_type_1660 = None
        add_379: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_378, mul_509);  add_378 = mul_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1057: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_379, [128, 16384]);  add_379 = None
        mm_236: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1057, permute_647);  permute_647 = None
        permute_648: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1057, [1, 0])
        mm_237: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_648, view_476);  permute_648 = None
        sum_109: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1057, [0], True, dtype = torch.float32);  view_1057 = None
        view_1058: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_109, [16384]);  sum_109 = None
        convert_element_type_1667: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1058, torch.bfloat16);  view_1058 = None
        view_1059: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_236, [1, 128, 4096]);  mm_236 = None
        convert_element_type_1668: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1059, torch.float32);  view_1059 = None
        convert_element_type_1669: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_237, torch.float32);  mm_237 = None
        convert_element_type_1670: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1667, torch.float32);  convert_element_type_1667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_238: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_644, view_498);  permute_644 = view_498 = None
        mm_239: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1054, permute_653);  view_1054 = permute_653 = None
        view_1061: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_239, [1, 128, 4096]);  mm_239 = None
        convert_element_type_1675: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_238, torch.float32);  mm_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1062: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1061, [1, 128, 16, 256]);  view_1061 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_655: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1062, [0, 2, 1, 3]);  view_1062 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1063: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_655, [16, 128, 256]);  permute_655 = None
        bmm_96: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_656, view_1063);  permute_656 = None
        bmm_97: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1063, permute_657);  view_1063 = permute_657 = None
        view_1064: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_96, [1, 16, 128, 256]);  bmm_96 = None
        view_1065: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_97, [1, 16, 128, 128]);  bmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1680: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1065, torch.float32);  view_1065 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_510: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1680, div_35);  convert_element_type_1680 = None
        sum_110: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_510, [-1], True)
        neg_87: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_35);  div_35 = None
        fma_10: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_87, sum_110, mul_510);  neg_87 = sum_110 = mul_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1681: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_79: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1681, 16.0);  convert_element_type_1681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1066: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_79, [16, 128, 128]);  div_79 = None
        bmm_98: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_658, view_1066);  permute_658 = None
        bmm_99: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1066, permute_659);  view_1066 = permute_659 = None
        view_1067: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_98, [1, 16, 256, 128]);  bmm_98 = None
        view_1068: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_99, [1, 16, 128, 256]);  bmm_99 = None
        convert_element_type_1687: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1067, torch.float32);  view_1067 = None
        permute_660: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1687, [0, 1, 3, 2]);  convert_element_type_1687 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1688: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_660, torch.bfloat16);  permute_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_17: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1068, torch.bfloat16);  view_1068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_661: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_17, [0, 2, 1, 3]);  convert_element_type_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_662: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1688, [0, 2, 1, 3]);  convert_element_type_1688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_269: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_661, 3, 0, 64)
        slice_270: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_661, 3, 64, 256);  permute_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_271: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_662, 3, 0, 64)
        slice_272: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_662, 3, 64, 256);  permute_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_137: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_233, [1, 128, 1, 32, 2]);  unsqueeze_233 = None
        clone_137: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
        view_485: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_137, [1, 128, 1, 64]);  clone_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_511: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_269, view_485)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1069: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_511, [1, 128, 16, 32, 2]);  mul_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_40: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1069, -1, 0)
        select_41: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1069, -1, 1);  view_1069 = None
        neg_88: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_40);  select_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_80: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_88, 3, 1, 9223372036854775807, 2);  neg_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_81: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_41, 3, 0, 9223372036854775807, 2);  select_41 = None
        add_380: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_80, slice_scatter_81);  slice_scatter_80 = slice_scatter_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_138: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_235, [1, 128, 1, 32, 2]);  unsqueeze_235 = None
        clone_138: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_138, memory_format = torch.contiguous_format);  expand_138 = None
        view_486: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_138, [1, 128, 1, 64]);  clone_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_512: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_269, view_486);  slice_269 = None
        add_381: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_380, mul_512);  add_380 = mul_512 = None
        mul_513: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_271, view_485);  view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1070: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_513, [1, 128, 16, 32, 2]);  mul_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_42: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1070, -1, 0)
        select_43: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1070, -1, 1);  view_1070 = None
        neg_89: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_42);  select_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_82: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_89, 3, 1, 9223372036854775807, 2);  neg_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_83: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_43, 3, 0, 9223372036854775807, 2);  select_43 = None
        add_382: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_82, slice_scatter_83);  slice_scatter_82 = slice_scatter_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_514: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_271, view_486);  slice_271 = view_486 = None
        add_383: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_382, mul_514);  add_382 = mul_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_84: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_270, 3, 64, 9223372036854775807);  slice_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_85: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_381, 3, 0, 64);  add_381 = None
        add_384: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_84, slice_scatter_85);  slice_scatter_84 = slice_scatter_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_86: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_272, 3, 64, 9223372036854775807);  slice_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_87: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_383, 3, 0, 64);  add_383 = None
        add_385: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_86, slice_scatter_87);  slice_scatter_86 = slice_scatter_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_663: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1064, [0, 2, 1, 3]);  view_1064 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_235: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_663, memory_format = torch.contiguous_format);  permute_663 = None
        view_1071: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_235, [1, 128, 4096]);  clone_235 = None
        view_1072: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_385, [1, 128, 4096]);  add_385 = None
        view_1073: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_384, [1, 128, 4096]);  add_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1074: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1071, [128, 4096]);  view_1071 = None
        permute_664: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1074, [1, 0])
        mm_240: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_664, view_476);  permute_664 = None
        mm_241: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1074, permute_666);  view_1074 = permute_666 = None
        view_1075: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_241, [1, 128, 4096]);  mm_241 = None
        convert_element_type_1694: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1075, torch.float32);  view_1075 = None
        add_386: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1668, convert_element_type_1694);  convert_element_type_1668 = convert_element_type_1694 = None
        convert_element_type_1695: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_240, torch.float32);  mm_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1076: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1072, [128, 4096]);  view_1072 = None
        permute_668: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1076, [1, 0])
        mm_242: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_668, view_476);  permute_668 = None
        mm_243: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1076, permute_670);  view_1076 = permute_670 = None
        view_1077: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_243, [1, 128, 4096]);  mm_243 = None
        convert_element_type_1700: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1077, torch.float32);  view_1077 = None
        add_387: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_386, convert_element_type_1700);  add_386 = convert_element_type_1700 = None
        convert_element_type_1701: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_242, torch.float32);  mm_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1078: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1073, [128, 4096]);  view_1073 = None
        permute_672: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1078, [1, 0])
        mm_244: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_672, view_476);  permute_672 = view_476 = None
        mm_245: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1078, permute_674);  view_1078 = permute_674 = None
        view_1079: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_245, [1, 128, 4096]);  mm_245 = None
        convert_element_type_1706: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1079, torch.float32);  view_1079 = None
        add_388: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_387, convert_element_type_1706);  add_387 = convert_element_type_1706 = None
        convert_element_type_1707: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_244, torch.float32);  mm_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_516: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_388, primals_190);  primals_190 = None
        mul_517: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_516, 4096)
        sum_111: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_516, [2], True)
        mul_518: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_516, mul_170);  mul_516 = None
        sum_112: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_518, [2], True);  mul_518 = None
        mul_519: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_170, sum_112);  sum_112 = None
        sub_107: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_517, sum_111);  mul_517 = sum_111 = None
        sub_108: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_107, mul_519);  sub_107 = mul_519 = None
        mul_520: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_80, sub_108);  div_80 = sub_108 = None
        mul_521: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_388, mul_170);  mul_170 = None
        sum_113: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_521, [0, 1]);  mul_521 = None
        sum_114: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_388, [0, 1]);  add_388 = None
        add_389: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_377, mul_520);  add_377 = mul_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1708: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_389, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1080: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1708, [128, 4096]);  convert_element_type_1708 = None
        mm_246: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1080, permute_676);  permute_676 = None
        permute_677: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1080, [1, 0])
        mm_247: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_677, view_474);  view_474 = None
        sum_115: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1080, [0], True, dtype = torch.float32)
        view_1081: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_115, [4096]);  sum_115 = None
        convert_element_type_1713: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1081, torch.bfloat16);  view_1081 = None
        view_1082: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_246, [1, 128, 16384]);  mm_246 = None
        convert_element_type_1714: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1082, torch.float32);  view_1082 = None
        convert_element_type_1715: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_247, torch.float32);  mm_247 = None
        convert_element_type_1716: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1713, torch.float32);  convert_element_type_1713 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_473: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [1, 128, 16384]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_166: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_473, 0.5)
        mul_522: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1714, mul_166);  mul_166 = None
        convert_element_type_639: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_473, torch.float32)
        pow_17: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_639, 3.0)
        mul_167: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_152: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_473, mul_167);  view_473 = mul_167 = None
        mul_168: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_152, 0.7978845608028654);  add_152 = None
        tanh_16: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_168);  mul_168 = None
        add_153: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_16, 1.0)
        mul_523: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1714, add_153);  convert_element_type_1714 = add_153 = None
        convert_element_type_1717: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_523, torch.bfloat16);  mul_523 = None
        mul_524: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_16, tanh_16);  tanh_16 = None
        sub_109: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_524);  mul_524 = None
        mul_525: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_522, sub_109);  mul_522 = sub_109 = None
        mul_526: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_525, 0.7978845608028654);  mul_525 = None
        convert_element_type_1718: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_526, torch.bfloat16)
        mul_527: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_526, 0.044715);  mul_526 = None
        pow_40: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_639, 2.0);  convert_element_type_639 = None
        mul_528: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_40, 3.0);  pow_40 = None
        mul_529: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_527, mul_528);  mul_527 = mul_528 = None
        convert_element_type_1719: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_529, torch.bfloat16);  mul_529 = None
        add_390: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1718, convert_element_type_1719);  convert_element_type_1718 = convert_element_type_1719 = None
        mul_530: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1717, 0.5);  convert_element_type_1717 = None
        add_391: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_390, mul_530);  add_390 = mul_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1083: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_391, [128, 16384]);  add_391 = None
        mm_248: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1083, permute_680);  permute_680 = None
        permute_681: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1083, [1, 0])
        mm_249: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_681, view_448);  permute_681 = None
        sum_116: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1083, [0], True, dtype = torch.float32);  view_1083 = None
        view_1084: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_116, [16384]);  sum_116 = None
        convert_element_type_1724: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1084, torch.bfloat16);  view_1084 = None
        view_1085: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_248, [1, 128, 4096]);  mm_248 = None
        convert_element_type_1725: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1085, torch.float32);  view_1085 = None
        convert_element_type_1726: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_249, torch.float32);  mm_249 = None
        convert_element_type_1727: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1724, torch.float32);  convert_element_type_1724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_250: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_677, view_470);  permute_677 = view_470 = None
        mm_251: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1080, permute_686);  view_1080 = permute_686 = None
        view_1087: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_251, [1, 128, 4096]);  mm_251 = None
        convert_element_type_1732: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_250, torch.float32);  mm_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1088: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1087, [1, 128, 16, 256]);  view_1087 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_688: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1088, [0, 2, 1, 3]);  view_1088 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1089: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_688, [16, 128, 256]);  permute_688 = None
        bmm_100: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_689, view_1089);  permute_689 = None
        bmm_101: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1089, permute_690);  view_1089 = permute_690 = None
        view_1090: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_100, [1, 16, 128, 256]);  bmm_100 = None
        view_1091: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_101, [1, 16, 128, 128]);  bmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1737: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1091, torch.float32);  view_1091 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_531: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1737, div_33);  convert_element_type_1737 = None
        sum_117: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_531, [-1], True)
        neg_90: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_33);  div_33 = None
        fma_11: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_90, sum_117, mul_531);  neg_90 = sum_117 = mul_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1738: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_81: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1738, 16.0);  convert_element_type_1738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1092: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_81, [16, 128, 128]);  div_81 = None
        bmm_102: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_691, view_1092);  permute_691 = None
        bmm_103: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1092, permute_692);  view_1092 = permute_692 = None
        view_1093: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_102, [1, 16, 256, 128]);  bmm_102 = None
        view_1094: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_103, [1, 16, 128, 256]);  bmm_103 = None
        convert_element_type_1744: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1093, torch.float32);  view_1093 = None
        permute_693: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1744, [0, 1, 3, 2]);  convert_element_type_1744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1745: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_693, torch.bfloat16);  permute_693 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_16: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1094, torch.bfloat16);  view_1094 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_694: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_16, [0, 2, 1, 3]);  convert_element_type_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_695: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1745, [0, 2, 1, 3]);  convert_element_type_1745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_273: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_694, 3, 0, 64)
        slice_274: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_694, 3, 64, 256);  permute_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_275: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_695, 3, 0, 64)
        slice_276: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_695, 3, 64, 256);  permute_695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_129: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_220, [1, 128, 1, 32, 2]);  unsqueeze_220 = None
        clone_129: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
        view_457: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [1, 128, 1, 64]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_532: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_273, view_457)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1095: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_532, [1, 128, 16, 32, 2]);  mul_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_44: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1095, -1, 0)
        select_45: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1095, -1, 1);  view_1095 = None
        neg_91: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_44);  select_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_88: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_91, 3, 1, 9223372036854775807, 2);  neg_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_89: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_45, 3, 0, 9223372036854775807, 2);  select_45 = None
        add_392: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_88, slice_scatter_89);  slice_scatter_88 = slice_scatter_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_130: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_222, [1, 128, 1, 32, 2]);  unsqueeze_222 = None
        clone_130: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_130, memory_format = torch.contiguous_format);  expand_130 = None
        view_458: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [1, 128, 1, 64]);  clone_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_533: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_273, view_458);  slice_273 = None
        add_393: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_392, mul_533);  add_392 = mul_533 = None
        mul_534: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_275, view_457);  view_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1096: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_534, [1, 128, 16, 32, 2]);  mul_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_46: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1096, -1, 0)
        select_47: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1096, -1, 1);  view_1096 = None
        neg_92: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_46);  select_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_90: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_92, 3, 1, 9223372036854775807, 2);  neg_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_91: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_47, 3, 0, 9223372036854775807, 2);  select_47 = None
        add_394: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_90, slice_scatter_91);  slice_scatter_90 = slice_scatter_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_535: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_275, view_458);  slice_275 = view_458 = None
        add_395: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_394, mul_535);  add_394 = mul_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_92: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_274, 3, 64, 9223372036854775807);  slice_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_93: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_393, 3, 0, 64);  add_393 = None
        add_396: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_92, slice_scatter_93);  slice_scatter_92 = slice_scatter_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_94: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_276, 3, 64, 9223372036854775807);  slice_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_95: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_395, 3, 0, 64);  add_395 = None
        add_397: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_94, slice_scatter_95);  slice_scatter_94 = slice_scatter_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_696: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1090, [0, 2, 1, 3]);  view_1090 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_236: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_696, memory_format = torch.contiguous_format);  permute_696 = None
        view_1097: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_236, [1, 128, 4096]);  clone_236 = None
        view_1098: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_397, [1, 128, 4096]);  add_397 = None
        view_1099: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_396, [1, 128, 4096]);  add_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1100: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1097, [128, 4096]);  view_1097 = None
        permute_697: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1100, [1, 0])
        mm_252: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_697, view_448);  permute_697 = None
        mm_253: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1100, permute_699);  view_1100 = permute_699 = None
        view_1101: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_253, [1, 128, 4096]);  mm_253 = None
        convert_element_type_1751: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1101, torch.float32);  view_1101 = None
        add_398: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1725, convert_element_type_1751);  convert_element_type_1725 = convert_element_type_1751 = None
        convert_element_type_1752: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_252, torch.float32);  mm_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1102: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1098, [128, 4096]);  view_1098 = None
        permute_701: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1102, [1, 0])
        mm_254: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_701, view_448);  permute_701 = None
        mm_255: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1102, permute_703);  view_1102 = permute_703 = None
        view_1103: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_255, [1, 128, 4096]);  mm_255 = None
        convert_element_type_1757: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1103, torch.float32);  view_1103 = None
        add_399: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_398, convert_element_type_1757);  add_398 = convert_element_type_1757 = None
        convert_element_type_1758: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_254, torch.float32);  mm_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1104: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1099, [128, 4096]);  view_1099 = None
        permute_705: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1104, [1, 0])
        mm_256: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_705, view_448);  permute_705 = view_448 = None
        mm_257: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1104, permute_707);  view_1104 = permute_707 = None
        view_1105: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_257, [1, 128, 4096]);  mm_257 = None
        convert_element_type_1763: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1105, torch.float32);  view_1105 = None
        add_400: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_399, convert_element_type_1763);  add_399 = convert_element_type_1763 = None
        convert_element_type_1764: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_256, torch.float32);  mm_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_537: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_400, primals_179);  primals_179 = None
        mul_538: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_537, 4096)
        sum_118: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_537, [2], True)
        mul_539: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_537, mul_160);  mul_537 = None
        sum_119: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_539, [2], True);  mul_539 = None
        mul_540: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, sum_119);  sum_119 = None
        sub_111: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_538, sum_118);  mul_538 = sum_118 = None
        sub_112: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_111, mul_540);  sub_111 = mul_540 = None
        mul_541: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_82, sub_112);  div_82 = sub_112 = None
        mul_542: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_400, mul_160);  mul_160 = None
        sum_120: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_542, [0, 1]);  mul_542 = None
        sum_121: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_400, [0, 1]);  add_400 = None
        add_401: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_389, mul_541);  add_389 = mul_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1765: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_401, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1106: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1765, [128, 4096]);  convert_element_type_1765 = None
        mm_258: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1106, permute_709);  permute_709 = None
        permute_710: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1106, [1, 0])
        mm_259: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_710, view_446);  view_446 = None
        sum_122: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1106, [0], True, dtype = torch.float32)
        view_1107: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_122, [4096]);  sum_122 = None
        convert_element_type_1770: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1107, torch.bfloat16);  view_1107 = None
        view_1108: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_258, [1, 128, 16384]);  mm_258 = None
        convert_element_type_1771: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1108, torch.float32);  view_1108 = None
        convert_element_type_1772: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_259, torch.float32);  mm_259 = None
        convert_element_type_1773: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1770, torch.float32);  convert_element_type_1770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_445: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [1, 128, 16384]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_156: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_445, 0.5)
        mul_543: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1771, mul_156);  mul_156 = None
        convert_element_type_601: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_445, torch.float32)
        pow_16: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_601, 3.0)
        mul_157: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_143: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_445, mul_157);  view_445 = mul_157 = None
        mul_158: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_143, 0.7978845608028654);  add_143 = None
        tanh_15: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
        add_144: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
        mul_544: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1771, add_144);  convert_element_type_1771 = add_144 = None
        convert_element_type_1774: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_544, torch.bfloat16);  mul_544 = None
        mul_545: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_15, tanh_15);  tanh_15 = None
        sub_113: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_545);  mul_545 = None
        mul_546: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_543, sub_113);  mul_543 = sub_113 = None
        mul_547: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_546, 0.7978845608028654);  mul_546 = None
        convert_element_type_1775: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_547, torch.bfloat16)
        mul_548: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_547, 0.044715);  mul_547 = None
        pow_41: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_601, 2.0);  convert_element_type_601 = None
        mul_549: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_41, 3.0);  pow_41 = None
        mul_550: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_548, mul_549);  mul_548 = mul_549 = None
        convert_element_type_1776: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_550, torch.bfloat16);  mul_550 = None
        add_402: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1775, convert_element_type_1776);  convert_element_type_1775 = convert_element_type_1776 = None
        mul_551: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1774, 0.5);  convert_element_type_1774 = None
        add_403: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_402, mul_551);  add_402 = mul_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1109: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_403, [128, 16384]);  add_403 = None
        mm_260: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1109, permute_713);  permute_713 = None
        permute_714: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1109, [1, 0])
        mm_261: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_714, view_420);  permute_714 = None
        sum_123: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1109, [0], True, dtype = torch.float32);  view_1109 = None
        view_1110: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_123, [16384]);  sum_123 = None
        convert_element_type_1781: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1110, torch.bfloat16);  view_1110 = None
        view_1111: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_260, [1, 128, 4096]);  mm_260 = None
        convert_element_type_1782: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1111, torch.float32);  view_1111 = None
        convert_element_type_1783: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_261, torch.float32);  mm_261 = None
        convert_element_type_1784: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1781, torch.float32);  convert_element_type_1781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_262: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_710, view_442);  permute_710 = view_442 = None
        mm_263: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1106, permute_719);  view_1106 = permute_719 = None
        view_1113: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_263, [1, 128, 4096]);  mm_263 = None
        convert_element_type_1789: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_262, torch.float32);  mm_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1114: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1113, [1, 128, 16, 256]);  view_1113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_721: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1114, [0, 2, 1, 3]);  view_1114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1115: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_721, [16, 128, 256]);  permute_721 = None
        bmm_104: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_722, view_1115);  permute_722 = None
        bmm_105: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1115, permute_723);  view_1115 = permute_723 = None
        view_1116: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_104, [1, 16, 128, 256]);  bmm_104 = None
        view_1117: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_105, [1, 16, 128, 128]);  bmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1794: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1117, torch.float32);  view_1117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_552: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1794, div_31);  convert_element_type_1794 = None
        sum_124: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_552, [-1], True)
        neg_93: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_31);  div_31 = None
        fma_12: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_93, sum_124, mul_552);  neg_93 = sum_124 = mul_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1795: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_12, torch.bfloat16);  fma_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_83: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1795, 16.0);  convert_element_type_1795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1118: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_83, [16, 128, 128]);  div_83 = None
        bmm_106: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_724, view_1118);  permute_724 = None
        bmm_107: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1118, permute_725);  view_1118 = permute_725 = None
        view_1119: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_106, [1, 16, 256, 128]);  bmm_106 = None
        view_1120: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_107, [1, 16, 128, 256]);  bmm_107 = None
        convert_element_type_1801: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1119, torch.float32);  view_1119 = None
        permute_726: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1801, [0, 1, 3, 2]);  convert_element_type_1801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1802: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_726, torch.bfloat16);  permute_726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_15: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1120, torch.bfloat16);  view_1120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_727: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_15, [0, 2, 1, 3]);  convert_element_type_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_728: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1802, [0, 2, 1, 3]);  convert_element_type_1802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_277: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_727, 3, 0, 64)
        slice_278: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_727, 3, 64, 256);  permute_727 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_279: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_728, 3, 0, 64)
        slice_280: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_728, 3, 64, 256);  permute_728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_121: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_207, [1, 128, 1, 32, 2]);  unsqueeze_207 = None
        clone_121: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
        view_429: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_121, [1, 128, 1, 64]);  clone_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_553: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_277, view_429)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1121: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_553, [1, 128, 16, 32, 2]);  mul_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_48: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1121, -1, 0)
        select_49: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1121, -1, 1);  view_1121 = None
        neg_94: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_48);  select_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_96: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_94, 3, 1, 9223372036854775807, 2);  neg_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_97: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_49, 3, 0, 9223372036854775807, 2);  select_49 = None
        add_404: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_96, slice_scatter_97);  slice_scatter_96 = slice_scatter_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_122: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_209, [1, 128, 1, 32, 2]);  unsqueeze_209 = None
        clone_122: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
        view_430: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_122, [1, 128, 1, 64]);  clone_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_554: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_277, view_430);  slice_277 = None
        add_405: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_404, mul_554);  add_404 = mul_554 = None
        mul_555: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_279, view_429);  view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1122: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_555, [1, 128, 16, 32, 2]);  mul_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_50: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1122, -1, 0)
        select_51: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1122, -1, 1);  view_1122 = None
        neg_95: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_50);  select_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_98: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_95, 3, 1, 9223372036854775807, 2);  neg_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_99: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_51, 3, 0, 9223372036854775807, 2);  select_51 = None
        add_406: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_98, slice_scatter_99);  slice_scatter_98 = slice_scatter_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_556: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_279, view_430);  slice_279 = view_430 = None
        add_407: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_406, mul_556);  add_406 = mul_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_100: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_278, 3, 64, 9223372036854775807);  slice_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_101: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_405, 3, 0, 64);  add_405 = None
        add_408: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_100, slice_scatter_101);  slice_scatter_100 = slice_scatter_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_102: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_280, 3, 64, 9223372036854775807);  slice_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_103: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_407, 3, 0, 64);  add_407 = None
        add_409: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_102, slice_scatter_103);  slice_scatter_102 = slice_scatter_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_729: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1116, [0, 2, 1, 3]);  view_1116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_237: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_729, memory_format = torch.contiguous_format);  permute_729 = None
        view_1123: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_237, [1, 128, 4096]);  clone_237 = None
        view_1124: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_409, [1, 128, 4096]);  add_409 = None
        view_1125: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_408, [1, 128, 4096]);  add_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1126: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1123, [128, 4096]);  view_1123 = None
        permute_730: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1126, [1, 0])
        mm_264: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_730, view_420);  permute_730 = None
        mm_265: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1126, permute_732);  view_1126 = permute_732 = None
        view_1127: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_265, [1, 128, 4096]);  mm_265 = None
        convert_element_type_1808: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1127, torch.float32);  view_1127 = None
        add_410: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1782, convert_element_type_1808);  convert_element_type_1782 = convert_element_type_1808 = None
        convert_element_type_1809: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_264, torch.float32);  mm_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1128: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1124, [128, 4096]);  view_1124 = None
        permute_734: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1128, [1, 0])
        mm_266: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_734, view_420);  permute_734 = None
        mm_267: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1128, permute_736);  view_1128 = permute_736 = None
        view_1129: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_267, [1, 128, 4096]);  mm_267 = None
        convert_element_type_1814: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1129, torch.float32);  view_1129 = None
        add_411: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_410, convert_element_type_1814);  add_410 = convert_element_type_1814 = None
        convert_element_type_1815: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_266, torch.float32);  mm_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1130: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1125, [128, 4096]);  view_1125 = None
        permute_738: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1130, [1, 0])
        mm_268: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_738, view_420);  permute_738 = view_420 = None
        mm_269: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1130, permute_740);  view_1130 = permute_740 = None
        view_1131: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_269, [1, 128, 4096]);  mm_269 = None
        convert_element_type_1820: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1131, torch.float32);  view_1131 = None
        add_412: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_411, convert_element_type_1820);  add_411 = convert_element_type_1820 = None
        convert_element_type_1821: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_268, torch.float32);  mm_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_558: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_412, primals_168);  primals_168 = None
        mul_559: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_558, 4096)
        sum_125: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_558, [2], True)
        mul_560: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_558, mul_150);  mul_558 = None
        sum_126: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_560, [2], True);  mul_560 = None
        mul_561: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, sum_126);  sum_126 = None
        sub_115: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_559, sum_125);  mul_559 = sum_125 = None
        sub_116: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_115, mul_561);  sub_115 = mul_561 = None
        mul_562: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_84, sub_116);  div_84 = sub_116 = None
        mul_563: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_412, mul_150);  mul_150 = None
        sum_127: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_563, [0, 1]);  mul_563 = None
        sum_128: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_412, [0, 1]);  add_412 = None
        add_413: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_401, mul_562);  add_401 = mul_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1822: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_413, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1132: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1822, [128, 4096]);  convert_element_type_1822 = None
        mm_270: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1132, permute_742);  permute_742 = None
        permute_743: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1132, [1, 0])
        mm_271: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_743, view_418);  view_418 = None
        sum_129: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1132, [0], True, dtype = torch.float32)
        view_1133: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_129, [4096]);  sum_129 = None
        convert_element_type_1827: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1133, torch.bfloat16);  view_1133 = None
        view_1134: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_270, [1, 128, 16384]);  mm_270 = None
        convert_element_type_1828: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1134, torch.float32);  view_1134 = None
        convert_element_type_1829: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_271, torch.float32);  mm_271 = None
        convert_element_type_1830: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1827, torch.float32);  convert_element_type_1827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_417: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [1, 128, 16384]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_146: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_417, 0.5)
        mul_564: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1828, mul_146);  mul_146 = None
        convert_element_type_563: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_417, torch.float32)
        pow_15: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_563, 3.0)
        mul_147: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_134: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_417, mul_147);  view_417 = mul_147 = None
        mul_148: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_134, 0.7978845608028654);  add_134 = None
        tanh_14: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_148);  mul_148 = None
        add_135: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
        mul_565: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1828, add_135);  convert_element_type_1828 = add_135 = None
        convert_element_type_1831: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_565, torch.bfloat16);  mul_565 = None
        mul_566: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_14, tanh_14);  tanh_14 = None
        sub_117: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_566);  mul_566 = None
        mul_567: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_564, sub_117);  mul_564 = sub_117 = None
        mul_568: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_567, 0.7978845608028654);  mul_567 = None
        convert_element_type_1832: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_568, torch.bfloat16)
        mul_569: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_568, 0.044715);  mul_568 = None
        pow_42: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_563, 2.0);  convert_element_type_563 = None
        mul_570: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_42, 3.0);  pow_42 = None
        mul_571: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_569, mul_570);  mul_569 = mul_570 = None
        convert_element_type_1833: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_571, torch.bfloat16);  mul_571 = None
        add_414: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1832, convert_element_type_1833);  convert_element_type_1832 = convert_element_type_1833 = None
        mul_572: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1831, 0.5);  convert_element_type_1831 = None
        add_415: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_414, mul_572);  add_414 = mul_572 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1135: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_415, [128, 16384]);  add_415 = None
        mm_272: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1135, permute_746);  permute_746 = None
        permute_747: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1135, [1, 0])
        mm_273: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_747, view_392);  permute_747 = None
        sum_130: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1135, [0], True, dtype = torch.float32);  view_1135 = None
        view_1136: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_130, [16384]);  sum_130 = None
        convert_element_type_1838: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1136, torch.bfloat16);  view_1136 = None
        view_1137: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_272, [1, 128, 4096]);  mm_272 = None
        convert_element_type_1839: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1137, torch.float32);  view_1137 = None
        convert_element_type_1840: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_273, torch.float32);  mm_273 = None
        convert_element_type_1841: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1838, torch.float32);  convert_element_type_1838 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_274: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_743, view_414);  permute_743 = view_414 = None
        mm_275: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1132, permute_752);  view_1132 = permute_752 = None
        view_1139: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_275, [1, 128, 4096]);  mm_275 = None
        convert_element_type_1846: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_274, torch.float32);  mm_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1140: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1139, [1, 128, 16, 256]);  view_1139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_754: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1140, [0, 2, 1, 3]);  view_1140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1141: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_754, [16, 128, 256]);  permute_754 = None
        bmm_108: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_755, view_1141);  permute_755 = None
        bmm_109: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1141, permute_756);  view_1141 = permute_756 = None
        view_1142: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_108, [1, 16, 128, 256]);  bmm_108 = None
        view_1143: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_109, [1, 16, 128, 128]);  bmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1851: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1143, torch.float32);  view_1143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_573: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1851, div_29);  convert_element_type_1851 = None
        sum_131: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_573, [-1], True)
        neg_96: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_29);  div_29 = None
        fma_13: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_96, sum_131, mul_573);  neg_96 = sum_131 = mul_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1852: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_13, torch.bfloat16);  fma_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_85: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1852, 16.0);  convert_element_type_1852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1144: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_85, [16, 128, 128]);  div_85 = None
        bmm_110: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_757, view_1144);  permute_757 = None
        bmm_111: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1144, permute_758);  view_1144 = permute_758 = None
        view_1145: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_110, [1, 16, 256, 128]);  bmm_110 = None
        view_1146: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_111, [1, 16, 128, 256]);  bmm_111 = None
        convert_element_type_1858: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1145, torch.float32);  view_1145 = None
        permute_759: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1858, [0, 1, 3, 2]);  convert_element_type_1858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1859: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_759, torch.bfloat16);  permute_759 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_14: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1146, torch.bfloat16);  view_1146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_760: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_14, [0, 2, 1, 3]);  convert_element_type_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_761: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1859, [0, 2, 1, 3]);  convert_element_type_1859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_281: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_760, 3, 0, 64)
        slice_282: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_760, 3, 64, 256);  permute_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_283: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_761, 3, 0, 64)
        slice_284: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_761, 3, 64, 256);  permute_761 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_113: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_194, [1, 128, 1, 32, 2]);  unsqueeze_194 = None
        clone_113: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
        view_401: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [1, 128, 1, 64]);  clone_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_574: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_281, view_401)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1147: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_574, [1, 128, 16, 32, 2]);  mul_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_52: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1147, -1, 0)
        select_53: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1147, -1, 1);  view_1147 = None
        neg_97: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_52);  select_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_104: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_97, 3, 1, 9223372036854775807, 2);  neg_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_105: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_53, 3, 0, 9223372036854775807, 2);  select_53 = None
        add_416: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_104, slice_scatter_105);  slice_scatter_104 = slice_scatter_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_114: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_196, [1, 128, 1, 32, 2]);  unsqueeze_196 = None
        clone_114: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_114, memory_format = torch.contiguous_format);  expand_114 = None
        view_402: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_114, [1, 128, 1, 64]);  clone_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_575: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_281, view_402);  slice_281 = None
        add_417: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_416, mul_575);  add_416 = mul_575 = None
        mul_576: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_283, view_401);  view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1148: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_576, [1, 128, 16, 32, 2]);  mul_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_54: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1148, -1, 0)
        select_55: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1148, -1, 1);  view_1148 = None
        neg_98: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_54);  select_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_106: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_98, 3, 1, 9223372036854775807, 2);  neg_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_107: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_55, 3, 0, 9223372036854775807, 2);  select_55 = None
        add_418: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_106, slice_scatter_107);  slice_scatter_106 = slice_scatter_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_577: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_283, view_402);  slice_283 = view_402 = None
        add_419: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_418, mul_577);  add_418 = mul_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_108: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_282, 3, 64, 9223372036854775807);  slice_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_109: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_417, 3, 0, 64);  add_417 = None
        add_420: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_108, slice_scatter_109);  slice_scatter_108 = slice_scatter_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_110: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_284, 3, 64, 9223372036854775807);  slice_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_111: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_419, 3, 0, 64);  add_419 = None
        add_421: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_110, slice_scatter_111);  slice_scatter_110 = slice_scatter_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_762: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1142, [0, 2, 1, 3]);  view_1142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_238: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_762, memory_format = torch.contiguous_format);  permute_762 = None
        view_1149: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_238, [1, 128, 4096]);  clone_238 = None
        view_1150: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_421, [1, 128, 4096]);  add_421 = None
        view_1151: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_420, [1, 128, 4096]);  add_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1152: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1149, [128, 4096]);  view_1149 = None
        permute_763: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1152, [1, 0])
        mm_276: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_763, view_392);  permute_763 = None
        mm_277: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1152, permute_765);  view_1152 = permute_765 = None
        view_1153: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_277, [1, 128, 4096]);  mm_277 = None
        convert_element_type_1865: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1153, torch.float32);  view_1153 = None
        add_422: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1839, convert_element_type_1865);  convert_element_type_1839 = convert_element_type_1865 = None
        convert_element_type_1866: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_276, torch.float32);  mm_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1154: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1150, [128, 4096]);  view_1150 = None
        permute_767: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1154, [1, 0])
        mm_278: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_767, view_392);  permute_767 = None
        mm_279: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1154, permute_769);  view_1154 = permute_769 = None
        view_1155: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_279, [1, 128, 4096]);  mm_279 = None
        convert_element_type_1871: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1155, torch.float32);  view_1155 = None
        add_423: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_422, convert_element_type_1871);  add_422 = convert_element_type_1871 = None
        convert_element_type_1872: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_278, torch.float32);  mm_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1156: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1151, [128, 4096]);  view_1151 = None
        permute_771: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1156, [1, 0])
        mm_280: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_771, view_392);  permute_771 = view_392 = None
        mm_281: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1156, permute_773);  view_1156 = permute_773 = None
        view_1157: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_281, [1, 128, 4096]);  mm_281 = None
        convert_element_type_1877: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1157, torch.float32);  view_1157 = None
        add_424: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_423, convert_element_type_1877);  add_423 = convert_element_type_1877 = None
        convert_element_type_1878: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_280, torch.float32);  mm_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_579: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_424, primals_157);  primals_157 = None
        mul_580: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_579, 4096)
        sum_132: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_579, [2], True)
        mul_581: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_579, mul_140);  mul_579 = None
        sum_133: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_581, [2], True);  mul_581 = None
        mul_582: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, sum_133);  sum_133 = None
        sub_119: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_580, sum_132);  mul_580 = sum_132 = None
        sub_120: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_119, mul_582);  sub_119 = mul_582 = None
        mul_583: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_86, sub_120);  div_86 = sub_120 = None
        mul_584: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_424, mul_140);  mul_140 = None
        sum_134: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 1]);  mul_584 = None
        sum_135: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_424, [0, 1]);  add_424 = None
        add_425: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_413, mul_583);  add_413 = mul_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1879: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_425, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1158: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1879, [128, 4096]);  convert_element_type_1879 = None
        mm_282: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1158, permute_775);  permute_775 = None
        permute_776: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1158, [1, 0])
        mm_283: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_776, view_390);  view_390 = None
        sum_136: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1158, [0], True, dtype = torch.float32)
        view_1159: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_136, [4096]);  sum_136 = None
        convert_element_type_1884: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1159, torch.bfloat16);  view_1159 = None
        view_1160: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_282, [1, 128, 16384]);  mm_282 = None
        convert_element_type_1885: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1160, torch.float32);  view_1160 = None
        convert_element_type_1886: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_283, torch.float32);  mm_283 = None
        convert_element_type_1887: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1884, torch.float32);  convert_element_type_1884 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_389: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [1, 128, 16384]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_136: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_389, 0.5)
        mul_585: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1885, mul_136);  mul_136 = None
        convert_element_type_525: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_389, torch.float32)
        pow_14: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_525, 3.0)
        mul_137: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_125: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_389, mul_137);  view_389 = mul_137 = None
        mul_138: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_125, 0.7978845608028654);  add_125 = None
        tanh_13: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_138);  mul_138 = None
        add_126: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
        mul_586: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1885, add_126);  convert_element_type_1885 = add_126 = None
        convert_element_type_1888: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_586, torch.bfloat16);  mul_586 = None
        mul_587: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
        sub_121: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_587);  mul_587 = None
        mul_588: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_585, sub_121);  mul_585 = sub_121 = None
        mul_589: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_588, 0.7978845608028654);  mul_588 = None
        convert_element_type_1889: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_589, torch.bfloat16)
        mul_590: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_589, 0.044715);  mul_589 = None
        pow_43: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_525, 2.0);  convert_element_type_525 = None
        mul_591: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_43, 3.0);  pow_43 = None
        mul_592: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_590, mul_591);  mul_590 = mul_591 = None
        convert_element_type_1890: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_592, torch.bfloat16);  mul_592 = None
        add_426: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1889, convert_element_type_1890);  convert_element_type_1889 = convert_element_type_1890 = None
        mul_593: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1888, 0.5);  convert_element_type_1888 = None
        add_427: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_426, mul_593);  add_426 = mul_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1161: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_427, [128, 16384]);  add_427 = None
        mm_284: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1161, permute_779);  permute_779 = None
        permute_780: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1161, [1, 0])
        mm_285: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_780, view_364);  permute_780 = None
        sum_137: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1161, [0], True, dtype = torch.float32);  view_1161 = None
        view_1162: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_137, [16384]);  sum_137 = None
        convert_element_type_1895: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1162, torch.bfloat16);  view_1162 = None
        view_1163: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_284, [1, 128, 4096]);  mm_284 = None
        convert_element_type_1896: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1163, torch.float32);  view_1163 = None
        convert_element_type_1897: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_285, torch.float32);  mm_285 = None
        convert_element_type_1898: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1895, torch.float32);  convert_element_type_1895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_286: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_776, view_386);  permute_776 = view_386 = None
        mm_287: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1158, permute_785);  view_1158 = permute_785 = None
        view_1165: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_287, [1, 128, 4096]);  mm_287 = None
        convert_element_type_1903: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_286, torch.float32);  mm_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1166: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1165, [1, 128, 16, 256]);  view_1165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_787: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1166, [0, 2, 1, 3]);  view_1166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1167: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_787, [16, 128, 256]);  permute_787 = None
        bmm_112: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_788, view_1167);  permute_788 = None
        bmm_113: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1167, permute_789);  view_1167 = permute_789 = None
        view_1168: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_112, [1, 16, 128, 256]);  bmm_112 = None
        view_1169: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_113, [1, 16, 128, 128]);  bmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1908: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1169, torch.float32);  view_1169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_594: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1908, div_27);  convert_element_type_1908 = None
        sum_138: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_594, [-1], True)
        neg_99: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_27);  div_27 = None
        fma_14: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_99, sum_138, mul_594);  neg_99 = sum_138 = mul_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1909: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_14, torch.bfloat16);  fma_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_87: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1909, 16.0);  convert_element_type_1909 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1170: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_87, [16, 128, 128]);  div_87 = None
        bmm_114: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_790, view_1170);  permute_790 = None
        bmm_115: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1170, permute_791);  view_1170 = permute_791 = None
        view_1171: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_114, [1, 16, 256, 128]);  bmm_114 = None
        view_1172: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_115, [1, 16, 128, 256]);  bmm_115 = None
        convert_element_type_1915: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1171, torch.float32);  view_1171 = None
        permute_792: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1915, [0, 1, 3, 2]);  convert_element_type_1915 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1916: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_792, torch.bfloat16);  permute_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_13: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1172, torch.bfloat16);  view_1172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_793: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_13, [0, 2, 1, 3]);  convert_element_type_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_794: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1916, [0, 2, 1, 3]);  convert_element_type_1916 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_285: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_793, 3, 0, 64)
        slice_286: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_793, 3, 64, 256);  permute_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_287: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_794, 3, 0, 64)
        slice_288: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_794, 3, 64, 256);  permute_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_105: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_181, [1, 128, 1, 32, 2]);  unsqueeze_181 = None
        clone_105: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_373: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_105, [1, 128, 1, 64]);  clone_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_595: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_285, view_373)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1173: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_595, [1, 128, 16, 32, 2]);  mul_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_56: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1173, -1, 0)
        select_57: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1173, -1, 1);  view_1173 = None
        neg_100: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_56);  select_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_112: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_100, 3, 1, 9223372036854775807, 2);  neg_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_113: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_57, 3, 0, 9223372036854775807, 2);  select_57 = None
        add_428: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_112, slice_scatter_113);  slice_scatter_112 = slice_scatter_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_106: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_183, [1, 128, 1, 32, 2]);  unsqueeze_183 = None
        clone_106: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_106, memory_format = torch.contiguous_format);  expand_106 = None
        view_374: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [1, 128, 1, 64]);  clone_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_596: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_285, view_374);  slice_285 = None
        add_429: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_428, mul_596);  add_428 = mul_596 = None
        mul_597: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_287, view_373);  view_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1174: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_597, [1, 128, 16, 32, 2]);  mul_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_58: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1174, -1, 0)
        select_59: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1174, -1, 1);  view_1174 = None
        neg_101: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_58);  select_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_114: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_101, 3, 1, 9223372036854775807, 2);  neg_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_115: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_59, 3, 0, 9223372036854775807, 2);  select_59 = None
        add_430: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_114, slice_scatter_115);  slice_scatter_114 = slice_scatter_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_598: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_287, view_374);  slice_287 = view_374 = None
        add_431: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_430, mul_598);  add_430 = mul_598 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_116: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_286, 3, 64, 9223372036854775807);  slice_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_117: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_429, 3, 0, 64);  add_429 = None
        add_432: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_116, slice_scatter_117);  slice_scatter_116 = slice_scatter_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_118: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_288, 3, 64, 9223372036854775807);  slice_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_119: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_431, 3, 0, 64);  add_431 = None
        add_433: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_118, slice_scatter_119);  slice_scatter_118 = slice_scatter_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_795: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1168, [0, 2, 1, 3]);  view_1168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_239: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_795, memory_format = torch.contiguous_format);  permute_795 = None
        view_1175: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_239, [1, 128, 4096]);  clone_239 = None
        view_1176: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_433, [1, 128, 4096]);  add_433 = None
        view_1177: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_432, [1, 128, 4096]);  add_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1178: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1175, [128, 4096]);  view_1175 = None
        permute_796: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1178, [1, 0])
        mm_288: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_796, view_364);  permute_796 = None
        mm_289: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1178, permute_798);  view_1178 = permute_798 = None
        view_1179: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_289, [1, 128, 4096]);  mm_289 = None
        convert_element_type_1922: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1179, torch.float32);  view_1179 = None
        add_434: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1896, convert_element_type_1922);  convert_element_type_1896 = convert_element_type_1922 = None
        convert_element_type_1923: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_288, torch.float32);  mm_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1180: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1176, [128, 4096]);  view_1176 = None
        permute_800: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1180, [1, 0])
        mm_290: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_800, view_364);  permute_800 = None
        mm_291: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1180, permute_802);  view_1180 = permute_802 = None
        view_1181: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_291, [1, 128, 4096]);  mm_291 = None
        convert_element_type_1928: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1181, torch.float32);  view_1181 = None
        add_435: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_434, convert_element_type_1928);  add_434 = convert_element_type_1928 = None
        convert_element_type_1929: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_290, torch.float32);  mm_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1182: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1177, [128, 4096]);  view_1177 = None
        permute_804: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1182, [1, 0])
        mm_292: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_804, view_364);  permute_804 = view_364 = None
        mm_293: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1182, permute_806);  view_1182 = permute_806 = None
        view_1183: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_293, [1, 128, 4096]);  mm_293 = None
        convert_element_type_1934: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1183, torch.float32);  view_1183 = None
        add_436: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_435, convert_element_type_1934);  add_435 = convert_element_type_1934 = None
        convert_element_type_1935: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_292, torch.float32);  mm_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_600: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_436, primals_146);  primals_146 = None
        mul_601: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_600, 4096)
        sum_139: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_600, [2], True)
        mul_602: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_600, mul_130);  mul_600 = None
        sum_140: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_602, [2], True);  mul_602 = None
        mul_603: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, sum_140);  sum_140 = None
        sub_123: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_601, sum_139);  mul_601 = sum_139 = None
        sub_124: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_123, mul_603);  sub_123 = mul_603 = None
        mul_604: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_88, sub_124);  div_88 = sub_124 = None
        mul_605: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_436, mul_130);  mul_130 = None
        sum_141: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_605, [0, 1]);  mul_605 = None
        sum_142: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_436, [0, 1]);  add_436 = None
        add_437: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_425, mul_604);  add_425 = mul_604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1936: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_437, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1184: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1936, [128, 4096]);  convert_element_type_1936 = None
        mm_294: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1184, permute_808);  permute_808 = None
        permute_809: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1184, [1, 0])
        mm_295: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_809, view_362);  view_362 = None
        sum_143: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1184, [0], True, dtype = torch.float32)
        view_1185: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_143, [4096]);  sum_143 = None
        convert_element_type_1941: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1185, torch.bfloat16);  view_1185 = None
        view_1186: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_294, [1, 128, 16384]);  mm_294 = None
        convert_element_type_1942: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1186, torch.float32);  view_1186 = None
        convert_element_type_1943: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_295, torch.float32);  mm_295 = None
        convert_element_type_1944: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1941, torch.float32);  convert_element_type_1941 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_361: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [1, 128, 16384]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_126: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_361, 0.5)
        mul_606: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1942, mul_126);  mul_126 = None
        convert_element_type_487: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_361, torch.float32)
        pow_13: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_487, 3.0)
        mul_127: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_116: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_361, mul_127);  view_361 = mul_127 = None
        mul_128: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_116, 0.7978845608028654);  add_116 = None
        tanh_12: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_128);  mul_128 = None
        add_117: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
        mul_607: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1942, add_117);  convert_element_type_1942 = add_117 = None
        convert_element_type_1945: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_607, torch.bfloat16);  mul_607 = None
        mul_608: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
        sub_125: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_608);  mul_608 = None
        mul_609: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_606, sub_125);  mul_606 = sub_125 = None
        mul_610: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_609, 0.7978845608028654);  mul_609 = None
        convert_element_type_1946: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_610, torch.bfloat16)
        mul_611: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_610, 0.044715);  mul_610 = None
        pow_44: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_487, 2.0);  convert_element_type_487 = None
        mul_612: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_44, 3.0);  pow_44 = None
        mul_613: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_611, mul_612);  mul_611 = mul_612 = None
        convert_element_type_1947: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_613, torch.bfloat16);  mul_613 = None
        add_438: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1946, convert_element_type_1947);  convert_element_type_1946 = convert_element_type_1947 = None
        mul_614: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1945, 0.5);  convert_element_type_1945 = None
        add_439: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_438, mul_614);  add_438 = mul_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1187: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_439, [128, 16384]);  add_439 = None
        mm_296: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1187, permute_812);  permute_812 = None
        permute_813: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1187, [1, 0])
        mm_297: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_813, view_336);  permute_813 = None
        sum_144: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1187, [0], True, dtype = torch.float32);  view_1187 = None
        view_1188: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_144, [16384]);  sum_144 = None
        convert_element_type_1952: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1188, torch.bfloat16);  view_1188 = None
        view_1189: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_296, [1, 128, 4096]);  mm_296 = None
        convert_element_type_1953: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1189, torch.float32);  view_1189 = None
        convert_element_type_1954: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_297, torch.float32);  mm_297 = None
        convert_element_type_1955: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1952, torch.float32);  convert_element_type_1952 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_298: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_809, view_358);  permute_809 = view_358 = None
        mm_299: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1184, permute_818);  view_1184 = permute_818 = None
        view_1191: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_299, [1, 128, 4096]);  mm_299 = None
        convert_element_type_1960: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_298, torch.float32);  mm_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1192: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1191, [1, 128, 16, 256]);  view_1191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_820: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1192, [0, 2, 1, 3]);  view_1192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1193: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_820, [16, 128, 256]);  permute_820 = None
        bmm_116: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_821, view_1193);  permute_821 = None
        bmm_117: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1193, permute_822);  view_1193 = permute_822 = None
        view_1194: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_116, [1, 16, 128, 256]);  bmm_116 = None
        view_1195: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_117, [1, 16, 128, 128]);  bmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1965: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1195, torch.float32);  view_1195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_615: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1965, div_25);  convert_element_type_1965 = None
        sum_145: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_615, [-1], True)
        neg_102: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_25);  div_25 = None
        fma_15: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_102, sum_145, mul_615);  neg_102 = sum_145 = mul_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1966: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_15, torch.bfloat16);  fma_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_89: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1966, 16.0);  convert_element_type_1966 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1196: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_89, [16, 128, 128]);  div_89 = None
        bmm_118: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_823, view_1196);  permute_823 = None
        bmm_119: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1196, permute_824);  view_1196 = permute_824 = None
        view_1197: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_118, [1, 16, 256, 128]);  bmm_118 = None
        view_1198: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_119, [1, 16, 128, 256]);  bmm_119 = None
        convert_element_type_1972: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1197, torch.float32);  view_1197 = None
        permute_825: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1972, [0, 1, 3, 2]);  convert_element_type_1972 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1973: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_825, torch.bfloat16);  permute_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_12: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1198, torch.bfloat16);  view_1198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_826: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_12, [0, 2, 1, 3]);  convert_element_type_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_827: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1973, [0, 2, 1, 3]);  convert_element_type_1973 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_289: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_826, 3, 0, 64)
        slice_290: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_826, 3, 64, 256);  permute_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_291: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_827, 3, 0, 64)
        slice_292: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_827, 3, 64, 256);  permute_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_97: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_168, [1, 128, 1, 32, 2]);  unsqueeze_168 = None
        clone_97: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_345: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_97, [1, 128, 1, 64]);  clone_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_616: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_289, view_345)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1199: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_616, [1, 128, 16, 32, 2]);  mul_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_60: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1199, -1, 0)
        select_61: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1199, -1, 1);  view_1199 = None
        neg_103: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_60);  select_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_120: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_103, 3, 1, 9223372036854775807, 2);  neg_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_121: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_61, 3, 0, 9223372036854775807, 2);  select_61 = None
        add_440: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_120, slice_scatter_121);  slice_scatter_120 = slice_scatter_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_98: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_170, [1, 128, 1, 32, 2]);  unsqueeze_170 = None
        clone_98: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
        view_346: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_98, [1, 128, 1, 64]);  clone_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_617: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_289, view_346);  slice_289 = None
        add_441: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_440, mul_617);  add_440 = mul_617 = None
        mul_618: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_291, view_345);  view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1200: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_618, [1, 128, 16, 32, 2]);  mul_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_62: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1200, -1, 0)
        select_63: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1200, -1, 1);  view_1200 = None
        neg_104: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_62);  select_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_122: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_104, 3, 1, 9223372036854775807, 2);  neg_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_123: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_63, 3, 0, 9223372036854775807, 2);  select_63 = None
        add_442: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_122, slice_scatter_123);  slice_scatter_122 = slice_scatter_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_619: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_291, view_346);  slice_291 = view_346 = None
        add_443: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_442, mul_619);  add_442 = mul_619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_124: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_290, 3, 64, 9223372036854775807);  slice_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_125: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_441, 3, 0, 64);  add_441 = None
        add_444: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_124, slice_scatter_125);  slice_scatter_124 = slice_scatter_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_126: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_292, 3, 64, 9223372036854775807);  slice_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_127: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_443, 3, 0, 64);  add_443 = None
        add_445: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_126, slice_scatter_127);  slice_scatter_126 = slice_scatter_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_828: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1194, [0, 2, 1, 3]);  view_1194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_240: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_828, memory_format = torch.contiguous_format);  permute_828 = None
        view_1201: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_240, [1, 128, 4096]);  clone_240 = None
        view_1202: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_445, [1, 128, 4096]);  add_445 = None
        view_1203: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_444, [1, 128, 4096]);  add_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1204: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1201, [128, 4096]);  view_1201 = None
        permute_829: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1204, [1, 0])
        mm_300: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_829, view_336);  permute_829 = None
        mm_301: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1204, permute_831);  view_1204 = permute_831 = None
        view_1205: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_301, [1, 128, 4096]);  mm_301 = None
        convert_element_type_1979: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1205, torch.float32);  view_1205 = None
        add_446: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1953, convert_element_type_1979);  convert_element_type_1953 = convert_element_type_1979 = None
        convert_element_type_1980: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_300, torch.float32);  mm_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1206: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1202, [128, 4096]);  view_1202 = None
        permute_833: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1206, [1, 0])
        mm_302: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_833, view_336);  permute_833 = None
        mm_303: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1206, permute_835);  view_1206 = permute_835 = None
        view_1207: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_303, [1, 128, 4096]);  mm_303 = None
        convert_element_type_1985: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1207, torch.float32);  view_1207 = None
        add_447: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_446, convert_element_type_1985);  add_446 = convert_element_type_1985 = None
        convert_element_type_1986: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_302, torch.float32);  mm_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1208: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1203, [128, 4096]);  view_1203 = None
        permute_837: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1208, [1, 0])
        mm_304: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_837, view_336);  permute_837 = view_336 = None
        mm_305: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1208, permute_839);  view_1208 = permute_839 = None
        view_1209: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_305, [1, 128, 4096]);  mm_305 = None
        convert_element_type_1991: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1209, torch.float32);  view_1209 = None
        add_448: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_447, convert_element_type_1991);  add_447 = convert_element_type_1991 = None
        convert_element_type_1992: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_304, torch.float32);  mm_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_621: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_448, primals_135);  primals_135 = None
        mul_622: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_621, 4096)
        sum_146: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_621, [2], True)
        mul_623: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_621, mul_120);  mul_621 = None
        sum_147: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_623, [2], True);  mul_623 = None
        mul_624: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, sum_147);  sum_147 = None
        sub_127: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_622, sum_146);  mul_622 = sum_146 = None
        sub_128: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_127, mul_624);  sub_127 = mul_624 = None
        mul_625: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_90, sub_128);  div_90 = sub_128 = None
        mul_626: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_448, mul_120);  mul_120 = None
        sum_148: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 1]);  mul_626 = None
        sum_149: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_448, [0, 1]);  add_448 = None
        add_449: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_437, mul_625);  add_437 = mul_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1993: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_449, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1210: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1993, [128, 4096]);  convert_element_type_1993 = None
        mm_306: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1210, permute_841);  permute_841 = None
        permute_842: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1210, [1, 0])
        mm_307: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_842, view_334);  view_334 = None
        sum_150: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True, dtype = torch.float32)
        view_1211: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_150, [4096]);  sum_150 = None
        convert_element_type_1998: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1211, torch.bfloat16);  view_1211 = None
        view_1212: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_306, [1, 128, 16384]);  mm_306 = None
        convert_element_type_1999: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1212, torch.float32);  view_1212 = None
        convert_element_type_2000: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_307, torch.float32);  mm_307 = None
        convert_element_type_2001: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1998, torch.float32);  convert_element_type_1998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_333: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [1, 128, 16384]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_333, 0.5)
        mul_627: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1999, mul_116);  mul_116 = None
        convert_element_type_449: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_333, torch.float32)
        pow_12: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_449, 3.0)
        mul_117: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_107: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_333, mul_117);  view_333 = mul_117 = None
        mul_118: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_107, 0.7978845608028654);  add_107 = None
        tanh_11: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_108: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_628: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1999, add_108);  convert_element_type_1999 = add_108 = None
        convert_element_type_2002: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_628, torch.bfloat16);  mul_628 = None
        mul_629: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_129: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_629);  mul_629 = None
        mul_630: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_627, sub_129);  mul_627 = sub_129 = None
        mul_631: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_630, 0.7978845608028654);  mul_630 = None
        convert_element_type_2003: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_631, torch.bfloat16)
        mul_632: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_631, 0.044715);  mul_631 = None
        pow_45: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_449, 2.0);  convert_element_type_449 = None
        mul_633: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_45, 3.0);  pow_45 = None
        mul_634: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_632, mul_633);  mul_632 = mul_633 = None
        convert_element_type_2004: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_634, torch.bfloat16);  mul_634 = None
        add_450: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2003, convert_element_type_2004);  convert_element_type_2003 = convert_element_type_2004 = None
        mul_635: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2002, 0.5);  convert_element_type_2002 = None
        add_451: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_450, mul_635);  add_450 = mul_635 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1213: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_451, [128, 16384]);  add_451 = None
        mm_308: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1213, permute_845);  permute_845 = None
        permute_846: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1213, [1, 0])
        mm_309: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_846, view_308);  permute_846 = None
        sum_151: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1213, [0], True, dtype = torch.float32);  view_1213 = None
        view_1214: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_151, [16384]);  sum_151 = None
        convert_element_type_2009: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1214, torch.bfloat16);  view_1214 = None
        view_1215: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_308, [1, 128, 4096]);  mm_308 = None
        convert_element_type_2010: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1215, torch.float32);  view_1215 = None
        convert_element_type_2011: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_309, torch.float32);  mm_309 = None
        convert_element_type_2012: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2009, torch.float32);  convert_element_type_2009 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_310: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_842, view_330);  permute_842 = view_330 = None
        mm_311: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1210, permute_851);  view_1210 = permute_851 = None
        view_1217: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_311, [1, 128, 4096]);  mm_311 = None
        convert_element_type_2017: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_310, torch.float32);  mm_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1218: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1217, [1, 128, 16, 256]);  view_1217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_853: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1218, [0, 2, 1, 3]);  view_1218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1219: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_853, [16, 128, 256]);  permute_853 = None
        bmm_120: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_854, view_1219);  permute_854 = None
        bmm_121: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1219, permute_855);  view_1219 = permute_855 = None
        view_1220: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_120, [1, 16, 128, 256]);  bmm_120 = None
        view_1221: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_121, [1, 16, 128, 128]);  bmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2022: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1221, torch.float32);  view_1221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_636: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2022, div_23);  convert_element_type_2022 = None
        sum_152: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_636, [-1], True)
        neg_105: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_23);  div_23 = None
        fma_16: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_105, sum_152, mul_636);  neg_105 = sum_152 = mul_636 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2023: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_16, torch.bfloat16);  fma_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_91: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2023, 16.0);  convert_element_type_2023 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1222: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_91, [16, 128, 128]);  div_91 = None
        bmm_122: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_856, view_1222);  permute_856 = None
        bmm_123: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1222, permute_857);  view_1222 = permute_857 = None
        view_1223: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_122, [1, 16, 256, 128]);  bmm_122 = None
        view_1224: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_123, [1, 16, 128, 256]);  bmm_123 = None
        convert_element_type_2029: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1223, torch.float32);  view_1223 = None
        permute_858: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2029, [0, 1, 3, 2]);  convert_element_type_2029 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2030: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_858, torch.bfloat16);  permute_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_11: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1224, torch.bfloat16);  view_1224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_859: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_11, [0, 2, 1, 3]);  convert_element_type_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_860: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2030, [0, 2, 1, 3]);  convert_element_type_2030 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_293: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_859, 3, 0, 64)
        slice_294: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_859, 3, 64, 256);  permute_859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_295: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_860, 3, 0, 64)
        slice_296: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_860, 3, 64, 256);  permute_860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_89: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_155, [1, 128, 1, 32, 2]);  unsqueeze_155 = None
        clone_89: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_317: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [1, 128, 1, 64]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_637: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_293, view_317)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1225: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_637, [1, 128, 16, 32, 2]);  mul_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_64: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1225, -1, 0)
        select_65: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1225, -1, 1);  view_1225 = None
        neg_106: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_64);  select_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_128: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_106, 3, 1, 9223372036854775807, 2);  neg_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_129: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_65, 3, 0, 9223372036854775807, 2);  select_65 = None
        add_452: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_128, slice_scatter_129);  slice_scatter_128 = slice_scatter_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_90: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_157, [1, 128, 1, 32, 2]);  unsqueeze_157 = None
        clone_90: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_318: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [1, 128, 1, 64]);  clone_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_638: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_293, view_318);  slice_293 = None
        add_453: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_452, mul_638);  add_452 = mul_638 = None
        mul_639: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_295, view_317);  view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1226: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_639, [1, 128, 16, 32, 2]);  mul_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_66: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1226, -1, 0)
        select_67: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1226, -1, 1);  view_1226 = None
        neg_107: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_66);  select_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_130: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_107, 3, 1, 9223372036854775807, 2);  neg_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_131: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_67, 3, 0, 9223372036854775807, 2);  select_67 = None
        add_454: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_130, slice_scatter_131);  slice_scatter_130 = slice_scatter_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_640: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_295, view_318);  slice_295 = view_318 = None
        add_455: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_454, mul_640);  add_454 = mul_640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_132: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_294, 3, 64, 9223372036854775807);  slice_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_133: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_453, 3, 0, 64);  add_453 = None
        add_456: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_132, slice_scatter_133);  slice_scatter_132 = slice_scatter_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_134: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_296, 3, 64, 9223372036854775807);  slice_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_135: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_455, 3, 0, 64);  add_455 = None
        add_457: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_134, slice_scatter_135);  slice_scatter_134 = slice_scatter_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_861: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1220, [0, 2, 1, 3]);  view_1220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_241: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_861, memory_format = torch.contiguous_format);  permute_861 = None
        view_1227: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_241, [1, 128, 4096]);  clone_241 = None
        view_1228: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_457, [1, 128, 4096]);  add_457 = None
        view_1229: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_456, [1, 128, 4096]);  add_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1230: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1227, [128, 4096]);  view_1227 = None
        permute_862: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1230, [1, 0])
        mm_312: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_862, view_308);  permute_862 = None
        mm_313: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1230, permute_864);  view_1230 = permute_864 = None
        view_1231: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_313, [1, 128, 4096]);  mm_313 = None
        convert_element_type_2036: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1231, torch.float32);  view_1231 = None
        add_458: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2010, convert_element_type_2036);  convert_element_type_2010 = convert_element_type_2036 = None
        convert_element_type_2037: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_312, torch.float32);  mm_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1232: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1228, [128, 4096]);  view_1228 = None
        permute_866: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1232, [1, 0])
        mm_314: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_866, view_308);  permute_866 = None
        mm_315: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1232, permute_868);  view_1232 = permute_868 = None
        view_1233: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_315, [1, 128, 4096]);  mm_315 = None
        convert_element_type_2042: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1233, torch.float32);  view_1233 = None
        add_459: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_458, convert_element_type_2042);  add_458 = convert_element_type_2042 = None
        convert_element_type_2043: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_314, torch.float32);  mm_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1234: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1229, [128, 4096]);  view_1229 = None
        permute_870: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1234, [1, 0])
        mm_316: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_870, view_308);  permute_870 = view_308 = None
        mm_317: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1234, permute_872);  view_1234 = permute_872 = None
        view_1235: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_317, [1, 128, 4096]);  mm_317 = None
        convert_element_type_2048: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1235, torch.float32);  view_1235 = None
        add_460: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_459, convert_element_type_2048);  add_459 = convert_element_type_2048 = None
        convert_element_type_2049: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_316, torch.float32);  mm_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_642: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_460, primals_124);  primals_124 = None
        mul_643: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_642, 4096)
        sum_153: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_642, [2], True)
        mul_644: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_642, mul_110);  mul_642 = None
        sum_154: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_644, [2], True);  mul_644 = None
        mul_645: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, sum_154);  sum_154 = None
        sub_131: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_643, sum_153);  mul_643 = sum_153 = None
        sub_132: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_131, mul_645);  sub_131 = mul_645 = None
        mul_646: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_92, sub_132);  div_92 = sub_132 = None
        mul_647: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_460, mul_110);  mul_110 = None
        sum_155: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_647, [0, 1]);  mul_647 = None
        sum_156: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_460, [0, 1]);  add_460 = None
        add_461: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_449, mul_646);  add_449 = mul_646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2050: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_461, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1236: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2050, [128, 4096]);  convert_element_type_2050 = None
        mm_318: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1236, permute_874);  permute_874 = None
        permute_875: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1236, [1, 0])
        mm_319: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_875, view_306);  view_306 = None
        sum_157: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1236, [0], True, dtype = torch.float32)
        view_1237: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_157, [4096]);  sum_157 = None
        convert_element_type_2055: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1237, torch.bfloat16);  view_1237 = None
        view_1238: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_318, [1, 128, 16384]);  mm_318 = None
        convert_element_type_2056: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1238, torch.float32);  view_1238 = None
        convert_element_type_2057: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_319, torch.float32);  mm_319 = None
        convert_element_type_2058: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2055, torch.float32);  convert_element_type_2055 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_305: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [1, 128, 16384]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_106: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_305, 0.5)
        mul_648: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2056, mul_106);  mul_106 = None
        convert_element_type_411: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_305, torch.float32)
        pow_11: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_411, 3.0)
        mul_107: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_98: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_305, mul_107);  view_305 = mul_107 = None
        mul_108: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_98, 0.7978845608028654);  add_98 = None
        tanh_10: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_108);  mul_108 = None
        add_99: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_649: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2056, add_99);  convert_element_type_2056 = add_99 = None
        convert_element_type_2059: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_649, torch.bfloat16);  mul_649 = None
        mul_650: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_133: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_650);  mul_650 = None
        mul_651: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_648, sub_133);  mul_648 = sub_133 = None
        mul_652: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_651, 0.7978845608028654);  mul_651 = None
        convert_element_type_2060: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_652, torch.bfloat16)
        mul_653: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_652, 0.044715);  mul_652 = None
        pow_46: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_411, 2.0);  convert_element_type_411 = None
        mul_654: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_46, 3.0);  pow_46 = None
        mul_655: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_653, mul_654);  mul_653 = mul_654 = None
        convert_element_type_2061: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_655, torch.bfloat16);  mul_655 = None
        add_462: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2060, convert_element_type_2061);  convert_element_type_2060 = convert_element_type_2061 = None
        mul_656: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2059, 0.5);  convert_element_type_2059 = None
        add_463: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_462, mul_656);  add_462 = mul_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1239: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_463, [128, 16384]);  add_463 = None
        mm_320: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1239, permute_878);  permute_878 = None
        permute_879: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1239, [1, 0])
        mm_321: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_879, view_280);  permute_879 = None
        sum_158: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1239, [0], True, dtype = torch.float32);  view_1239 = None
        view_1240: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_158, [16384]);  sum_158 = None
        convert_element_type_2066: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1240, torch.bfloat16);  view_1240 = None
        view_1241: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_320, [1, 128, 4096]);  mm_320 = None
        convert_element_type_2067: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1241, torch.float32);  view_1241 = None
        convert_element_type_2068: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_321, torch.float32);  mm_321 = None
        convert_element_type_2069: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2066, torch.float32);  convert_element_type_2066 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_322: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_875, view_302);  permute_875 = view_302 = None
        mm_323: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1236, permute_884);  view_1236 = permute_884 = None
        view_1243: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_323, [1, 128, 4096]);  mm_323 = None
        convert_element_type_2074: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_322, torch.float32);  mm_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1244: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1243, [1, 128, 16, 256]);  view_1243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_886: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1244, [0, 2, 1, 3]);  view_1244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1245: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_886, [16, 128, 256]);  permute_886 = None
        bmm_124: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_887, view_1245);  permute_887 = None
        bmm_125: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1245, permute_888);  view_1245 = permute_888 = None
        view_1246: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_124, [1, 16, 128, 256]);  bmm_124 = None
        view_1247: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_125, [1, 16, 128, 128]);  bmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2079: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1247, torch.float32);  view_1247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_657: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2079, div_21);  convert_element_type_2079 = None
        sum_159: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_657, [-1], True)
        neg_108: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_21);  div_21 = None
        fma_17: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_108, sum_159, mul_657);  neg_108 = sum_159 = mul_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2080: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_17, torch.bfloat16);  fma_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_93: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2080, 16.0);  convert_element_type_2080 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1248: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_93, [16, 128, 128]);  div_93 = None
        bmm_126: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_889, view_1248);  permute_889 = None
        bmm_127: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1248, permute_890);  view_1248 = permute_890 = None
        view_1249: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_126, [1, 16, 256, 128]);  bmm_126 = None
        view_1250: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_127, [1, 16, 128, 256]);  bmm_127 = None
        convert_element_type_2086: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1249, torch.float32);  view_1249 = None
        permute_891: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2086, [0, 1, 3, 2]);  convert_element_type_2086 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2087: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_891, torch.bfloat16);  permute_891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_10: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1250, torch.bfloat16);  view_1250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_892: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_10, [0, 2, 1, 3]);  convert_element_type_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_893: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2087, [0, 2, 1, 3]);  convert_element_type_2087 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_297: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_892, 3, 0, 64)
        slice_298: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_892, 3, 64, 256);  permute_892 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_299: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_893, 3, 0, 64)
        slice_300: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_893, 3, 64, 256);  permute_893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_81: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_142, [1, 128, 1, 32, 2]);  unsqueeze_142 = None
        clone_81: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_289: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [1, 128, 1, 64]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_658: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_297, view_289)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1251: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_658, [1, 128, 16, 32, 2]);  mul_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_68: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1251, -1, 0)
        select_69: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1251, -1, 1);  view_1251 = None
        neg_109: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_68);  select_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_136: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_109, 3, 1, 9223372036854775807, 2);  neg_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_137: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_69, 3, 0, 9223372036854775807, 2);  select_69 = None
        add_464: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_136, slice_scatter_137);  slice_scatter_136 = slice_scatter_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_82: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_144, [1, 128, 1, 32, 2]);  unsqueeze_144 = None
        clone_82: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_290: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [1, 128, 1, 64]);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_659: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_297, view_290);  slice_297 = None
        add_465: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_464, mul_659);  add_464 = mul_659 = None
        mul_660: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_299, view_289);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1252: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_660, [1, 128, 16, 32, 2]);  mul_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_70: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1252, -1, 0)
        select_71: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1252, -1, 1);  view_1252 = None
        neg_110: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_70);  select_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_138: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_110, 3, 1, 9223372036854775807, 2);  neg_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_139: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_71, 3, 0, 9223372036854775807, 2);  select_71 = None
        add_466: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_138, slice_scatter_139);  slice_scatter_138 = slice_scatter_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_661: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_299, view_290);  slice_299 = view_290 = None
        add_467: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_466, mul_661);  add_466 = mul_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_140: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_298, 3, 64, 9223372036854775807);  slice_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_141: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_465, 3, 0, 64);  add_465 = None
        add_468: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_140, slice_scatter_141);  slice_scatter_140 = slice_scatter_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_142: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_300, 3, 64, 9223372036854775807);  slice_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_143: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_467, 3, 0, 64);  add_467 = None
        add_469: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_142, slice_scatter_143);  slice_scatter_142 = slice_scatter_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_894: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1246, [0, 2, 1, 3]);  view_1246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_242: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_894, memory_format = torch.contiguous_format);  permute_894 = None
        view_1253: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_242, [1, 128, 4096]);  clone_242 = None
        view_1254: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_469, [1, 128, 4096]);  add_469 = None
        view_1255: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_468, [1, 128, 4096]);  add_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1256: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1253, [128, 4096]);  view_1253 = None
        permute_895: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1256, [1, 0])
        mm_324: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_895, view_280);  permute_895 = None
        mm_325: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1256, permute_897);  view_1256 = permute_897 = None
        view_1257: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_325, [1, 128, 4096]);  mm_325 = None
        convert_element_type_2093: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1257, torch.float32);  view_1257 = None
        add_470: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2067, convert_element_type_2093);  convert_element_type_2067 = convert_element_type_2093 = None
        convert_element_type_2094: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_324, torch.float32);  mm_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1258: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1254, [128, 4096]);  view_1254 = None
        permute_899: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1258, [1, 0])
        mm_326: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_899, view_280);  permute_899 = None
        mm_327: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1258, permute_901);  view_1258 = permute_901 = None
        view_1259: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_327, [1, 128, 4096]);  mm_327 = None
        convert_element_type_2099: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1259, torch.float32);  view_1259 = None
        add_471: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_470, convert_element_type_2099);  add_470 = convert_element_type_2099 = None
        convert_element_type_2100: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_326, torch.float32);  mm_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1260: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1255, [128, 4096]);  view_1255 = None
        permute_903: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1260, [1, 0])
        mm_328: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_903, view_280);  permute_903 = view_280 = None
        mm_329: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1260, permute_905);  view_1260 = permute_905 = None
        view_1261: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_329, [1, 128, 4096]);  mm_329 = None
        convert_element_type_2105: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1261, torch.float32);  view_1261 = None
        add_472: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_471, convert_element_type_2105);  add_471 = convert_element_type_2105 = None
        convert_element_type_2106: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_328, torch.float32);  mm_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_663: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_472, primals_113);  primals_113 = None
        mul_664: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_663, 4096)
        sum_160: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_663, [2], True)
        mul_665: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_663, mul_100);  mul_663 = None
        sum_161: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_665, [2], True);  mul_665 = None
        mul_666: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, sum_161);  sum_161 = None
        sub_135: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_664, sum_160);  mul_664 = sum_160 = None
        sub_136: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_135, mul_666);  sub_135 = mul_666 = None
        mul_667: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_94, sub_136);  div_94 = sub_136 = None
        mul_668: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_472, mul_100);  mul_100 = None
        sum_162: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_668, [0, 1]);  mul_668 = None
        sum_163: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_472, [0, 1]);  add_472 = None
        add_473: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_461, mul_667);  add_461 = mul_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2107: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_473, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1262: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2107, [128, 4096]);  convert_element_type_2107 = None
        mm_330: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1262, permute_907);  permute_907 = None
        permute_908: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1262, [1, 0])
        mm_331: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_908, view_278);  view_278 = None
        sum_164: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1262, [0], True, dtype = torch.float32)
        view_1263: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_164, [4096]);  sum_164 = None
        convert_element_type_2112: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1263, torch.bfloat16);  view_1263 = None
        view_1264: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_330, [1, 128, 16384]);  mm_330 = None
        convert_element_type_2113: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1264, torch.float32);  view_1264 = None
        convert_element_type_2114: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_331, torch.float32);  mm_331 = None
        convert_element_type_2115: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2112, torch.float32);  convert_element_type_2112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_277: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [1, 128, 16384]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_96: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_277, 0.5)
        mul_669: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2113, mul_96);  mul_96 = None
        convert_element_type_373: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_277, torch.float32)
        pow_10: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_373, 3.0)
        mul_97: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_89: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_277, mul_97);  view_277 = mul_97 = None
        mul_98: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_9: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_98);  mul_98 = None
        add_90: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_670: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2113, add_90);  convert_element_type_2113 = add_90 = None
        convert_element_type_2116: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_670, torch.bfloat16);  mul_670 = None
        mul_671: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_137: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_671);  mul_671 = None
        mul_672: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_669, sub_137);  mul_669 = sub_137 = None
        mul_673: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_672, 0.7978845608028654);  mul_672 = None
        convert_element_type_2117: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_673, torch.bfloat16)
        mul_674: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_673, 0.044715);  mul_673 = None
        pow_47: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_373, 2.0);  convert_element_type_373 = None
        mul_675: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_47, 3.0);  pow_47 = None
        mul_676: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_674, mul_675);  mul_674 = mul_675 = None
        convert_element_type_2118: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_676, torch.bfloat16);  mul_676 = None
        add_474: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2117, convert_element_type_2118);  convert_element_type_2117 = convert_element_type_2118 = None
        mul_677: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2116, 0.5);  convert_element_type_2116 = None
        add_475: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_474, mul_677);  add_474 = mul_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1265: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_475, [128, 16384]);  add_475 = None
        mm_332: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1265, permute_911);  permute_911 = None
        permute_912: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1265, [1, 0])
        mm_333: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_912, view_252);  permute_912 = None
        sum_165: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1265, [0], True, dtype = torch.float32);  view_1265 = None
        view_1266: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_165, [16384]);  sum_165 = None
        convert_element_type_2123: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1266, torch.bfloat16);  view_1266 = None
        view_1267: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_332, [1, 128, 4096]);  mm_332 = None
        convert_element_type_2124: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1267, torch.float32);  view_1267 = None
        convert_element_type_2125: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_333, torch.float32);  mm_333 = None
        convert_element_type_2126: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2123, torch.float32);  convert_element_type_2123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_334: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_908, view_274);  permute_908 = view_274 = None
        mm_335: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1262, permute_917);  view_1262 = permute_917 = None
        view_1269: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_335, [1, 128, 4096]);  mm_335 = None
        convert_element_type_2131: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_334, torch.float32);  mm_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1270: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1269, [1, 128, 16, 256]);  view_1269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_919: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1270, [0, 2, 1, 3]);  view_1270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1271: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_919, [16, 128, 256]);  permute_919 = None
        bmm_128: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_920, view_1271);  permute_920 = None
        bmm_129: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1271, permute_921);  view_1271 = permute_921 = None
        view_1272: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_128, [1, 16, 128, 256]);  bmm_128 = None
        view_1273: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_129, [1, 16, 128, 128]);  bmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2136: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1273, torch.float32);  view_1273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_678: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2136, div_19);  convert_element_type_2136 = None
        sum_166: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_678, [-1], True)
        neg_111: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_19);  div_19 = None
        fma_18: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_111, sum_166, mul_678);  neg_111 = sum_166 = mul_678 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2137: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_18, torch.bfloat16);  fma_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_95: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2137, 16.0);  convert_element_type_2137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1274: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_95, [16, 128, 128]);  div_95 = None
        bmm_130: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_922, view_1274);  permute_922 = None
        bmm_131: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1274, permute_923);  view_1274 = permute_923 = None
        view_1275: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_130, [1, 16, 256, 128]);  bmm_130 = None
        view_1276: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_131, [1, 16, 128, 256]);  bmm_131 = None
        convert_element_type_2143: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1275, torch.float32);  view_1275 = None
        permute_924: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2143, [0, 1, 3, 2]);  convert_element_type_2143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2144: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_924, torch.bfloat16);  permute_924 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_9: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1276, torch.bfloat16);  view_1276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_925: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_9, [0, 2, 1, 3]);  convert_element_type_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_926: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2144, [0, 2, 1, 3]);  convert_element_type_2144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_301: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_925, 3, 0, 64)
        slice_302: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_925, 3, 64, 256);  permute_925 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_303: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_926, 3, 0, 64)
        slice_304: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_926, 3, 64, 256);  permute_926 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_73: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_129, [1, 128, 1, 32, 2]);  unsqueeze_129 = None
        clone_73: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_261: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [1, 128, 1, 64]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_679: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_301, view_261)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1277: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_679, [1, 128, 16, 32, 2]);  mul_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_72: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1277, -1, 0)
        select_73: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1277, -1, 1);  view_1277 = None
        neg_112: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_72);  select_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_144: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_112, 3, 1, 9223372036854775807, 2);  neg_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_145: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_73, 3, 0, 9223372036854775807, 2);  select_73 = None
        add_476: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_144, slice_scatter_145);  slice_scatter_144 = slice_scatter_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_74: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_131, [1, 128, 1, 32, 2]);  unsqueeze_131 = None
        clone_74: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_262: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [1, 128, 1, 64]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_680: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_301, view_262);  slice_301 = None
        add_477: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_476, mul_680);  add_476 = mul_680 = None
        mul_681: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_303, view_261);  view_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1278: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_681, [1, 128, 16, 32, 2]);  mul_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_74: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1278, -1, 0)
        select_75: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1278, -1, 1);  view_1278 = None
        neg_113: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_74);  select_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_146: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_113, 3, 1, 9223372036854775807, 2);  neg_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_147: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_75, 3, 0, 9223372036854775807, 2);  select_75 = None
        add_478: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_146, slice_scatter_147);  slice_scatter_146 = slice_scatter_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_682: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_303, view_262);  slice_303 = view_262 = None
        add_479: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_478, mul_682);  add_478 = mul_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_148: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_302, 3, 64, 9223372036854775807);  slice_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_149: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_477, 3, 0, 64);  add_477 = None
        add_480: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_148, slice_scatter_149);  slice_scatter_148 = slice_scatter_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_150: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_304, 3, 64, 9223372036854775807);  slice_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_151: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_479, 3, 0, 64);  add_479 = None
        add_481: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_150, slice_scatter_151);  slice_scatter_150 = slice_scatter_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_927: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1272, [0, 2, 1, 3]);  view_1272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_243: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_927, memory_format = torch.contiguous_format);  permute_927 = None
        view_1279: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_243, [1, 128, 4096]);  clone_243 = None
        view_1280: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_481, [1, 128, 4096]);  add_481 = None
        view_1281: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_480, [1, 128, 4096]);  add_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1282: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1279, [128, 4096]);  view_1279 = None
        permute_928: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1282, [1, 0])
        mm_336: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_928, view_252);  permute_928 = None
        mm_337: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1282, permute_930);  view_1282 = permute_930 = None
        view_1283: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_337, [1, 128, 4096]);  mm_337 = None
        convert_element_type_2150: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1283, torch.float32);  view_1283 = None
        add_482: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2124, convert_element_type_2150);  convert_element_type_2124 = convert_element_type_2150 = None
        convert_element_type_2151: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_336, torch.float32);  mm_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1284: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1280, [128, 4096]);  view_1280 = None
        permute_932: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1284, [1, 0])
        mm_338: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_932, view_252);  permute_932 = None
        mm_339: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1284, permute_934);  view_1284 = permute_934 = None
        view_1285: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_339, [1, 128, 4096]);  mm_339 = None
        convert_element_type_2156: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1285, torch.float32);  view_1285 = None
        add_483: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_482, convert_element_type_2156);  add_482 = convert_element_type_2156 = None
        convert_element_type_2157: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_338, torch.float32);  mm_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1286: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1281, [128, 4096]);  view_1281 = None
        permute_936: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1286, [1, 0])
        mm_340: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_936, view_252);  permute_936 = view_252 = None
        mm_341: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1286, permute_938);  view_1286 = permute_938 = None
        view_1287: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_341, [1, 128, 4096]);  mm_341 = None
        convert_element_type_2162: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1287, torch.float32);  view_1287 = None
        add_484: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_483, convert_element_type_2162);  add_483 = convert_element_type_2162 = None
        convert_element_type_2163: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_340, torch.float32);  mm_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_684: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_484, primals_102);  primals_102 = None
        mul_685: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_684, 4096)
        sum_167: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_684, [2], True)
        mul_686: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_684, mul_90);  mul_684 = None
        sum_168: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_686, [2], True);  mul_686 = None
        mul_687: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, sum_168);  sum_168 = None
        sub_139: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_685, sum_167);  mul_685 = sum_167 = None
        sub_140: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_139, mul_687);  sub_139 = mul_687 = None
        mul_688: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_96, sub_140);  div_96 = sub_140 = None
        mul_689: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_484, mul_90);  mul_90 = None
        sum_169: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_689, [0, 1]);  mul_689 = None
        sum_170: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_484, [0, 1]);  add_484 = None
        add_485: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_473, mul_688);  add_473 = mul_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2164: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_485, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1288: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2164, [128, 4096]);  convert_element_type_2164 = None
        mm_342: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1288, permute_940);  permute_940 = None
        permute_941: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1288, [1, 0])
        mm_343: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_941, view_250);  view_250 = None
        sum_171: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1288, [0], True, dtype = torch.float32)
        view_1289: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_171, [4096]);  sum_171 = None
        convert_element_type_2169: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1289, torch.bfloat16);  view_1289 = None
        view_1290: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_342, [1, 128, 16384]);  mm_342 = None
        convert_element_type_2170: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1290, torch.float32);  view_1290 = None
        convert_element_type_2171: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_343, torch.float32);  mm_343 = None
        convert_element_type_2172: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2169, torch.float32);  convert_element_type_2169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_249: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [1, 128, 16384]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_86: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_249, 0.5)
        mul_690: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2170, mul_86);  mul_86 = None
        convert_element_type_335: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_249, torch.float32)
        pow_9: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_335, 3.0)
        mul_87: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_80: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_249, mul_87);  view_249 = mul_87 = None
        mul_88: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_80, 0.7978845608028654);  add_80 = None
        tanh_8: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_88);  mul_88 = None
        add_81: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_691: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2170, add_81);  convert_element_type_2170 = add_81 = None
        convert_element_type_2173: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_691, torch.bfloat16);  mul_691 = None
        mul_692: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_141: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_692);  mul_692 = None
        mul_693: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_690, sub_141);  mul_690 = sub_141 = None
        mul_694: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_693, 0.7978845608028654);  mul_693 = None
        convert_element_type_2174: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_694, torch.bfloat16)
        mul_695: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_694, 0.044715);  mul_694 = None
        pow_48: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_335, 2.0);  convert_element_type_335 = None
        mul_696: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_48, 3.0);  pow_48 = None
        mul_697: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_695, mul_696);  mul_695 = mul_696 = None
        convert_element_type_2175: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_697, torch.bfloat16);  mul_697 = None
        add_486: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2174, convert_element_type_2175);  convert_element_type_2174 = convert_element_type_2175 = None
        mul_698: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2173, 0.5);  convert_element_type_2173 = None
        add_487: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_486, mul_698);  add_486 = mul_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1291: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_487, [128, 16384]);  add_487 = None
        mm_344: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1291, permute_944);  permute_944 = None
        permute_945: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1291, [1, 0])
        mm_345: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_945, view_224);  permute_945 = None
        sum_172: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1291, [0], True, dtype = torch.float32);  view_1291 = None
        view_1292: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_172, [16384]);  sum_172 = None
        convert_element_type_2180: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1292, torch.bfloat16);  view_1292 = None
        view_1293: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_344, [1, 128, 4096]);  mm_344 = None
        convert_element_type_2181: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1293, torch.float32);  view_1293 = None
        convert_element_type_2182: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_345, torch.float32);  mm_345 = None
        convert_element_type_2183: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2180, torch.float32);  convert_element_type_2180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_346: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_941, view_246);  permute_941 = view_246 = None
        mm_347: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1288, permute_950);  view_1288 = permute_950 = None
        view_1295: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_347, [1, 128, 4096]);  mm_347 = None
        convert_element_type_2188: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_346, torch.float32);  mm_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1296: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1295, [1, 128, 16, 256]);  view_1295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_952: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1296, [0, 2, 1, 3]);  view_1296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1297: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_952, [16, 128, 256]);  permute_952 = None
        bmm_132: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_953, view_1297);  permute_953 = None
        bmm_133: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1297, permute_954);  view_1297 = permute_954 = None
        view_1298: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_132, [1, 16, 128, 256]);  bmm_132 = None
        view_1299: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_133, [1, 16, 128, 128]);  bmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2193: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1299, torch.float32);  view_1299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_699: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2193, div_17);  convert_element_type_2193 = None
        sum_173: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_699, [-1], True)
        neg_114: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_17);  div_17 = None
        fma_19: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_114, sum_173, mul_699);  neg_114 = sum_173 = mul_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2194: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_19, torch.bfloat16);  fma_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_97: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2194, 16.0);  convert_element_type_2194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1300: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_97, [16, 128, 128]);  div_97 = None
        bmm_134: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_955, view_1300);  permute_955 = None
        bmm_135: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1300, permute_956);  view_1300 = permute_956 = None
        view_1301: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_134, [1, 16, 256, 128]);  bmm_134 = None
        view_1302: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_135, [1, 16, 128, 256]);  bmm_135 = None
        convert_element_type_2200: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1301, torch.float32);  view_1301 = None
        permute_957: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2200, [0, 1, 3, 2]);  convert_element_type_2200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2201: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_957, torch.bfloat16);  permute_957 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_8: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1302, torch.bfloat16);  view_1302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_958: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_8, [0, 2, 1, 3]);  convert_element_type_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_959: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2201, [0, 2, 1, 3]);  convert_element_type_2201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_305: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_958, 3, 0, 64)
        slice_306: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_958, 3, 64, 256);  permute_958 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_307: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_959, 3, 0, 64)
        slice_308: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_959, 3, 64, 256);  permute_959 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_65: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_116, [1, 128, 1, 32, 2]);  unsqueeze_116 = None
        clone_65: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_233: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [1, 128, 1, 64]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_700: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_305, view_233)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1303: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_700, [1, 128, 16, 32, 2]);  mul_700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_76: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1303, -1, 0)
        select_77: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1303, -1, 1);  view_1303 = None
        neg_115: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_76);  select_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_152: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_115, 3, 1, 9223372036854775807, 2);  neg_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_153: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_77, 3, 0, 9223372036854775807, 2);  select_77 = None
        add_488: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_152, slice_scatter_153);  slice_scatter_152 = slice_scatter_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_66: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_118, [1, 128, 1, 32, 2]);  unsqueeze_118 = None
        clone_66: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_234: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_66, [1, 128, 1, 64]);  clone_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_701: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_305, view_234);  slice_305 = None
        add_489: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_488, mul_701);  add_488 = mul_701 = None
        mul_702: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_307, view_233);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1304: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_702, [1, 128, 16, 32, 2]);  mul_702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_78: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1304, -1, 0)
        select_79: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1304, -1, 1);  view_1304 = None
        neg_116: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_78);  select_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_154: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_116, 3, 1, 9223372036854775807, 2);  neg_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_155: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_79, 3, 0, 9223372036854775807, 2);  select_79 = None
        add_490: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_154, slice_scatter_155);  slice_scatter_154 = slice_scatter_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_703: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_307, view_234);  slice_307 = view_234 = None
        add_491: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_490, mul_703);  add_490 = mul_703 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_156: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_306, 3, 64, 9223372036854775807);  slice_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_157: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_489, 3, 0, 64);  add_489 = None
        add_492: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_156, slice_scatter_157);  slice_scatter_156 = slice_scatter_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_158: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_308, 3, 64, 9223372036854775807);  slice_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_159: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_491, 3, 0, 64);  add_491 = None
        add_493: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_158, slice_scatter_159);  slice_scatter_158 = slice_scatter_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_960: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1298, [0, 2, 1, 3]);  view_1298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_244: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_960, memory_format = torch.contiguous_format);  permute_960 = None
        view_1305: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_244, [1, 128, 4096]);  clone_244 = None
        view_1306: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_493, [1, 128, 4096]);  add_493 = None
        view_1307: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_492, [1, 128, 4096]);  add_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1308: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1305, [128, 4096]);  view_1305 = None
        permute_961: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1308, [1, 0])
        mm_348: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_961, view_224);  permute_961 = None
        mm_349: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1308, permute_963);  view_1308 = permute_963 = None
        view_1309: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_349, [1, 128, 4096]);  mm_349 = None
        convert_element_type_2207: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1309, torch.float32);  view_1309 = None
        add_494: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2181, convert_element_type_2207);  convert_element_type_2181 = convert_element_type_2207 = None
        convert_element_type_2208: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_348, torch.float32);  mm_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1310: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1306, [128, 4096]);  view_1306 = None
        permute_965: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1310, [1, 0])
        mm_350: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_965, view_224);  permute_965 = None
        mm_351: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1310, permute_967);  view_1310 = permute_967 = None
        view_1311: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_351, [1, 128, 4096]);  mm_351 = None
        convert_element_type_2213: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1311, torch.float32);  view_1311 = None
        add_495: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_494, convert_element_type_2213);  add_494 = convert_element_type_2213 = None
        convert_element_type_2214: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_350, torch.float32);  mm_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1312: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1307, [128, 4096]);  view_1307 = None
        permute_969: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1312, [1, 0])
        mm_352: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_969, view_224);  permute_969 = view_224 = None
        mm_353: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1312, permute_971);  view_1312 = permute_971 = None
        view_1313: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_353, [1, 128, 4096]);  mm_353 = None
        convert_element_type_2219: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1313, torch.float32);  view_1313 = None
        add_496: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_495, convert_element_type_2219);  add_495 = convert_element_type_2219 = None
        convert_element_type_2220: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_352, torch.float32);  mm_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_705: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_496, primals_91);  primals_91 = None
        mul_706: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_705, 4096)
        sum_174: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_705, [2], True)
        mul_707: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_705, mul_80);  mul_705 = None
        sum_175: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_707, [2], True);  mul_707 = None
        mul_708: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, sum_175);  sum_175 = None
        sub_143: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_706, sum_174);  mul_706 = sum_174 = None
        sub_144: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_143, mul_708);  sub_143 = mul_708 = None
        mul_709: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_98, sub_144);  div_98 = sub_144 = None
        mul_710: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_496, mul_80);  mul_80 = None
        sum_176: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_710, [0, 1]);  mul_710 = None
        sum_177: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_496, [0, 1]);  add_496 = None
        add_497: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_485, mul_709);  add_485 = mul_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2221: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_497, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1314: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2221, [128, 4096]);  convert_element_type_2221 = None
        mm_354: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1314, permute_973);  permute_973 = None
        permute_974: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1314, [1, 0])
        mm_355: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_974, view_222);  view_222 = None
        sum_178: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1314, [0], True, dtype = torch.float32)
        view_1315: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_178, [4096]);  sum_178 = None
        convert_element_type_2226: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1315, torch.bfloat16);  view_1315 = None
        view_1316: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_354, [1, 128, 16384]);  mm_354 = None
        convert_element_type_2227: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1316, torch.float32);  view_1316 = None
        convert_element_type_2228: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_355, torch.float32);  mm_355 = None
        convert_element_type_2229: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2226, torch.float32);  convert_element_type_2226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_221: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [1, 128, 16384]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_221, 0.5)
        mul_711: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2227, mul_76);  mul_76 = None
        convert_element_type_297: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_221, torch.float32)
        pow_8: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_297, 3.0)
        mul_77: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_71: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_221, mul_77);  view_221 = mul_77 = None
        mul_78: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_7: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_72: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_712: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2227, add_72);  convert_element_type_2227 = add_72 = None
        convert_element_type_2230: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_712, torch.bfloat16);  mul_712 = None
        mul_713: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_145: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_713);  mul_713 = None
        mul_714: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_711, sub_145);  mul_711 = sub_145 = None
        mul_715: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_714, 0.7978845608028654);  mul_714 = None
        convert_element_type_2231: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_715, torch.bfloat16)
        mul_716: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_715, 0.044715);  mul_715 = None
        pow_49: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_297, 2.0);  convert_element_type_297 = None
        mul_717: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_49, 3.0);  pow_49 = None
        mul_718: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_716, mul_717);  mul_716 = mul_717 = None
        convert_element_type_2232: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_718, torch.bfloat16);  mul_718 = None
        add_498: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2231, convert_element_type_2232);  convert_element_type_2231 = convert_element_type_2232 = None
        mul_719: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2230, 0.5);  convert_element_type_2230 = None
        add_499: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_498, mul_719);  add_498 = mul_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1317: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_499, [128, 16384]);  add_499 = None
        mm_356: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1317, permute_977);  permute_977 = None
        permute_978: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1317, [1, 0])
        mm_357: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_978, view_196);  permute_978 = None
        sum_179: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1317, [0], True, dtype = torch.float32);  view_1317 = None
        view_1318: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_179, [16384]);  sum_179 = None
        convert_element_type_2237: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1318, torch.bfloat16);  view_1318 = None
        view_1319: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_356, [1, 128, 4096]);  mm_356 = None
        convert_element_type_2238: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1319, torch.float32);  view_1319 = None
        convert_element_type_2239: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_357, torch.float32);  mm_357 = None
        convert_element_type_2240: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2237, torch.float32);  convert_element_type_2237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_358: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_974, view_218);  permute_974 = view_218 = None
        mm_359: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1314, permute_983);  view_1314 = permute_983 = None
        view_1321: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_359, [1, 128, 4096]);  mm_359 = None
        convert_element_type_2245: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_358, torch.float32);  mm_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1322: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1321, [1, 128, 16, 256]);  view_1321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_985: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1322, [0, 2, 1, 3]);  view_1322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1323: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_985, [16, 128, 256]);  permute_985 = None
        bmm_136: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_986, view_1323);  permute_986 = None
        bmm_137: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1323, permute_987);  view_1323 = permute_987 = None
        view_1324: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_136, [1, 16, 128, 256]);  bmm_136 = None
        view_1325: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_137, [1, 16, 128, 128]);  bmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2250: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1325, torch.float32);  view_1325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_720: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2250, div_15);  convert_element_type_2250 = None
        sum_180: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_720, [-1], True)
        neg_117: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_15);  div_15 = None
        fma_20: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_117, sum_180, mul_720);  neg_117 = sum_180 = mul_720 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2251: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_20, torch.bfloat16);  fma_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_99: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2251, 16.0);  convert_element_type_2251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1326: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_99, [16, 128, 128]);  div_99 = None
        bmm_138: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_988, view_1326);  permute_988 = None
        bmm_139: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1326, permute_989);  view_1326 = permute_989 = None
        view_1327: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_138, [1, 16, 256, 128]);  bmm_138 = None
        view_1328: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_139, [1, 16, 128, 256]);  bmm_139 = None
        convert_element_type_2257: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1327, torch.float32);  view_1327 = None
        permute_990: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2257, [0, 1, 3, 2]);  convert_element_type_2257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2258: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_990, torch.bfloat16);  permute_990 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_7: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1328, torch.bfloat16);  view_1328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_991: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_7, [0, 2, 1, 3]);  convert_element_type_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_992: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2258, [0, 2, 1, 3]);  convert_element_type_2258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_309: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_991, 3, 0, 64)
        slice_310: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_991, 3, 64, 256);  permute_991 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_311: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_992, 3, 0, 64)
        slice_312: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_992, 3, 64, 256);  permute_992 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_57: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_103, [1, 128, 1, 32, 2]);  unsqueeze_103 = None
        clone_57: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_205: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [1, 128, 1, 64]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_721: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_309, view_205)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1329: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_721, [1, 128, 16, 32, 2]);  mul_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_80: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1329, -1, 0)
        select_81: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1329, -1, 1);  view_1329 = None
        neg_118: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_80);  select_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_160: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_118, 3, 1, 9223372036854775807, 2);  neg_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_161: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_81, 3, 0, 9223372036854775807, 2);  select_81 = None
        add_500: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_160, slice_scatter_161);  slice_scatter_160 = slice_scatter_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_58: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_105, [1, 128, 1, 32, 2]);  unsqueeze_105 = None
        clone_58: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_206: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [1, 128, 1, 64]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_722: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_309, view_206);  slice_309 = None
        add_501: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_500, mul_722);  add_500 = mul_722 = None
        mul_723: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_311, view_205);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1330: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_723, [1, 128, 16, 32, 2]);  mul_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_82: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1330, -1, 0)
        select_83: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1330, -1, 1);  view_1330 = None
        neg_119: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_82);  select_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_162: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_119, 3, 1, 9223372036854775807, 2);  neg_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_163: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_83, 3, 0, 9223372036854775807, 2);  select_83 = None
        add_502: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_162, slice_scatter_163);  slice_scatter_162 = slice_scatter_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_724: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_311, view_206);  slice_311 = view_206 = None
        add_503: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_502, mul_724);  add_502 = mul_724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_164: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_310, 3, 64, 9223372036854775807);  slice_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_165: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_501, 3, 0, 64);  add_501 = None
        add_504: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_164, slice_scatter_165);  slice_scatter_164 = slice_scatter_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_166: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_312, 3, 64, 9223372036854775807);  slice_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_167: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_503, 3, 0, 64);  add_503 = None
        add_505: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_166, slice_scatter_167);  slice_scatter_166 = slice_scatter_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_993: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1324, [0, 2, 1, 3]);  view_1324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_245: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_993, memory_format = torch.contiguous_format);  permute_993 = None
        view_1331: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_245, [1, 128, 4096]);  clone_245 = None
        view_1332: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_505, [1, 128, 4096]);  add_505 = None
        view_1333: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_504, [1, 128, 4096]);  add_504 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1334: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1331, [128, 4096]);  view_1331 = None
        permute_994: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1334, [1, 0])
        mm_360: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_994, view_196);  permute_994 = None
        mm_361: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1334, permute_996);  view_1334 = permute_996 = None
        view_1335: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_361, [1, 128, 4096]);  mm_361 = None
        convert_element_type_2264: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1335, torch.float32);  view_1335 = None
        add_506: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2238, convert_element_type_2264);  convert_element_type_2238 = convert_element_type_2264 = None
        convert_element_type_2265: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_360, torch.float32);  mm_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1336: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1332, [128, 4096]);  view_1332 = None
        permute_998: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1336, [1, 0])
        mm_362: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_998, view_196);  permute_998 = None
        mm_363: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1336, permute_1000);  view_1336 = permute_1000 = None
        view_1337: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_363, [1, 128, 4096]);  mm_363 = None
        convert_element_type_2270: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1337, torch.float32);  view_1337 = None
        add_507: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_506, convert_element_type_2270);  add_506 = convert_element_type_2270 = None
        convert_element_type_2271: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_362, torch.float32);  mm_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1338: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1333, [128, 4096]);  view_1333 = None
        permute_1002: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1338, [1, 0])
        mm_364: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1002, view_196);  permute_1002 = view_196 = None
        mm_365: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1338, permute_1004);  view_1338 = permute_1004 = None
        view_1339: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_365, [1, 128, 4096]);  mm_365 = None
        convert_element_type_2276: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1339, torch.float32);  view_1339 = None
        add_508: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_507, convert_element_type_2276);  add_507 = convert_element_type_2276 = None
        convert_element_type_2277: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_364, torch.float32);  mm_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_726: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_508, primals_80);  primals_80 = None
        mul_727: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_726, 4096)
        sum_181: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_726, [2], True)
        mul_728: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_726, mul_70);  mul_726 = None
        sum_182: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_728, [2], True);  mul_728 = None
        mul_729: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, sum_182);  sum_182 = None
        sub_147: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_727, sum_181);  mul_727 = sum_181 = None
        sub_148: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_147, mul_729);  sub_147 = mul_729 = None
        mul_730: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_100, sub_148);  div_100 = sub_148 = None
        mul_731: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_508, mul_70);  mul_70 = None
        sum_183: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_731, [0, 1]);  mul_731 = None
        sum_184: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_508, [0, 1]);  add_508 = None
        add_509: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_497, mul_730);  add_497 = mul_730 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2278: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_509, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1340: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2278, [128, 4096]);  convert_element_type_2278 = None
        mm_366: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1340, permute_1006);  permute_1006 = None
        permute_1007: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1340, [1, 0])
        mm_367: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_1007, view_194);  view_194 = None
        sum_185: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1340, [0], True, dtype = torch.float32)
        view_1341: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_185, [4096]);  sum_185 = None
        convert_element_type_2283: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1341, torch.bfloat16);  view_1341 = None
        view_1342: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_366, [1, 128, 16384]);  mm_366 = None
        convert_element_type_2284: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1342, torch.float32);  view_1342 = None
        convert_element_type_2285: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_367, torch.float32);  mm_367 = None
        convert_element_type_2286: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2283, torch.float32);  convert_element_type_2283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_193: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [1, 128, 16384]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_66: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_193, 0.5)
        mul_732: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2284, mul_66);  mul_66 = None
        convert_element_type_259: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_193, torch.float32)
        pow_7: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_259, 3.0)
        mul_67: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_62: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_193, mul_67);  view_193 = mul_67 = None
        mul_68: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
        tanh_6: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_68);  mul_68 = None
        add_63: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_733: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2284, add_63);  convert_element_type_2284 = add_63 = None
        convert_element_type_2287: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_733, torch.bfloat16);  mul_733 = None
        mul_734: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_149: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_734);  mul_734 = None
        mul_735: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_732, sub_149);  mul_732 = sub_149 = None
        mul_736: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_735, 0.7978845608028654);  mul_735 = None
        convert_element_type_2288: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_736, torch.bfloat16)
        mul_737: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_736, 0.044715);  mul_736 = None
        pow_50: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_259, 2.0);  convert_element_type_259 = None
        mul_738: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_50, 3.0);  pow_50 = None
        mul_739: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_737, mul_738);  mul_737 = mul_738 = None
        convert_element_type_2289: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_739, torch.bfloat16);  mul_739 = None
        add_510: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2288, convert_element_type_2289);  convert_element_type_2288 = convert_element_type_2289 = None
        mul_740: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2287, 0.5);  convert_element_type_2287 = None
        add_511: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_510, mul_740);  add_510 = mul_740 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1343: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_511, [128, 16384]);  add_511 = None
        mm_368: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1343, permute_1010);  permute_1010 = None
        permute_1011: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1343, [1, 0])
        mm_369: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1011, view_168);  permute_1011 = None
        sum_186: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1343, [0], True, dtype = torch.float32);  view_1343 = None
        view_1344: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_186, [16384]);  sum_186 = None
        convert_element_type_2294: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1344, torch.bfloat16);  view_1344 = None
        view_1345: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_368, [1, 128, 4096]);  mm_368 = None
        convert_element_type_2295: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1345, torch.float32);  view_1345 = None
        convert_element_type_2296: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_369, torch.float32);  mm_369 = None
        convert_element_type_2297: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2294, torch.float32);  convert_element_type_2294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_370: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1007, view_190);  permute_1007 = view_190 = None
        mm_371: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1340, permute_1016);  view_1340 = permute_1016 = None
        view_1347: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_371, [1, 128, 4096]);  mm_371 = None
        convert_element_type_2302: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_370, torch.float32);  mm_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1348: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1347, [1, 128, 16, 256]);  view_1347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1018: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1348, [0, 2, 1, 3]);  view_1348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1349: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1018, [16, 128, 256]);  permute_1018 = None
        bmm_140: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1019, view_1349);  permute_1019 = None
        bmm_141: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1349, permute_1020);  view_1349 = permute_1020 = None
        view_1350: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_140, [1, 16, 128, 256]);  bmm_140 = None
        view_1351: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_141, [1, 16, 128, 128]);  bmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2307: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1351, torch.float32);  view_1351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_741: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2307, div_13);  convert_element_type_2307 = None
        sum_187: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_741, [-1], True)
        neg_120: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_21: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_120, sum_187, mul_741);  neg_120 = sum_187 = mul_741 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2308: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_21, torch.bfloat16);  fma_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_101: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2308, 16.0);  convert_element_type_2308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1352: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_101, [16, 128, 128]);  div_101 = None
        bmm_142: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1021, view_1352);  permute_1021 = None
        bmm_143: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1352, permute_1022);  view_1352 = permute_1022 = None
        view_1353: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_142, [1, 16, 256, 128]);  bmm_142 = None
        view_1354: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_143, [1, 16, 128, 256]);  bmm_143 = None
        convert_element_type_2314: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1353, torch.float32);  view_1353 = None
        permute_1023: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2314, [0, 1, 3, 2]);  convert_element_type_2314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2315: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1023, torch.bfloat16);  permute_1023 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_6: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1354, torch.bfloat16);  view_1354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1024: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_6, [0, 2, 1, 3]);  convert_element_type_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1025: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2315, [0, 2, 1, 3]);  convert_element_type_2315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_313: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1024, 3, 0, 64)
        slice_314: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1024, 3, 64, 256);  permute_1024 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_315: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1025, 3, 0, 64)
        slice_316: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1025, 3, 64, 256);  permute_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_49: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_90, [1, 128, 1, 32, 2]);  unsqueeze_90 = None
        clone_49: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_177: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [1, 128, 1, 64]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_742: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_313, view_177)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1355: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_742, [1, 128, 16, 32, 2]);  mul_742 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_84: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1355, -1, 0)
        select_85: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1355, -1, 1);  view_1355 = None
        neg_121: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_84);  select_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_168: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_121, 3, 1, 9223372036854775807, 2);  neg_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_169: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_85, 3, 0, 9223372036854775807, 2);  select_85 = None
        add_512: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_168, slice_scatter_169);  slice_scatter_168 = slice_scatter_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_50: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_92, [1, 128, 1, 32, 2]);  unsqueeze_92 = None
        clone_50: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_178: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [1, 128, 1, 64]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_743: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_313, view_178);  slice_313 = None
        add_513: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_512, mul_743);  add_512 = mul_743 = None
        mul_744: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_315, view_177);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1356: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_744, [1, 128, 16, 32, 2]);  mul_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_86: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1356, -1, 0)
        select_87: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1356, -1, 1);  view_1356 = None
        neg_122: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_86);  select_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_170: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_122, 3, 1, 9223372036854775807, 2);  neg_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_171: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_87, 3, 0, 9223372036854775807, 2);  select_87 = None
        add_514: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_170, slice_scatter_171);  slice_scatter_170 = slice_scatter_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_745: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_315, view_178);  slice_315 = view_178 = None
        add_515: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_514, mul_745);  add_514 = mul_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_172: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_314, 3, 64, 9223372036854775807);  slice_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_173: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_513, 3, 0, 64);  add_513 = None
        add_516: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_172, slice_scatter_173);  slice_scatter_172 = slice_scatter_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_174: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_316, 3, 64, 9223372036854775807);  slice_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_175: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_515, 3, 0, 64);  add_515 = None
        add_517: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_174, slice_scatter_175);  slice_scatter_174 = slice_scatter_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1026: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1350, [0, 2, 1, 3]);  view_1350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_246: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_1026, memory_format = torch.contiguous_format);  permute_1026 = None
        view_1357: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_246, [1, 128, 4096]);  clone_246 = None
        view_1358: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_517, [1, 128, 4096]);  add_517 = None
        view_1359: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_516, [1, 128, 4096]);  add_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1360: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1357, [128, 4096]);  view_1357 = None
        permute_1027: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1360, [1, 0])
        mm_372: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1027, view_168);  permute_1027 = None
        mm_373: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1360, permute_1029);  view_1360 = permute_1029 = None
        view_1361: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_373, [1, 128, 4096]);  mm_373 = None
        convert_element_type_2321: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1361, torch.float32);  view_1361 = None
        add_518: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2295, convert_element_type_2321);  convert_element_type_2295 = convert_element_type_2321 = None
        convert_element_type_2322: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_372, torch.float32);  mm_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1362: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1358, [128, 4096]);  view_1358 = None
        permute_1031: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1362, [1, 0])
        mm_374: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1031, view_168);  permute_1031 = None
        mm_375: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1362, permute_1033);  view_1362 = permute_1033 = None
        view_1363: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_375, [1, 128, 4096]);  mm_375 = None
        convert_element_type_2327: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1363, torch.float32);  view_1363 = None
        add_519: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_518, convert_element_type_2327);  add_518 = convert_element_type_2327 = None
        convert_element_type_2328: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_374, torch.float32);  mm_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1364: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1359, [128, 4096]);  view_1359 = None
        permute_1035: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1364, [1, 0])
        mm_376: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1035, view_168);  permute_1035 = view_168 = None
        mm_377: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1364, permute_1037);  view_1364 = permute_1037 = None
        view_1365: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_377, [1, 128, 4096]);  mm_377 = None
        convert_element_type_2333: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1365, torch.float32);  view_1365 = None
        add_520: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_519, convert_element_type_2333);  add_519 = convert_element_type_2333 = None
        convert_element_type_2334: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_376, torch.float32);  mm_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_747: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_520, primals_69);  primals_69 = None
        mul_748: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_747, 4096)
        sum_188: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_747, [2], True)
        mul_749: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_747, mul_60);  mul_747 = None
        sum_189: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_749, [2], True);  mul_749 = None
        mul_750: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, sum_189);  sum_189 = None
        sub_151: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_748, sum_188);  mul_748 = sum_188 = None
        sub_152: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_151, mul_750);  sub_151 = mul_750 = None
        mul_751: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_102, sub_152);  div_102 = sub_152 = None
        mul_752: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_520, mul_60);  mul_60 = None
        sum_190: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 1]);  mul_752 = None
        sum_191: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_520, [0, 1]);  add_520 = None
        add_521: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_509, mul_751);  add_509 = mul_751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2335: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_521, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1366: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2335, [128, 4096]);  convert_element_type_2335 = None
        mm_378: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1366, permute_1039);  permute_1039 = None
        permute_1040: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1366, [1, 0])
        mm_379: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_1040, view_166);  view_166 = None
        sum_192: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1366, [0], True, dtype = torch.float32)
        view_1367: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_192, [4096]);  sum_192 = None
        convert_element_type_2340: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1367, torch.bfloat16);  view_1367 = None
        view_1368: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_378, [1, 128, 16384]);  mm_378 = None
        convert_element_type_2341: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1368, torch.float32);  view_1368 = None
        convert_element_type_2342: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_379, torch.float32);  mm_379 = None
        convert_element_type_2343: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2340, torch.float32);  convert_element_type_2340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_165: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [1, 128, 16384]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_165, 0.5)
        mul_753: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2341, mul_56);  mul_56 = None
        convert_element_type_221: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_165, torch.float32)
        pow_6: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_221, 3.0)
        mul_57: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_53: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_165, mul_57);  view_165 = mul_57 = None
        mul_58: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_53, 0.7978845608028654);  add_53 = None
        tanh_5: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_54: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_754: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2341, add_54);  convert_element_type_2341 = add_54 = None
        convert_element_type_2344: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_754, torch.bfloat16);  mul_754 = None
        mul_755: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_153: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_755);  mul_755 = None
        mul_756: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_753, sub_153);  mul_753 = sub_153 = None
        mul_757: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_756, 0.7978845608028654);  mul_756 = None
        convert_element_type_2345: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_757, torch.bfloat16)
        mul_758: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_757, 0.044715);  mul_757 = None
        pow_51: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_221, 2.0);  convert_element_type_221 = None
        mul_759: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_51, 3.0);  pow_51 = None
        mul_760: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_758, mul_759);  mul_758 = mul_759 = None
        convert_element_type_2346: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_760, torch.bfloat16);  mul_760 = None
        add_522: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2345, convert_element_type_2346);  convert_element_type_2345 = convert_element_type_2346 = None
        mul_761: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2344, 0.5);  convert_element_type_2344 = None
        add_523: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_522, mul_761);  add_522 = mul_761 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1369: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_523, [128, 16384]);  add_523 = None
        mm_380: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1369, permute_1043);  permute_1043 = None
        permute_1044: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1369, [1, 0])
        mm_381: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1044, view_140);  permute_1044 = None
        sum_193: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1369, [0], True, dtype = torch.float32);  view_1369 = None
        view_1370: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_193, [16384]);  sum_193 = None
        convert_element_type_2351: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1370, torch.bfloat16);  view_1370 = None
        view_1371: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_380, [1, 128, 4096]);  mm_380 = None
        convert_element_type_2352: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1371, torch.float32);  view_1371 = None
        convert_element_type_2353: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_381, torch.float32);  mm_381 = None
        convert_element_type_2354: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2351, torch.float32);  convert_element_type_2351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_382: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1040, view_162);  permute_1040 = view_162 = None
        mm_383: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1366, permute_1049);  view_1366 = permute_1049 = None
        view_1373: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_383, [1, 128, 4096]);  mm_383 = None
        convert_element_type_2359: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_382, torch.float32);  mm_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1374: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1373, [1, 128, 16, 256]);  view_1373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1051: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1374, [0, 2, 1, 3]);  view_1374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1375: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1051, [16, 128, 256]);  permute_1051 = None
        bmm_144: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1052, view_1375);  permute_1052 = None
        bmm_145: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1375, permute_1053);  view_1375 = permute_1053 = None
        view_1376: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_144, [1, 16, 128, 256]);  bmm_144 = None
        view_1377: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_145, [1, 16, 128, 128]);  bmm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2364: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1377, torch.float32);  view_1377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_762: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2364, div_11);  convert_element_type_2364 = None
        sum_194: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_762, [-1], True)
        neg_123: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_11);  div_11 = None
        fma_22: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_123, sum_194, mul_762);  neg_123 = sum_194 = mul_762 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2365: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_22, torch.bfloat16);  fma_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_103: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2365, 16.0);  convert_element_type_2365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1378: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_103, [16, 128, 128]);  div_103 = None
        bmm_146: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1054, view_1378);  permute_1054 = None
        bmm_147: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1378, permute_1055);  view_1378 = permute_1055 = None
        view_1379: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_146, [1, 16, 256, 128]);  bmm_146 = None
        view_1380: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_147, [1, 16, 128, 256]);  bmm_147 = None
        convert_element_type_2371: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1379, torch.float32);  view_1379 = None
        permute_1056: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2371, [0, 1, 3, 2]);  convert_element_type_2371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2372: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1056, torch.bfloat16);  permute_1056 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_5: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1380, torch.bfloat16);  view_1380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1057: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_5, [0, 2, 1, 3]);  convert_element_type_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1058: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2372, [0, 2, 1, 3]);  convert_element_type_2372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_317: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1057, 3, 0, 64)
        slice_318: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1057, 3, 64, 256);  permute_1057 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_319: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1058, 3, 0, 64)
        slice_320: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1058, 3, 64, 256);  permute_1058 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_41: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_77, [1, 128, 1, 32, 2]);  unsqueeze_77 = None
        clone_41: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_149: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [1, 128, 1, 64]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_763: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_317, view_149)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1381: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_763, [1, 128, 16, 32, 2]);  mul_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_88: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1381, -1, 0)
        select_89: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1381, -1, 1);  view_1381 = None
        neg_124: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_88);  select_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_176: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_124, 3, 1, 9223372036854775807, 2);  neg_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_177: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_89, 3, 0, 9223372036854775807, 2);  select_89 = None
        add_524: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_176, slice_scatter_177);  slice_scatter_176 = slice_scatter_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_42: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_79, [1, 128, 1, 32, 2]);  unsqueeze_79 = None
        clone_42: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_150: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [1, 128, 1, 64]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_764: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_317, view_150);  slice_317 = None
        add_525: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_524, mul_764);  add_524 = mul_764 = None
        mul_765: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_319, view_149);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1382: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_765, [1, 128, 16, 32, 2]);  mul_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_90: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1382, -1, 0)
        select_91: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1382, -1, 1);  view_1382 = None
        neg_125: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_90);  select_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_178: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_125, 3, 1, 9223372036854775807, 2);  neg_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_179: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_91, 3, 0, 9223372036854775807, 2);  select_91 = None
        add_526: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_178, slice_scatter_179);  slice_scatter_178 = slice_scatter_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_766: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_319, view_150);  slice_319 = view_150 = None
        add_527: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_526, mul_766);  add_526 = mul_766 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_180: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_318, 3, 64, 9223372036854775807);  slice_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_181: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_525, 3, 0, 64);  add_525 = None
        add_528: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_180, slice_scatter_181);  slice_scatter_180 = slice_scatter_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_182: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_320, 3, 64, 9223372036854775807);  slice_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_183: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_527, 3, 0, 64);  add_527 = None
        add_529: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_182, slice_scatter_183);  slice_scatter_182 = slice_scatter_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1059: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1376, [0, 2, 1, 3]);  view_1376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_247: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_1059, memory_format = torch.contiguous_format);  permute_1059 = None
        view_1383: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_247, [1, 128, 4096]);  clone_247 = None
        view_1384: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_529, [1, 128, 4096]);  add_529 = None
        view_1385: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_528, [1, 128, 4096]);  add_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1386: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1383, [128, 4096]);  view_1383 = None
        permute_1060: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1386, [1, 0])
        mm_384: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1060, view_140);  permute_1060 = None
        mm_385: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1386, permute_1062);  view_1386 = permute_1062 = None
        view_1387: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_385, [1, 128, 4096]);  mm_385 = None
        convert_element_type_2378: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1387, torch.float32);  view_1387 = None
        add_530: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2352, convert_element_type_2378);  convert_element_type_2352 = convert_element_type_2378 = None
        convert_element_type_2379: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_384, torch.float32);  mm_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1388: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1384, [128, 4096]);  view_1384 = None
        permute_1064: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1388, [1, 0])
        mm_386: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1064, view_140);  permute_1064 = None
        mm_387: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1388, permute_1066);  view_1388 = permute_1066 = None
        view_1389: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_387, [1, 128, 4096]);  mm_387 = None
        convert_element_type_2384: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1389, torch.float32);  view_1389 = None
        add_531: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_530, convert_element_type_2384);  add_530 = convert_element_type_2384 = None
        convert_element_type_2385: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_386, torch.float32);  mm_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1390: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1385, [128, 4096]);  view_1385 = None
        permute_1068: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1390, [1, 0])
        mm_388: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1068, view_140);  permute_1068 = view_140 = None
        mm_389: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1390, permute_1070);  view_1390 = permute_1070 = None
        view_1391: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_389, [1, 128, 4096]);  mm_389 = None
        convert_element_type_2390: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1391, torch.float32);  view_1391 = None
        add_532: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_531, convert_element_type_2390);  add_531 = convert_element_type_2390 = None
        convert_element_type_2391: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_388, torch.float32);  mm_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_768: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_532, primals_58);  primals_58 = None
        mul_769: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_768, 4096)
        sum_195: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_768, [2], True)
        mul_770: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_768, mul_50);  mul_768 = None
        sum_196: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_770, [2], True);  mul_770 = None
        mul_771: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, sum_196);  sum_196 = None
        sub_155: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_769, sum_195);  mul_769 = sum_195 = None
        sub_156: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_155, mul_771);  sub_155 = mul_771 = None
        mul_772: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_104, sub_156);  div_104 = sub_156 = None
        mul_773: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_532, mul_50);  mul_50 = None
        sum_197: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_773, [0, 1]);  mul_773 = None
        sum_198: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_532, [0, 1]);  add_532 = None
        add_533: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_521, mul_772);  add_521 = mul_772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2392: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_533, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1392: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2392, [128, 4096]);  convert_element_type_2392 = None
        mm_390: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1392, permute_1072);  permute_1072 = None
        permute_1073: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1392, [1, 0])
        mm_391: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_1073, view_138);  view_138 = None
        sum_199: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1392, [0], True, dtype = torch.float32)
        view_1393: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_199, [4096]);  sum_199 = None
        convert_element_type_2397: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1393, torch.bfloat16);  view_1393 = None
        view_1394: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_390, [1, 128, 16384]);  mm_390 = None
        convert_element_type_2398: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1394, torch.float32);  view_1394 = None
        convert_element_type_2399: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_391, torch.float32);  mm_391 = None
        convert_element_type_2400: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2397, torch.float32);  convert_element_type_2397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_137: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 16384]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_46: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_137, 0.5)
        mul_774: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2398, mul_46);  mul_46 = None
        convert_element_type_183: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_137, torch.float32)
        pow_5: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_183, 3.0)
        mul_47: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_44: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_137, mul_47);  view_137 = mul_47 = None
        mul_48: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_44, 0.7978845608028654);  add_44 = None
        tanh_4: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
        add_45: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_775: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2398, add_45);  convert_element_type_2398 = add_45 = None
        convert_element_type_2401: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_775, torch.bfloat16);  mul_775 = None
        mul_776: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_157: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_776);  mul_776 = None
        mul_777: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_774, sub_157);  mul_774 = sub_157 = None
        mul_778: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_777, 0.7978845608028654);  mul_777 = None
        convert_element_type_2402: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_778, torch.bfloat16)
        mul_779: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_778, 0.044715);  mul_778 = None
        pow_52: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_183, 2.0);  convert_element_type_183 = None
        mul_780: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_52, 3.0);  pow_52 = None
        mul_781: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_779, mul_780);  mul_779 = mul_780 = None
        convert_element_type_2403: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_781, torch.bfloat16);  mul_781 = None
        add_534: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2402, convert_element_type_2403);  convert_element_type_2402 = convert_element_type_2403 = None
        mul_782: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2401, 0.5);  convert_element_type_2401 = None
        add_535: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_534, mul_782);  add_534 = mul_782 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1395: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_535, [128, 16384]);  add_535 = None
        mm_392: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1395, permute_1076);  permute_1076 = None
        permute_1077: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1395, [1, 0])
        mm_393: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1077, view_112);  permute_1077 = None
        sum_200: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1395, [0], True, dtype = torch.float32);  view_1395 = None
        view_1396: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_200, [16384]);  sum_200 = None
        convert_element_type_2408: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1396, torch.bfloat16);  view_1396 = None
        view_1397: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_392, [1, 128, 4096]);  mm_392 = None
        convert_element_type_2409: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1397, torch.float32);  view_1397 = None
        convert_element_type_2410: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_393, torch.float32);  mm_393 = None
        convert_element_type_2411: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2408, torch.float32);  convert_element_type_2408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_394: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1073, view_134);  permute_1073 = view_134 = None
        mm_395: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1392, permute_1082);  view_1392 = permute_1082 = None
        view_1399: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_395, [1, 128, 4096]);  mm_395 = None
        convert_element_type_2416: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_394, torch.float32);  mm_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1400: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1399, [1, 128, 16, 256]);  view_1399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1084: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1400, [0, 2, 1, 3]);  view_1400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1401: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1084, [16, 128, 256]);  permute_1084 = None
        bmm_148: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1085, view_1401);  permute_1085 = None
        bmm_149: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1401, permute_1086);  view_1401 = permute_1086 = None
        view_1402: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_148, [1, 16, 128, 256]);  bmm_148 = None
        view_1403: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_149, [1, 16, 128, 128]);  bmm_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2421: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1403, torch.float32);  view_1403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_783: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2421, div_9);  convert_element_type_2421 = None
        sum_201: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_783, [-1], True)
        neg_126: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_9);  div_9 = None
        fma_23: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_126, sum_201, mul_783);  neg_126 = sum_201 = mul_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2422: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_23, torch.bfloat16);  fma_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_105: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2422, 16.0);  convert_element_type_2422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1404: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_105, [16, 128, 128]);  div_105 = None
        bmm_150: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1087, view_1404);  permute_1087 = None
        bmm_151: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1404, permute_1088);  view_1404 = permute_1088 = None
        view_1405: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_150, [1, 16, 256, 128]);  bmm_150 = None
        view_1406: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_151, [1, 16, 128, 256]);  bmm_151 = None
        convert_element_type_2428: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1405, torch.float32);  view_1405 = None
        permute_1089: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2428, [0, 1, 3, 2]);  convert_element_type_2428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2429: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1089, torch.bfloat16);  permute_1089 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_4: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1406, torch.bfloat16);  view_1406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1090: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_4, [0, 2, 1, 3]);  convert_element_type_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1091: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2429, [0, 2, 1, 3]);  convert_element_type_2429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_321: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1090, 3, 0, 64)
        slice_322: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1090, 3, 64, 256);  permute_1090 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_323: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1091, 3, 0, 64)
        slice_324: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1091, 3, 64, 256);  permute_1091 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_33: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_64, [1, 128, 1, 32, 2]);  unsqueeze_64 = None
        clone_33: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_121: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [1, 128, 1, 64]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_784: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_321, view_121)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1407: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_784, [1, 128, 16, 32, 2]);  mul_784 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_92: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1407, -1, 0)
        select_93: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1407, -1, 1);  view_1407 = None
        neg_127: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_92);  select_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_184: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_127, 3, 1, 9223372036854775807, 2);  neg_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_185: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_93, 3, 0, 9223372036854775807, 2);  select_93 = None
        add_536: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_184, slice_scatter_185);  slice_scatter_184 = slice_scatter_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_34: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_66, [1, 128, 1, 32, 2]);  unsqueeze_66 = None
        clone_34: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_122: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [1, 128, 1, 64]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_785: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_321, view_122);  slice_321 = None
        add_537: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_536, mul_785);  add_536 = mul_785 = None
        mul_786: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_323, view_121);  view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1408: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_786, [1, 128, 16, 32, 2]);  mul_786 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_94: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1408, -1, 0)
        select_95: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1408, -1, 1);  view_1408 = None
        neg_128: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_94);  select_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_186: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_128, 3, 1, 9223372036854775807, 2);  neg_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_187: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_95, 3, 0, 9223372036854775807, 2);  select_95 = None
        add_538: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_186, slice_scatter_187);  slice_scatter_186 = slice_scatter_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_787: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_323, view_122);  slice_323 = view_122 = None
        add_539: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_538, mul_787);  add_538 = mul_787 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_188: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_322, 3, 64, 9223372036854775807);  slice_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_189: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_537, 3, 0, 64);  add_537 = None
        add_540: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_188, slice_scatter_189);  slice_scatter_188 = slice_scatter_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_190: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_324, 3, 64, 9223372036854775807);  slice_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_191: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_539, 3, 0, 64);  add_539 = None
        add_541: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_190, slice_scatter_191);  slice_scatter_190 = slice_scatter_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1092: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1402, [0, 2, 1, 3]);  view_1402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_248: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_1092, memory_format = torch.contiguous_format);  permute_1092 = None
        view_1409: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_248, [1, 128, 4096]);  clone_248 = None
        view_1410: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_541, [1, 128, 4096]);  add_541 = None
        view_1411: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_540, [1, 128, 4096]);  add_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1412: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1409, [128, 4096]);  view_1409 = None
        permute_1093: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1412, [1, 0])
        mm_396: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1093, view_112);  permute_1093 = None
        mm_397: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1412, permute_1095);  view_1412 = permute_1095 = None
        view_1413: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_397, [1, 128, 4096]);  mm_397 = None
        convert_element_type_2435: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1413, torch.float32);  view_1413 = None
        add_542: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2409, convert_element_type_2435);  convert_element_type_2409 = convert_element_type_2435 = None
        convert_element_type_2436: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_396, torch.float32);  mm_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1414: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1410, [128, 4096]);  view_1410 = None
        permute_1097: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1414, [1, 0])
        mm_398: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1097, view_112);  permute_1097 = None
        mm_399: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1414, permute_1099);  view_1414 = permute_1099 = None
        view_1415: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_399, [1, 128, 4096]);  mm_399 = None
        convert_element_type_2441: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1415, torch.float32);  view_1415 = None
        add_543: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_542, convert_element_type_2441);  add_542 = convert_element_type_2441 = None
        convert_element_type_2442: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_398, torch.float32);  mm_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1416: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1411, [128, 4096]);  view_1411 = None
        permute_1101: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1416, [1, 0])
        mm_400: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1101, view_112);  permute_1101 = view_112 = None
        mm_401: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1416, permute_1103);  view_1416 = permute_1103 = None
        view_1417: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_401, [1, 128, 4096]);  mm_401 = None
        convert_element_type_2447: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1417, torch.float32);  view_1417 = None
        add_544: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_543, convert_element_type_2447);  add_543 = convert_element_type_2447 = None
        convert_element_type_2448: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_400, torch.float32);  mm_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_789: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_544, primals_47);  primals_47 = None
        mul_790: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_789, 4096)
        sum_202: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_789, [2], True)
        mul_791: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_789, mul_40);  mul_789 = None
        sum_203: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_791, [2], True);  mul_791 = None
        mul_792: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, sum_203);  sum_203 = None
        sub_159: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_790, sum_202);  mul_790 = sum_202 = None
        sub_160: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_159, mul_792);  sub_159 = mul_792 = None
        mul_793: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_106, sub_160);  div_106 = sub_160 = None
        mul_794: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_544, mul_40);  mul_40 = None
        sum_204: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_794, [0, 1]);  mul_794 = None
        sum_205: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_544, [0, 1]);  add_544 = None
        add_545: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_533, mul_793);  add_533 = mul_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2449: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_545, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1418: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2449, [128, 4096]);  convert_element_type_2449 = None
        mm_402: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1418, permute_1105);  permute_1105 = None
        permute_1106: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1418, [1, 0])
        mm_403: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_1106, view_110);  view_110 = None
        sum_206: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1418, [0], True, dtype = torch.float32)
        view_1419: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_206, [4096]);  sum_206 = None
        convert_element_type_2454: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1419, torch.bfloat16);  view_1419 = None
        view_1420: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_402, [1, 128, 16384]);  mm_402 = None
        convert_element_type_2455: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1420, torch.float32);  view_1420 = None
        convert_element_type_2456: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_403, torch.float32);  mm_403 = None
        convert_element_type_2457: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2454, torch.float32);  convert_element_type_2454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_109: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [1, 128, 16384]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_795: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2455, mul_36);  mul_36 = None
        convert_element_type_145: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_109, torch.float32)
        pow_4: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_145, 3.0)
        mul_37: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_35: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_109, mul_37);  view_109 = mul_37 = None
        mul_38: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_35, 0.7978845608028654);  add_35 = None
        tanh_3: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_36: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_796: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2455, add_36);  convert_element_type_2455 = add_36 = None
        convert_element_type_2458: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_796, torch.bfloat16);  mul_796 = None
        mul_797: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_161: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_797);  mul_797 = None
        mul_798: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_795, sub_161);  mul_795 = sub_161 = None
        mul_799: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_798, 0.7978845608028654);  mul_798 = None
        convert_element_type_2459: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_799, torch.bfloat16)
        mul_800: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_799, 0.044715);  mul_799 = None
        pow_53: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_145, 2.0);  convert_element_type_145 = None
        mul_801: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_53, 3.0);  pow_53 = None
        mul_802: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_800, mul_801);  mul_800 = mul_801 = None
        convert_element_type_2460: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_802, torch.bfloat16);  mul_802 = None
        add_546: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2459, convert_element_type_2460);  convert_element_type_2459 = convert_element_type_2460 = None
        mul_803: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2458, 0.5);  convert_element_type_2458 = None
        add_547: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_546, mul_803);  add_546 = mul_803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1421: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_547, [128, 16384]);  add_547 = None
        mm_404: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1421, permute_1109);  permute_1109 = None
        permute_1110: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1421, [1, 0])
        mm_405: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1110, view_84);  permute_1110 = None
        sum_207: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1421, [0], True, dtype = torch.float32);  view_1421 = None
        view_1422: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_207, [16384]);  sum_207 = None
        convert_element_type_2465: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1422, torch.bfloat16);  view_1422 = None
        view_1423: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_404, [1, 128, 4096]);  mm_404 = None
        convert_element_type_2466: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1423, torch.float32);  view_1423 = None
        convert_element_type_2467: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_405, torch.float32);  mm_405 = None
        convert_element_type_2468: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2465, torch.float32);  convert_element_type_2465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_406: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1106, view_106);  permute_1106 = view_106 = None
        mm_407: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1418, permute_1115);  view_1418 = permute_1115 = None
        view_1425: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_407, [1, 128, 4096]);  mm_407 = None
        convert_element_type_2473: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_406, torch.float32);  mm_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1426: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1425, [1, 128, 16, 256]);  view_1425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1117: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1426, [0, 2, 1, 3]);  view_1426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1427: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1117, [16, 128, 256]);  permute_1117 = None
        bmm_152: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1118, view_1427);  permute_1118 = None
        bmm_153: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1427, permute_1119);  view_1427 = permute_1119 = None
        view_1428: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_152, [1, 16, 128, 256]);  bmm_152 = None
        view_1429: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_153, [1, 16, 128, 128]);  bmm_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2478: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1429, torch.float32);  view_1429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_804: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2478, div_7);  convert_element_type_2478 = None
        sum_208: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_804, [-1], True)
        neg_129: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_24: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_129, sum_208, mul_804);  neg_129 = sum_208 = mul_804 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2479: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_24, torch.bfloat16);  fma_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_107: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2479, 16.0);  convert_element_type_2479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1430: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_107, [16, 128, 128]);  div_107 = None
        bmm_154: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1120, view_1430);  permute_1120 = None
        bmm_155: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1430, permute_1121);  view_1430 = permute_1121 = None
        view_1431: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_154, [1, 16, 256, 128]);  bmm_154 = None
        view_1432: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_155, [1, 16, 128, 256]);  bmm_155 = None
        convert_element_type_2485: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1431, torch.float32);  view_1431 = None
        permute_1122: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2485, [0, 1, 3, 2]);  convert_element_type_2485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2486: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1122, torch.bfloat16);  permute_1122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_3: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1432, torch.bfloat16);  view_1432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1123: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_3, [0, 2, 1, 3]);  convert_element_type_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1124: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2486, [0, 2, 1, 3]);  convert_element_type_2486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_325: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1123, 3, 0, 64)
        slice_326: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1123, 3, 64, 256);  permute_1123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_327: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1124, 3, 0, 64)
        slice_328: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1124, 3, 64, 256);  permute_1124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_25: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_51, [1, 128, 1, 32, 2]);  unsqueeze_51 = None
        clone_25: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_93: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [1, 128, 1, 64]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_805: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_325, view_93)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1433: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_805, [1, 128, 16, 32, 2]);  mul_805 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_96: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1433, -1, 0)
        select_97: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1433, -1, 1);  view_1433 = None
        neg_130: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_96);  select_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_192: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_130, 3, 1, 9223372036854775807, 2);  neg_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_193: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_97, 3, 0, 9223372036854775807, 2);  select_97 = None
        add_548: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_192, slice_scatter_193);  slice_scatter_192 = slice_scatter_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_26: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_53, [1, 128, 1, 32, 2]);  unsqueeze_53 = None
        clone_26: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_94: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [1, 128, 1, 64]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_806: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_325, view_94);  slice_325 = None
        add_549: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_548, mul_806);  add_548 = mul_806 = None
        mul_807: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_327, view_93);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1434: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_807, [1, 128, 16, 32, 2]);  mul_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_98: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1434, -1, 0)
        select_99: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1434, -1, 1);  view_1434 = None
        neg_131: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_98);  select_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_194: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_131, 3, 1, 9223372036854775807, 2);  neg_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_195: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_99, 3, 0, 9223372036854775807, 2);  select_99 = None
        add_550: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_194, slice_scatter_195);  slice_scatter_194 = slice_scatter_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_808: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_327, view_94);  slice_327 = view_94 = None
        add_551: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_550, mul_808);  add_550 = mul_808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_196: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_326, 3, 64, 9223372036854775807);  slice_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_197: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_549, 3, 0, 64);  add_549 = None
        add_552: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_196, slice_scatter_197);  slice_scatter_196 = slice_scatter_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_198: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_328, 3, 64, 9223372036854775807);  slice_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_199: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_551, 3, 0, 64);  add_551 = None
        add_553: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_198, slice_scatter_199);  slice_scatter_198 = slice_scatter_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1125: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1428, [0, 2, 1, 3]);  view_1428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_249: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_1125, memory_format = torch.contiguous_format);  permute_1125 = None
        view_1435: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_249, [1, 128, 4096]);  clone_249 = None
        view_1436: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_553, [1, 128, 4096]);  add_553 = None
        view_1437: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_552, [1, 128, 4096]);  add_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1438: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1435, [128, 4096]);  view_1435 = None
        permute_1126: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1438, [1, 0])
        mm_408: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1126, view_84);  permute_1126 = None
        mm_409: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1438, permute_1128);  view_1438 = permute_1128 = None
        view_1439: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_409, [1, 128, 4096]);  mm_409 = None
        convert_element_type_2492: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1439, torch.float32);  view_1439 = None
        add_554: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2466, convert_element_type_2492);  convert_element_type_2466 = convert_element_type_2492 = None
        convert_element_type_2493: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_408, torch.float32);  mm_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1440: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1436, [128, 4096]);  view_1436 = None
        permute_1130: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1440, [1, 0])
        mm_410: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1130, view_84);  permute_1130 = None
        mm_411: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1440, permute_1132);  view_1440 = permute_1132 = None
        view_1441: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_411, [1, 128, 4096]);  mm_411 = None
        convert_element_type_2498: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1441, torch.float32);  view_1441 = None
        add_555: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_554, convert_element_type_2498);  add_554 = convert_element_type_2498 = None
        convert_element_type_2499: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_410, torch.float32);  mm_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1442: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1437, [128, 4096]);  view_1437 = None
        permute_1134: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1442, [1, 0])
        mm_412: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1134, view_84);  permute_1134 = view_84 = None
        mm_413: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1442, permute_1136);  view_1442 = permute_1136 = None
        view_1443: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_413, [1, 128, 4096]);  mm_413 = None
        convert_element_type_2504: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1443, torch.float32);  view_1443 = None
        add_556: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_555, convert_element_type_2504);  add_555 = convert_element_type_2504 = None
        convert_element_type_2505: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_412, torch.float32);  mm_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_810: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_556, primals_36);  primals_36 = None
        mul_811: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_810, 4096)
        sum_209: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_810, [2], True)
        mul_812: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_810, mul_30);  mul_810 = None
        sum_210: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_812, [2], True);  mul_812 = None
        mul_813: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, sum_210);  sum_210 = None
        sub_163: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_811, sum_209);  mul_811 = sum_209 = None
        sub_164: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_163, mul_813);  sub_163 = mul_813 = None
        mul_814: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_108, sub_164);  div_108 = sub_164 = None
        mul_815: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_556, mul_30);  mul_30 = None
        sum_211: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_815, [0, 1]);  mul_815 = None
        sum_212: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_556, [0, 1]);  add_556 = None
        add_557: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_545, mul_814);  add_545 = mul_814 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2506: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_557, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1444: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2506, [128, 4096]);  convert_element_type_2506 = None
        mm_414: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1444, permute_1138);  permute_1138 = None
        permute_1139: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1444, [1, 0])
        mm_415: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_1139, view_82);  view_82 = None
        sum_213: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1444, [0], True, dtype = torch.float32)
        view_1445: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_213, [4096]);  sum_213 = None
        convert_element_type_2511: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1445, torch.bfloat16);  view_1445 = None
        view_1446: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_414, [1, 128, 16384]);  mm_414 = None
        convert_element_type_2512: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1446, torch.float32);  view_1446 = None
        convert_element_type_2513: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_415, torch.float32);  mm_415 = None
        convert_element_type_2514: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2511, torch.float32);  convert_element_type_2511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_81: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 16384]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_26: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_81, 0.5)
        mul_816: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2512, mul_26);  mul_26 = None
        convert_element_type_107: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_81, torch.float32)
        pow_3: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_107, 3.0)
        mul_27: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_26: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_81, mul_27);  view_81 = mul_27 = None
        mul_28: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_26, 0.7978845608028654);  add_26 = None
        tanh_2: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_28);  mul_28 = None
        add_27: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_817: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2512, add_27);  convert_element_type_2512 = add_27 = None
        convert_element_type_2515: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_817, torch.bfloat16);  mul_817 = None
        mul_818: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_165: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_818);  mul_818 = None
        mul_819: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_816, sub_165);  mul_816 = sub_165 = None
        mul_820: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_819, 0.7978845608028654);  mul_819 = None
        convert_element_type_2516: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_820, torch.bfloat16)
        mul_821: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_820, 0.044715);  mul_820 = None
        pow_54: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_107, 2.0);  convert_element_type_107 = None
        mul_822: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_54, 3.0);  pow_54 = None
        mul_823: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_821, mul_822);  mul_821 = mul_822 = None
        convert_element_type_2517: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_823, torch.bfloat16);  mul_823 = None
        add_558: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2516, convert_element_type_2517);  convert_element_type_2516 = convert_element_type_2517 = None
        mul_824: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2515, 0.5);  convert_element_type_2515 = None
        add_559: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_558, mul_824);  add_558 = mul_824 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1447: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_559, [128, 16384]);  add_559 = None
        mm_416: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1447, permute_1142);  permute_1142 = None
        permute_1143: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1447, [1, 0])
        mm_417: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1143, view_56);  permute_1143 = None
        sum_214: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1447, [0], True, dtype = torch.float32);  view_1447 = None
        view_1448: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_214, [16384]);  sum_214 = None
        convert_element_type_2522: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1448, torch.bfloat16);  view_1448 = None
        view_1449: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_416, [1, 128, 4096]);  mm_416 = None
        convert_element_type_2523: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1449, torch.float32);  view_1449 = None
        convert_element_type_2524: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_417, torch.float32);  mm_417 = None
        convert_element_type_2525: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2522, torch.float32);  convert_element_type_2522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_418: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1139, view_78);  permute_1139 = view_78 = None
        mm_419: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1444, permute_1148);  view_1444 = permute_1148 = None
        view_1451: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_419, [1, 128, 4096]);  mm_419 = None
        convert_element_type_2530: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_418, torch.float32);  mm_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1452: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1451, [1, 128, 16, 256]);  view_1451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1150: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1452, [0, 2, 1, 3]);  view_1452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1453: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1150, [16, 128, 256]);  permute_1150 = None
        bmm_156: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1151, view_1453);  permute_1151 = None
        bmm_157: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1453, permute_1152);  view_1453 = permute_1152 = None
        view_1454: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_156, [1, 16, 128, 256]);  bmm_156 = None
        view_1455: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_157, [1, 16, 128, 128]);  bmm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2535: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1455, torch.float32);  view_1455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_825: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2535, div_5);  convert_element_type_2535 = None
        sum_215: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_825, [-1], True)
        neg_132: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_25: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_132, sum_215, mul_825);  neg_132 = sum_215 = mul_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2536: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_25, torch.bfloat16);  fma_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_109: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2536, 16.0);  convert_element_type_2536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1456: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_109, [16, 128, 128]);  div_109 = None
        bmm_158: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1153, view_1456);  permute_1153 = None
        bmm_159: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1456, permute_1154);  view_1456 = permute_1154 = None
        view_1457: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_158, [1, 16, 256, 128]);  bmm_158 = None
        view_1458: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_159, [1, 16, 128, 256]);  bmm_159 = None
        convert_element_type_2542: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1457, torch.float32);  view_1457 = None
        permute_1155: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2542, [0, 1, 3, 2]);  convert_element_type_2542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2543: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1155, torch.bfloat16);  permute_1155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_2: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1458, torch.bfloat16);  view_1458 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1156: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_2, [0, 2, 1, 3]);  convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1157: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2543, [0, 2, 1, 3]);  convert_element_type_2543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_329: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1156, 3, 0, 64)
        slice_330: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1156, 3, 64, 256);  permute_1156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_331: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1157, 3, 0, 64)
        slice_332: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1157, 3, 64, 256);  permute_1157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_17: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_38, [1, 128, 1, 32, 2]);  unsqueeze_38 = None
        clone_17: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_65: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [1, 128, 1, 64]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_826: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_329, view_65)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1459: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_826, [1, 128, 16, 32, 2]);  mul_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_100: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1459, -1, 0)
        select_101: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1459, -1, 1);  view_1459 = None
        neg_133: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_100);  select_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_200: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_133, 3, 1, 9223372036854775807, 2);  neg_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_201: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_101, 3, 0, 9223372036854775807, 2);  select_101 = None
        add_560: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_200, slice_scatter_201);  slice_scatter_200 = slice_scatter_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_18: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_40, [1, 128, 1, 32, 2]);  unsqueeze_40 = None
        clone_18: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_66: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [1, 128, 1, 64]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_827: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_329, view_66);  slice_329 = None
        add_561: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_560, mul_827);  add_560 = mul_827 = None
        mul_828: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_331, view_65);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1460: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_828, [1, 128, 16, 32, 2]);  mul_828 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_102: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1460, -1, 0)
        select_103: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1460, -1, 1);  view_1460 = None
        neg_134: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_102);  select_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_202: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_134, 3, 1, 9223372036854775807, 2);  neg_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_203: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_103, 3, 0, 9223372036854775807, 2);  select_103 = None
        add_562: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_202, slice_scatter_203);  slice_scatter_202 = slice_scatter_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_829: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_331, view_66);  slice_331 = view_66 = None
        add_563: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_562, mul_829);  add_562 = mul_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_204: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_330, 3, 64, 9223372036854775807);  slice_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_205: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_561, 3, 0, 64);  add_561 = None
        add_564: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_204, slice_scatter_205);  slice_scatter_204 = slice_scatter_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_206: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_332, 3, 64, 9223372036854775807);  slice_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_207: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_563, 3, 0, 64);  add_563 = None
        add_565: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_206, slice_scatter_207);  slice_scatter_206 = slice_scatter_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1158: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1454, [0, 2, 1, 3]);  view_1454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_250: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_1158, memory_format = torch.contiguous_format);  permute_1158 = None
        view_1461: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_250, [1, 128, 4096]);  clone_250 = None
        view_1462: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_565, [1, 128, 4096]);  add_565 = None
        view_1463: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_564, [1, 128, 4096]);  add_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1464: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1461, [128, 4096]);  view_1461 = None
        permute_1159: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1464, [1, 0])
        mm_420: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1159, view_56);  permute_1159 = None
        mm_421: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1464, permute_1161);  view_1464 = permute_1161 = None
        view_1465: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_421, [1, 128, 4096]);  mm_421 = None
        convert_element_type_2549: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1465, torch.float32);  view_1465 = None
        add_566: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2523, convert_element_type_2549);  convert_element_type_2523 = convert_element_type_2549 = None
        convert_element_type_2550: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_420, torch.float32);  mm_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1466: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1462, [128, 4096]);  view_1462 = None
        permute_1163: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1466, [1, 0])
        mm_422: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1163, view_56);  permute_1163 = None
        mm_423: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1466, permute_1165);  view_1466 = permute_1165 = None
        view_1467: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_423, [1, 128, 4096]);  mm_423 = None
        convert_element_type_2555: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1467, torch.float32);  view_1467 = None
        add_567: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_566, convert_element_type_2555);  add_566 = convert_element_type_2555 = None
        convert_element_type_2556: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_422, torch.float32);  mm_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1468: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1463, [128, 4096]);  view_1463 = None
        permute_1167: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1468, [1, 0])
        mm_424: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1167, view_56);  permute_1167 = view_56 = None
        mm_425: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1468, permute_1169);  view_1468 = permute_1169 = None
        view_1469: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_425, [1, 128, 4096]);  mm_425 = None
        convert_element_type_2561: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1469, torch.float32);  view_1469 = None
        add_568: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_567, convert_element_type_2561);  add_567 = convert_element_type_2561 = None
        convert_element_type_2562: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_424, torch.float32);  mm_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_831: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_568, primals_25);  primals_25 = None
        mul_832: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_831, 4096)
        sum_216: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_831, [2], True)
        mul_833: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_831, mul_20);  mul_831 = None
        sum_217: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_833, [2], True);  mul_833 = None
        mul_834: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, sum_217);  sum_217 = None
        sub_167: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_832, sum_216);  mul_832 = sum_216 = None
        sub_168: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_167, mul_834);  sub_167 = mul_834 = None
        mul_835: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_110, sub_168);  div_110 = sub_168 = None
        mul_836: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_568, mul_20);  mul_20 = None
        sum_218: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_836, [0, 1]);  mul_836 = None
        sum_219: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_568, [0, 1]);  add_568 = None
        add_569: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_557, mul_835);  add_557 = mul_835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2563: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_569, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1470: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2563, [128, 4096]);  convert_element_type_2563 = None
        mm_426: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1470, permute_1171);  permute_1171 = None
        permute_1172: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1470, [1, 0])
        mm_427: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_1172, view_54);  view_54 = None
        sum_220: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1470, [0], True, dtype = torch.float32)
        view_1471: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_220, [4096]);  sum_220 = None
        convert_element_type_2568: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1471, torch.bfloat16);  view_1471 = None
        view_1472: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_426, [1, 128, 16384]);  mm_426 = None
        convert_element_type_2569: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1472, torch.float32);  view_1472 = None
        convert_element_type_2570: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_427, torch.float32);  mm_427 = None
        convert_element_type_2571: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2568, torch.float32);  convert_element_type_2568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_53: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [1, 128, 16384]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_16: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_53, 0.5)
        mul_837: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2569, mul_16);  mul_16 = None
        convert_element_type_69: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_53, torch.float32)
        pow_2: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_69, 3.0)
        mul_17: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_53, mul_17);  view_53 = mul_17 = None
        mul_18: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
        add_18: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_838: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2569, add_18);  convert_element_type_2569 = add_18 = None
        convert_element_type_2572: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_838, torch.bfloat16);  mul_838 = None
        mul_839: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_169: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_839);  mul_839 = None
        mul_840: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_837, sub_169);  mul_837 = sub_169 = None
        mul_841: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_840, 0.7978845608028654);  mul_840 = None
        convert_element_type_2573: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_841, torch.bfloat16)
        mul_842: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_841, 0.044715);  mul_841 = None
        pow_55: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_69, 2.0);  convert_element_type_69 = None
        mul_843: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_55, 3.0);  pow_55 = None
        mul_844: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_842, mul_843);  mul_842 = mul_843 = None
        convert_element_type_2574: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_844, torch.bfloat16);  mul_844 = None
        add_570: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2573, convert_element_type_2574);  convert_element_type_2573 = convert_element_type_2574 = None
        mul_845: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2572, 0.5);  convert_element_type_2572 = None
        add_571: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_570, mul_845);  add_570 = mul_845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1473: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_571, [128, 16384]);  add_571 = None
        mm_428: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1473, permute_1175);  permute_1175 = None
        permute_1176: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1473, [1, 0])
        mm_429: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1176, view_28);  permute_1176 = None
        sum_221: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1473, [0], True, dtype = torch.float32);  view_1473 = None
        view_1474: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_221, [16384]);  sum_221 = None
        convert_element_type_2579: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1474, torch.bfloat16);  view_1474 = None
        view_1475: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_428, [1, 128, 4096]);  mm_428 = None
        convert_element_type_2580: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1475, torch.float32);  view_1475 = None
        convert_element_type_2581: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_429, torch.float32);  mm_429 = None
        convert_element_type_2582: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2579, torch.float32);  convert_element_type_2579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_430: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1172, view_50);  permute_1172 = view_50 = None
        mm_431: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1470, permute_1181);  view_1470 = permute_1181 = None
        view_1477: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_431, [1, 128, 4096]);  mm_431 = None
        convert_element_type_2587: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_430, torch.float32);  mm_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1478: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1477, [1, 128, 16, 256]);  view_1477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1183: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1478, [0, 2, 1, 3]);  view_1478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1479: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1183, [16, 128, 256]);  permute_1183 = None
        bmm_160: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1184, view_1479);  permute_1184 = None
        bmm_161: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1479, permute_1185);  view_1479 = permute_1185 = None
        view_1480: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_160, [1, 16, 128, 256]);  bmm_160 = None
        view_1481: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_161, [1, 16, 128, 128]);  bmm_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2592: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1481, torch.float32);  view_1481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_846: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2592, div_3);  convert_element_type_2592 = None
        sum_222: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_846, [-1], True)
        neg_135: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_26: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_135, sum_222, mul_846);  neg_135 = sum_222 = mul_846 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2593: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_26, torch.bfloat16);  fma_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_111: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2593, 16.0);  convert_element_type_2593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1482: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_111, [16, 128, 128]);  div_111 = None
        bmm_162: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1186, view_1482);  permute_1186 = None
        bmm_163: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1482, permute_1187);  view_1482 = permute_1187 = None
        view_1483: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_162, [1, 16, 256, 128]);  bmm_162 = None
        view_1484: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_163, [1, 16, 128, 256]);  bmm_163 = None
        convert_element_type_2599: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1483, torch.float32);  view_1483 = None
        permute_1188: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2599, [0, 1, 3, 2]);  convert_element_type_2599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2600: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1188, torch.bfloat16);  permute_1188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_1: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1484, torch.bfloat16);  view_1484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1189: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_1, [0, 2, 1, 3]);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1190: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2600, [0, 2, 1, 3]);  convert_element_type_2600 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_333: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1189, 3, 0, 64)
        slice_334: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1189, 3, 64, 256);  permute_1189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_335: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1190, 3, 0, 64)
        slice_336: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1190, 3, 64, 256);  permute_1190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_9: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_25, [1, 128, 1, 32, 2]);  unsqueeze_25 = None
        clone_9: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_37: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [1, 128, 1, 64]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_847: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_333, view_37)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1485: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_847, [1, 128, 16, 32, 2]);  mul_847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_104: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1485, -1, 0)
        select_105: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1485, -1, 1);  view_1485 = None
        neg_136: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_104);  select_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_208: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_136, 3, 1, 9223372036854775807, 2);  neg_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_209: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_105, 3, 0, 9223372036854775807, 2);  select_105 = None
        add_572: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_208, slice_scatter_209);  slice_scatter_208 = slice_scatter_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_10: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_27, [1, 128, 1, 32, 2]);  unsqueeze_27 = None
        clone_10: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_38: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [1, 128, 1, 64]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_848: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_333, view_38);  slice_333 = None
        add_573: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_572, mul_848);  add_572 = mul_848 = None
        mul_849: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_335, view_37);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1486: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_849, [1, 128, 16, 32, 2]);  mul_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_106: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1486, -1, 0)
        select_107: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1486, -1, 1);  view_1486 = None
        neg_137: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_106);  select_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_210: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_137, 3, 1, 9223372036854775807, 2);  neg_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_211: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_107, 3, 0, 9223372036854775807, 2);  select_107 = None
        add_574: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_210, slice_scatter_211);  slice_scatter_210 = slice_scatter_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_850: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_335, view_38);  slice_335 = view_38 = None
        add_575: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_574, mul_850);  add_574 = mul_850 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_212: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_334, 3, 64, 9223372036854775807);  slice_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_213: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_573, 3, 0, 64);  add_573 = None
        add_576: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_212, slice_scatter_213);  slice_scatter_212 = slice_scatter_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_214: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_336, 3, 64, 9223372036854775807);  slice_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_215: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_575, 3, 0, 64);  add_575 = None
        add_577: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_214, slice_scatter_215);  slice_scatter_214 = slice_scatter_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1191: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1480, [0, 2, 1, 3]);  view_1480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_251: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_1191, memory_format = torch.contiguous_format);  permute_1191 = None
        view_1487: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_251, [1, 128, 4096]);  clone_251 = None
        view_1488: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_577, [1, 128, 4096]);  add_577 = None
        view_1489: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_576, [1, 128, 4096]);  add_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1490: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1487, [128, 4096]);  view_1487 = None
        permute_1192: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1490, [1, 0])
        mm_432: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1192, view_28);  permute_1192 = None
        mm_433: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1490, permute_1194);  view_1490 = permute_1194 = None
        view_1491: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_433, [1, 128, 4096]);  mm_433 = None
        convert_element_type_2606: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1491, torch.float32);  view_1491 = None
        add_578: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2580, convert_element_type_2606);  convert_element_type_2580 = convert_element_type_2606 = None
        convert_element_type_2607: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_432, torch.float32);  mm_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1492: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1488, [128, 4096]);  view_1488 = None
        permute_1196: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1492, [1, 0])
        mm_434: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1196, view_28);  permute_1196 = None
        mm_435: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1492, permute_1198);  view_1492 = permute_1198 = None
        view_1493: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_435, [1, 128, 4096]);  mm_435 = None
        convert_element_type_2612: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1493, torch.float32);  view_1493 = None
        add_579: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_578, convert_element_type_2612);  add_578 = convert_element_type_2612 = None
        convert_element_type_2613: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_434, torch.float32);  mm_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1494: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1489, [128, 4096]);  view_1489 = None
        permute_1200: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1494, [1, 0])
        mm_436: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1200, view_28);  permute_1200 = view_28 = None
        mm_437: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1494, permute_1202);  view_1494 = permute_1202 = None
        view_1495: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_437, [1, 128, 4096]);  mm_437 = None
        convert_element_type_2618: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1495, torch.float32);  view_1495 = None
        add_580: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_579, convert_element_type_2618);  add_579 = convert_element_type_2618 = None
        convert_element_type_2619: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_436, torch.float32);  mm_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_852: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_580, primals_14);  primals_14 = None
        mul_853: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_852, 4096)
        sum_223: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_852, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_11: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_10, embedding);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_4: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_11, getitem_5);  add_11 = getitem_5 = None
        mul_10: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = None
        mul_854: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_852, mul_10);  mul_852 = None
        sum_224: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_854, [2], True);  mul_854 = None
        mul_855: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, sum_224);  sum_224 = None
        sub_171: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_853, sum_223);  mul_853 = sum_223 = None
        sub_172: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_171, mul_855);  sub_171 = mul_855 = None
        div_112: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 4096);  rsqrt_1 = None
        mul_856: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_112, sub_172);  div_112 = sub_172 = None
        mul_857: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_580, mul_10);  mul_10 = None
        sum_225: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_857, [0, 1]);  mul_857 = None
        sum_226: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_580, [0, 1]);  add_580 = None
        add_581: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_569, mul_856);  add_569 = mul_856 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2620: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_581, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1496: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2620, [128, 4096]);  convert_element_type_2620 = None
        mm_438: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1496, permute_1204);  permute_1204 = None
        permute_1205: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1496, [1, 0])
        mm_439: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_1205, view_26);  view_26 = None
        sum_227: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1496, [0], True, dtype = torch.float32)
        view_1497: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_227, [4096]);  sum_227 = None
        convert_element_type_2625: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1497, torch.bfloat16);  view_1497 = None
        view_1498: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_438, [1, 128, 16384]);  mm_438 = None
        convert_element_type_2626: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1498, torch.float32);  view_1498 = None
        convert_element_type_2627: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_439, torch.float32);  mm_439 = None
        convert_element_type_2628: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2625, torch.float32);  convert_element_type_2625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_25: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [1, 128, 16384]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_6: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_25, 0.5)
        mul_858: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2626, mul_6);  mul_6 = None
        convert_element_type_31: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_25, torch.float32)
        pow_1: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_31, 3.0)
        mul_7: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_8: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_25, mul_7);  view_25 = mul_7 = None
        mul_8: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_8, 0.7978845608028654);  add_8 = None
        tanh: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
        add_9: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_859: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2626, add_9);  convert_element_type_2626 = add_9 = None
        convert_element_type_2629: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_859, torch.bfloat16);  mul_859 = None
        mul_860: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_173: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_860);  mul_860 = None
        mul_861: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_858, sub_173);  mul_858 = sub_173 = None
        mul_862: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_861, 0.7978845608028654);  mul_861 = None
        convert_element_type_2630: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_862, torch.bfloat16)
        mul_863: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_862, 0.044715);  mul_862 = None
        pow_56: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_31, 2.0);  convert_element_type_31 = None
        mul_864: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_56, 3.0);  pow_56 = None
        mul_865: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_863, mul_864);  mul_863 = mul_864 = None
        convert_element_type_2631: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_865, torch.bfloat16);  mul_865 = None
        add_582: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2630, convert_element_type_2631);  convert_element_type_2630 = convert_element_type_2631 = None
        mul_866: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2629, 0.5);  convert_element_type_2629 = None
        add_583: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_582, mul_866);  add_582 = mul_866 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1499: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_583, [128, 16384]);  add_583 = None
        mm_440: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1499, permute_1208);  permute_1208 = None
        permute_1209: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1499, [1, 0])
        mm_441: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1209, view);  permute_1209 = None
        sum_228: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1499, [0], True, dtype = torch.float32);  view_1499 = None
        view_1500: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_228, [16384]);  sum_228 = None
        convert_element_type_2636: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1500, torch.bfloat16);  view_1500 = None
        view_1501: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_440, [1, 128, 4096]);  mm_440 = None
        convert_element_type_2637: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1501, torch.float32);  view_1501 = None
        convert_element_type_2638: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_441, torch.float32);  mm_441 = None
        convert_element_type_2639: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2636, torch.float32);  convert_element_type_2636 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_442: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1205, view_22);  permute_1205 = view_22 = None
        mm_443: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1496, permute_1214);  view_1496 = permute_1214 = None
        view_1503: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_443, [1, 128, 4096]);  mm_443 = None
        convert_element_type_2644: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_442, torch.float32);  mm_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1504: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1503, [1, 128, 16, 256]);  view_1503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1216: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1504, [0, 2, 1, 3]);  view_1504 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1505: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1216, [16, 128, 256]);  permute_1216 = None
        bmm_164: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1217, view_1505);  permute_1217 = None
        bmm_165: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1505, permute_1218);  view_1505 = permute_1218 = None
        view_1506: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_164, [1, 16, 128, 256]);  bmm_164 = None
        view_1507: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_165, [1, 16, 128, 128]);  bmm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2649: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1507, torch.float32);  view_1507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_867: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2649, div_1);  convert_element_type_2649 = None
        sum_229: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_867, [-1], True)
        neg_138: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_27: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_138, sum_229, mul_867);  neg_138 = sum_229 = mul_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2650: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_27, torch.bfloat16);  fma_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_113: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2650, 16.0);  convert_element_type_2650 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1508: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_113, [16, 128, 128]);  div_113 = None
        bmm_166: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1219, view_1508);  permute_1219 = None
        bmm_167: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1508, permute_1220);  view_1508 = permute_1220 = None
        view_1509: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_166, [1, 16, 256, 128]);  bmm_166 = None
        view_1510: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_167, [1, 16, 128, 256]);  bmm_167 = None
        convert_element_type_2656: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1509, torch.float32);  view_1509 = None
        permute_1221: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2656, [0, 1, 3, 2]);  convert_element_type_2656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2657: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1221, torch.bfloat16);  permute_1221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1510, torch.bfloat16);  view_1510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1222: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default, [0, 2, 1, 3]);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1223: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2657, [0, 2, 1, 3]);  convert_element_type_2657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_337: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1222, 3, 0, 64)
        slice_338: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1222, 3, 64, 256);  permute_1222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_339: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1223, 3, 0, 64)
        slice_340: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1223, 3, 64, 256);  permute_1223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_1: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_12, [1, 128, 1, 32, 2]);  unsqueeze_12 = None
        clone_1: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_9: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [1, 128, 1, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_868: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_337, view_9)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1511: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_868, [1, 128, 16, 32, 2]);  mul_868 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_108: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1511, -1, 0)
        select_109: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1511, -1, 1);  view_1511 = None
        neg_139: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_108);  select_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_216: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_139, 3, 1, 9223372036854775807, 2);  neg_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_217: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_109, 3, 0, 9223372036854775807, 2);  select_109 = None
        add_584: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_216, slice_scatter_217);  slice_scatter_216 = slice_scatter_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_2: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_14, [1, 128, 1, 32, 2]);  unsqueeze_14 = None
        clone_2: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_10: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [1, 128, 1, 64]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_869: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_337, view_10);  slice_337 = None
        add_585: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_584, mul_869);  add_584 = mul_869 = None
        mul_870: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_339, view_9);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1512: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_870, [1, 128, 16, 32, 2]);  mul_870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_110: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1512, -1, 0)
        select_111: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1512, -1, 1);  view_1512 = None
        neg_140: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_110);  select_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_218: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, neg_140, 3, 1, 9223372036854775807, 2);  neg_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_219: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_7, select_111, 3, 0, 9223372036854775807, 2);  full_default_7 = select_111 = None
        add_586: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_218, slice_scatter_219);  slice_scatter_218 = slice_scatter_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_871: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_339, view_10);  slice_339 = view_10 = None
        add_587: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_586, mul_871);  add_586 = mul_871 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_220: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_338, 3, 64, 9223372036854775807);  slice_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_221: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_585, 3, 0, 64);  add_585 = None
        add_588: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_220, slice_scatter_221);  slice_scatter_220 = slice_scatter_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_222: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, slice_340, 3, 64, 9223372036854775807);  slice_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_223: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_11, add_587, 3, 0, 64);  full_default_11 = add_587 = None
        add_589: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_222, slice_scatter_223);  slice_scatter_222 = slice_scatter_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1224: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1506, [0, 2, 1, 3]);  view_1506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_252: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_1224, memory_format = torch.contiguous_format);  permute_1224 = None
        view_1513: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_252, [1, 128, 4096]);  clone_252 = None
        view_1514: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_589, [1, 128, 4096]);  add_589 = None
        view_1515: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_588, [1, 128, 4096]);  add_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1516: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1513, [128, 4096]);  view_1513 = None
        permute_1225: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1516, [1, 0])
        mm_444: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1225, view);  permute_1225 = None
        mm_445: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1516, permute_1227);  view_1516 = permute_1227 = None
        view_1517: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_445, [1, 128, 4096]);  mm_445 = None
        convert_element_type_2663: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1517, torch.float32);  view_1517 = None
        add_590: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2637, convert_element_type_2663);  convert_element_type_2637 = convert_element_type_2663 = None
        convert_element_type_2664: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_444, torch.float32);  mm_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1518: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1514, [128, 4096]);  view_1514 = None
        permute_1229: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1518, [1, 0])
        mm_446: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1229, view);  permute_1229 = None
        mm_447: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1518, permute_1231);  view_1518 = permute_1231 = None
        view_1519: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_447, [1, 128, 4096]);  mm_447 = None
        convert_element_type_2669: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1519, torch.float32);  view_1519 = None
        add_591: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_590, convert_element_type_2669);  add_590 = convert_element_type_2669 = None
        convert_element_type_2670: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_446, torch.float32);  mm_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1520: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1515, [128, 4096]);  view_1515 = None
        permute_1233: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1520, [1, 0])
        mm_448: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1233, view);  permute_1233 = view = None
        mm_449: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1520, permute_1235);  view_1520 = permute_1235 = None
        view_1521: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_449, [1, 128, 4096]);  mm_449 = None
        convert_element_type_2675: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1521, torch.float32);  view_1521 = None
        add_592: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_591, convert_element_type_2675);  add_591 = convert_element_type_2675 = None
        convert_element_type_2676: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_448, torch.float32);  mm_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_873: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_592, primals_3);  primals_3 = None
        mul_874: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_873, 4096)
        sum_230: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_873, [2], True)
        sub_2: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(embedding, getitem_1);  embedding = getitem_1 = None
        mul: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_875: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_873, mul);  mul_873 = None
        sum_231: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_875, [2], True);  mul_875 = None
        mul_876: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_231);  sum_231 = None
        sub_175: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_874, sum_230);  mul_874 = sum_230 = None
        sub_176: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_175, mul_876);  sub_175 = mul_876 = None
        div_114: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 4096);  rsqrt = None
        mul_877: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_114, sub_176);  div_114 = sub_176 = None
        mul_878: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_592, mul);  mul = None
        sum_232: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_878, [0, 1]);  mul_878 = None
        sum_233: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_592, [0, 1]);  add_592 = None
        add_593: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_581, mul_877);  add_581 = mul_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:494 in forward, code: inputs_embeds = self.wte(input_ids)
        ge: "b8[1, 128][128, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_1, 0)
        lt: "b8[1, 128][128, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_1, 50400)
        bitwise_and_2: "b8[1, 128][128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne_6: "b8[1, 128][128, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_1, -1)
        bitwise_and_3: "b8[1, 128][128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_6);  bitwise_and_2 = ne_6 = None
        unsqueeze_376: "b8[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_default_231: "f32[50400, 4096][4096, 1]cuda:0" = torch.ops.aten.full.default([50400, 4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate: "f32[50400, 4096][4096, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_231, unsqueeze_376, [primals_1], add_593);  full_default_231 = unsqueeze_376 = primals_1 = add_593 = None
        return (None, _unsafe_masked_index_put_accumulate, sum_232, sum_233, convert_element_type_2676, convert_element_type_2670, convert_element_type_2664, None, convert_element_type_2644, convert_element_type_2638, convert_element_type_2639, convert_element_type_2627, convert_element_type_2628, sum_225, sum_226, convert_element_type_2619, convert_element_type_2613, convert_element_type_2607, None, convert_element_type_2587, convert_element_type_2581, convert_element_type_2582, convert_element_type_2570, convert_element_type_2571, sum_218, sum_219, convert_element_type_2562, convert_element_type_2556, convert_element_type_2550, None, convert_element_type_2530, convert_element_type_2524, convert_element_type_2525, convert_element_type_2513, convert_element_type_2514, sum_211, sum_212, convert_element_type_2505, convert_element_type_2499, convert_element_type_2493, None, convert_element_type_2473, convert_element_type_2467, convert_element_type_2468, convert_element_type_2456, convert_element_type_2457, sum_204, sum_205, convert_element_type_2448, convert_element_type_2442, convert_element_type_2436, None, convert_element_type_2416, convert_element_type_2410, convert_element_type_2411, convert_element_type_2399, convert_element_type_2400, sum_197, sum_198, convert_element_type_2391, convert_element_type_2385, convert_element_type_2379, None, convert_element_type_2359, convert_element_type_2353, convert_element_type_2354, convert_element_type_2342, convert_element_type_2343, sum_190, sum_191, convert_element_type_2334, convert_element_type_2328, convert_element_type_2322, None, convert_element_type_2302, convert_element_type_2296, convert_element_type_2297, convert_element_type_2285, convert_element_type_2286, sum_183, sum_184, convert_element_type_2277, convert_element_type_2271, convert_element_type_2265, None, convert_element_type_2245, convert_element_type_2239, convert_element_type_2240, convert_element_type_2228, convert_element_type_2229, sum_176, sum_177, convert_element_type_2220, convert_element_type_2214, convert_element_type_2208, None, convert_element_type_2188, convert_element_type_2182, convert_element_type_2183, convert_element_type_2171, convert_element_type_2172, sum_169, sum_170, convert_element_type_2163, convert_element_type_2157, convert_element_type_2151, None, convert_element_type_2131, convert_element_type_2125, convert_element_type_2126, convert_element_type_2114, convert_element_type_2115, sum_162, sum_163, convert_element_type_2106, convert_element_type_2100, convert_element_type_2094, None, convert_element_type_2074, convert_element_type_2068, convert_element_type_2069, convert_element_type_2057, convert_element_type_2058, sum_155, sum_156, convert_element_type_2049, convert_element_type_2043, convert_element_type_2037, None, convert_element_type_2017, convert_element_type_2011, convert_element_type_2012, convert_element_type_2000, convert_element_type_2001, sum_148, sum_149, convert_element_type_1992, convert_element_type_1986, convert_element_type_1980, None, convert_element_type_1960, convert_element_type_1954, convert_element_type_1955, convert_element_type_1943, convert_element_type_1944, sum_141, sum_142, convert_element_type_1935, convert_element_type_1929, convert_element_type_1923, None, convert_element_type_1903, convert_element_type_1897, convert_element_type_1898, convert_element_type_1886, convert_element_type_1887, sum_134, sum_135, convert_element_type_1878, convert_element_type_1872, convert_element_type_1866, None, convert_element_type_1846, convert_element_type_1840, convert_element_type_1841, convert_element_type_1829, convert_element_type_1830, sum_127, sum_128, convert_element_type_1821, convert_element_type_1815, convert_element_type_1809, None, convert_element_type_1789, convert_element_type_1783, convert_element_type_1784, convert_element_type_1772, convert_element_type_1773, sum_120, sum_121, convert_element_type_1764, convert_element_type_1758, convert_element_type_1752, None, convert_element_type_1732, convert_element_type_1726, convert_element_type_1727, convert_element_type_1715, convert_element_type_1716, sum_113, sum_114, convert_element_type_1707, convert_element_type_1701, convert_element_type_1695, None, convert_element_type_1675, convert_element_type_1669, convert_element_type_1670, convert_element_type_1658, convert_element_type_1659, sum_106, sum_107, convert_element_type_1650, convert_element_type_1644, convert_element_type_1638, None, convert_element_type_1618, convert_element_type_1612, convert_element_type_1613, convert_element_type_1601, convert_element_type_1602, sum_99, sum_100, convert_element_type_1593, convert_element_type_1587, convert_element_type_1581, None, convert_element_type_1561, convert_element_type_1555, convert_element_type_1556, convert_element_type_1544, convert_element_type_1545, sum_92, sum_93, convert_element_type_1536, convert_element_type_1530, convert_element_type_1524, None, convert_element_type_1504, convert_element_type_1498, convert_element_type_1499, convert_element_type_1487, convert_element_type_1488, sum_85, sum_86, convert_element_type_1479, convert_element_type_1473, convert_element_type_1467, None, convert_element_type_1447, convert_element_type_1441, convert_element_type_1442, convert_element_type_1430, convert_element_type_1431, sum_78, sum_79, convert_element_type_1422, convert_element_type_1416, convert_element_type_1410, None, convert_element_type_1390, convert_element_type_1384, convert_element_type_1385, convert_element_type_1373, convert_element_type_1374, sum_71, sum_72, convert_element_type_1365, convert_element_type_1359, convert_element_type_1353, None, convert_element_type_1333, convert_element_type_1327, convert_element_type_1328, convert_element_type_1316, convert_element_type_1317, sum_64, sum_65, convert_element_type_1308, convert_element_type_1302, convert_element_type_1296, None, convert_element_type_1276, convert_element_type_1270, convert_element_type_1271, convert_element_type_1259, convert_element_type_1260, sum_57, sum_58, convert_element_type_1251, convert_element_type_1245, convert_element_type_1239, None, convert_element_type_1219, convert_element_type_1213, convert_element_type_1214, convert_element_type_1202, convert_element_type_1203, sum_50, sum_51, convert_element_type_1194, convert_element_type_1188, convert_element_type_1182, None, convert_element_type_1162, convert_element_type_1156, convert_element_type_1157, convert_element_type_1145, convert_element_type_1146, sum_43, sum_44, convert_element_type_1137, convert_element_type_1131, convert_element_type_1125, None, convert_element_type_1105, convert_element_type_1099, convert_element_type_1100, convert_element_type_1088, convert_element_type_1089, sum_36, sum_37, convert_element_type_1079, convert_element_type_1080, None)
