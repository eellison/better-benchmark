class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[1, 128][128, 1]cuda:0", primals_3: "f32[4096][1]cuda:0", primals_14: "f32[4096][1]cuda:0", primals_25: "f32[4096][1]cuda:0", primals_36: "f32[4096][1]cuda:0", primals_47: "f32[4096][1]cuda:0", primals_58: "f32[4096][1]cuda:0", primals_69: "f32[4096][1]cuda:0", primals_80: "f32[4096][1]cuda:0", primals_91: "f32[4096][1]cuda:0", primals_102: "f32[4096][1]cuda:0", primals_113: "f32[4096][1]cuda:0", primals_124: "f32[4096][1]cuda:0", primals_135: "f32[4096][1]cuda:0", primals_146: "f32[4096][1]cuda:0", primals_157: "f32[4096][1]cuda:0", primals_168: "f32[4096][1]cuda:0", primals_179: "f32[4096][1]cuda:0", primals_190: "f32[4096][1]cuda:0", primals_201: "f32[4096][1]cuda:0", primals_212: "f32[4096][1]cuda:0", primals_223: "f32[4096][1]cuda:0", primals_234: "f32[4096][1]cuda:0", primals_245: "f32[4096][1]cuda:0", primals_256: "f32[4096][1]cuda:0", primals_267: "f32[4096][1]cuda:0", primals_278: "f32[4096][1]cuda:0", primals_289: "f32[4096][1]cuda:0", primals_300: "f32[4096][1]cuda:0", primals_311: "f32[4096][1]cuda:0", primals_315: "i64[1][1]cuda:0", primals_316: "i64[1][1]cuda:0", embedding: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", getitem_1: "f32[1, 128, 1][128, 1, 1]cuda:0", rsqrt: "f32[1, 128, 1][128, 1, 1]cuda:0", view: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_12: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_14: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_1: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_22: "bf16[128, 4096][4096, 1]cuda:0", addmm: "bf16[128, 16384][16384, 1]cuda:0", view_26: "bf16[128, 16384][16384, 1]cuda:0", add_10: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0", getitem_5: "f32[1, 128, 1][128, 1, 1]cuda:0", rsqrt_1: "f32[1, 128, 1][128, 1, 1]cuda:0", view_28: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_25: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_27: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_3: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_50: "bf16[128, 4096][4096, 1]cuda:0", addmm_2: "bf16[128, 16384][16384, 1]cuda:0", view_54: "bf16[128, 16384][16384, 1]cuda:0", mul_20: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_56: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_38: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_40: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_5: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_78: "bf16[128, 4096][4096, 1]cuda:0", addmm_4: "bf16[128, 16384][16384, 1]cuda:0", view_82: "bf16[128, 16384][16384, 1]cuda:0", mul_30: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_84: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_51: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_53: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_7: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_106: "bf16[128, 4096][4096, 1]cuda:0", addmm_6: "bf16[128, 16384][16384, 1]cuda:0", view_110: "bf16[128, 16384][16384, 1]cuda:0", mul_40: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_112: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_64: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_66: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_9: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_134: "bf16[128, 4096][4096, 1]cuda:0", addmm_8: "bf16[128, 16384][16384, 1]cuda:0", view_138: "bf16[128, 16384][16384, 1]cuda:0", mul_50: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_140: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_77: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_79: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_11: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_162: "bf16[128, 4096][4096, 1]cuda:0", addmm_10: "bf16[128, 16384][16384, 1]cuda:0", view_166: "bf16[128, 16384][16384, 1]cuda:0", mul_60: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_168: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_90: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_92: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_13: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_190: "bf16[128, 4096][4096, 1]cuda:0", addmm_12: "bf16[128, 16384][16384, 1]cuda:0", view_194: "bf16[128, 16384][16384, 1]cuda:0", mul_70: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_196: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_103: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_105: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_15: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_218: "bf16[128, 4096][4096, 1]cuda:0", addmm_14: "bf16[128, 16384][16384, 1]cuda:0", view_222: "bf16[128, 16384][16384, 1]cuda:0", mul_80: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_224: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_116: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_118: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_17: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_246: "bf16[128, 4096][4096, 1]cuda:0", addmm_16: "bf16[128, 16384][16384, 1]cuda:0", view_250: "bf16[128, 16384][16384, 1]cuda:0", mul_90: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_252: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_129: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_131: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_19: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_274: "bf16[128, 4096][4096, 1]cuda:0", addmm_18: "bf16[128, 16384][16384, 1]cuda:0", view_278: "bf16[128, 16384][16384, 1]cuda:0", mul_100: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_280: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_142: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_144: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_21: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_302: "bf16[128, 4096][4096, 1]cuda:0", addmm_20: "bf16[128, 16384][16384, 1]cuda:0", view_306: "bf16[128, 16384][16384, 1]cuda:0", mul_110: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_308: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_155: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_157: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_23: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_330: "bf16[128, 4096][4096, 1]cuda:0", addmm_22: "bf16[128, 16384][16384, 1]cuda:0", view_334: "bf16[128, 16384][16384, 1]cuda:0", mul_120: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_336: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_168: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_170: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_25: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_358: "bf16[128, 4096][4096, 1]cuda:0", addmm_24: "bf16[128, 16384][16384, 1]cuda:0", view_362: "bf16[128, 16384][16384, 1]cuda:0", mul_130: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_364: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_181: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_183: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_27: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_386: "bf16[128, 4096][4096, 1]cuda:0", addmm_26: "bf16[128, 16384][16384, 1]cuda:0", view_390: "bf16[128, 16384][16384, 1]cuda:0", mul_140: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_392: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_194: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_196: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_29: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_414: "bf16[128, 4096][4096, 1]cuda:0", addmm_28: "bf16[128, 16384][16384, 1]cuda:0", view_418: "bf16[128, 16384][16384, 1]cuda:0", mul_150: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_420: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_207: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_209: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_31: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_442: "bf16[128, 4096][4096, 1]cuda:0", addmm_30: "bf16[128, 16384][16384, 1]cuda:0", view_446: "bf16[128, 16384][16384, 1]cuda:0", mul_160: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_448: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_220: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_222: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_33: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_470: "bf16[128, 4096][4096, 1]cuda:0", addmm_32: "bf16[128, 16384][16384, 1]cuda:0", view_474: "bf16[128, 16384][16384, 1]cuda:0", mul_170: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_476: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_233: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_235: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_35: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_498: "bf16[128, 4096][4096, 1]cuda:0", addmm_34: "bf16[128, 16384][16384, 1]cuda:0", view_502: "bf16[128, 16384][16384, 1]cuda:0", mul_180: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_504: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_246: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_248: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_37: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_526: "bf16[128, 4096][4096, 1]cuda:0", addmm_36: "bf16[128, 16384][16384, 1]cuda:0", view_530: "bf16[128, 16384][16384, 1]cuda:0", mul_190: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_532: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_259: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_261: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_39: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_554: "bf16[128, 4096][4096, 1]cuda:0", addmm_38: "bf16[128, 16384][16384, 1]cuda:0", view_558: "bf16[128, 16384][16384, 1]cuda:0", mul_200: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_560: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_272: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_274: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_41: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_582: "bf16[128, 4096][4096, 1]cuda:0", addmm_40: "bf16[128, 16384][16384, 1]cuda:0", view_586: "bf16[128, 16384][16384, 1]cuda:0", mul_210: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_588: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_285: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_287: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_43: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_610: "bf16[128, 4096][4096, 1]cuda:0", addmm_42: "bf16[128, 16384][16384, 1]cuda:0", view_614: "bf16[128, 16384][16384, 1]cuda:0", mul_220: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_616: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_298: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_300: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_45: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_638: "bf16[128, 4096][4096, 1]cuda:0", addmm_44: "bf16[128, 16384][16384, 1]cuda:0", view_642: "bf16[128, 16384][16384, 1]cuda:0", mul_230: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_644: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_311: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_313: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_47: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_666: "bf16[128, 4096][4096, 1]cuda:0", addmm_46: "bf16[128, 16384][16384, 1]cuda:0", view_670: "bf16[128, 16384][16384, 1]cuda:0", mul_240: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_672: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_324: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_326: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_49: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_694: "bf16[128, 4096][4096, 1]cuda:0", addmm_48: "bf16[128, 16384][16384, 1]cuda:0", view_698: "bf16[128, 16384][16384, 1]cuda:0", mul_250: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_700: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_337: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_339: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_51: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_722: "bf16[128, 4096][4096, 1]cuda:0", addmm_50: "bf16[128, 16384][16384, 1]cuda:0", view_726: "bf16[128, 16384][16384, 1]cuda:0", mul_260: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_728: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_350: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_352: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_53: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_750: "bf16[128, 4096][4096, 1]cuda:0", addmm_52: "bf16[128, 16384][16384, 1]cuda:0", view_754: "bf16[128, 16384][16384, 1]cuda:0", mul_270: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_756: "bf16[128, 4096][4096, 1]cuda:0", unsqueeze_363: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", unsqueeze_365: "bf16[1, 128, 1, 32, 1][8192, 64, 32, 1, 1]cuda:0", div_55: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_778: "bf16[128, 4096][4096, 1]cuda:0", addmm_54: "bf16[128, 16384][16384, 1]cuda:0", view_782: "bf16[128, 16384][16384, 1]cuda:0", mul_280: "f32[1, 128, 4096][524288, 4096, 1]cuda:0", view_785: "bf16[128, 4096][4096, 1]cuda:0", clone_225: "bf16[1, 128][128, 1]cuda:0", clone_226: "bf16[1, 128][128, 1]cuda:0", amax_28: "f32[1, 1][1, 1]cuda:0", log: "f32[1, 1][1, 1]cuda:0", ne_1: "b8[1][1]cuda:0", amax_29: "f32[1, 1][1, 1]cuda:0", log_1: "f32[1, 1][1, 1]cuda:0", ne_4: "b8[1][1]cuda:0", permute_309: "bf16[2, 4096][4096, 1]cuda:0", div_62: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_313: "bf16[4096, 16384][16384, 1]cuda:0", permute_317: "bf16[16384, 4096][4096, 1]cuda:0", permute_323: "bf16[4096, 4096][4096, 1]cuda:0", permute_326: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_327: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_328: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_329: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_336: "bf16[4096, 4096][4096, 1]cuda:0", permute_340: "bf16[4096, 4096][4096, 1]cuda:0", permute_344: "bf16[4096, 4096][4096, 1]cuda:0", div_64: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_346: "bf16[4096, 16384][16384, 1]cuda:0", permute_350: "bf16[16384, 4096][4096, 1]cuda:0", permute_356: "bf16[4096, 4096][4096, 1]cuda:0", permute_359: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_360: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_361: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_362: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_369: "bf16[4096, 4096][4096, 1]cuda:0", permute_373: "bf16[4096, 4096][4096, 1]cuda:0", permute_377: "bf16[4096, 4096][4096, 1]cuda:0", div_66: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_379: "bf16[4096, 16384][16384, 1]cuda:0", permute_383: "bf16[16384, 4096][4096, 1]cuda:0", permute_389: "bf16[4096, 4096][4096, 1]cuda:0", permute_392: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_393: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_394: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_395: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_402: "bf16[4096, 4096][4096, 1]cuda:0", permute_406: "bf16[4096, 4096][4096, 1]cuda:0", permute_410: "bf16[4096, 4096][4096, 1]cuda:0", div_68: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_412: "bf16[4096, 16384][16384, 1]cuda:0", permute_416: "bf16[16384, 4096][4096, 1]cuda:0", permute_422: "bf16[4096, 4096][4096, 1]cuda:0", permute_425: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_426: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_427: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_428: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_435: "bf16[4096, 4096][4096, 1]cuda:0", permute_439: "bf16[4096, 4096][4096, 1]cuda:0", permute_443: "bf16[4096, 4096][4096, 1]cuda:0", div_70: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_445: "bf16[4096, 16384][16384, 1]cuda:0", permute_449: "bf16[16384, 4096][4096, 1]cuda:0", permute_455: "bf16[4096, 4096][4096, 1]cuda:0", permute_458: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_459: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_460: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_461: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_468: "bf16[4096, 4096][4096, 1]cuda:0", permute_472: "bf16[4096, 4096][4096, 1]cuda:0", permute_476: "bf16[4096, 4096][4096, 1]cuda:0", div_72: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_478: "bf16[4096, 16384][16384, 1]cuda:0", permute_482: "bf16[16384, 4096][4096, 1]cuda:0", permute_488: "bf16[4096, 4096][4096, 1]cuda:0", permute_491: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_492: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_493: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_494: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_501: "bf16[4096, 4096][4096, 1]cuda:0", permute_505: "bf16[4096, 4096][4096, 1]cuda:0", permute_509: "bf16[4096, 4096][4096, 1]cuda:0", div_74: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_511: "bf16[4096, 16384][16384, 1]cuda:0", permute_515: "bf16[16384, 4096][4096, 1]cuda:0", permute_521: "bf16[4096, 4096][4096, 1]cuda:0", permute_524: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_525: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_526: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_527: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_534: "bf16[4096, 4096][4096, 1]cuda:0", permute_538: "bf16[4096, 4096][4096, 1]cuda:0", permute_542: "bf16[4096, 4096][4096, 1]cuda:0", div_76: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_544: "bf16[4096, 16384][16384, 1]cuda:0", permute_548: "bf16[16384, 4096][4096, 1]cuda:0", permute_554: "bf16[4096, 4096][4096, 1]cuda:0", permute_557: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_558: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_559: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_560: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_567: "bf16[4096, 4096][4096, 1]cuda:0", permute_571: "bf16[4096, 4096][4096, 1]cuda:0", permute_575: "bf16[4096, 4096][4096, 1]cuda:0", div_78: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_577: "bf16[4096, 16384][16384, 1]cuda:0", permute_581: "bf16[16384, 4096][4096, 1]cuda:0", permute_587: "bf16[4096, 4096][4096, 1]cuda:0", permute_590: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_591: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_592: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_593: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_600: "bf16[4096, 4096][4096, 1]cuda:0", permute_604: "bf16[4096, 4096][4096, 1]cuda:0", permute_608: "bf16[4096, 4096][4096, 1]cuda:0", div_80: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_610: "bf16[4096, 16384][16384, 1]cuda:0", permute_614: "bf16[16384, 4096][4096, 1]cuda:0", permute_620: "bf16[4096, 4096][4096, 1]cuda:0", permute_623: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_624: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_625: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_626: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_633: "bf16[4096, 4096][4096, 1]cuda:0", permute_637: "bf16[4096, 4096][4096, 1]cuda:0", permute_641: "bf16[4096, 4096][4096, 1]cuda:0", div_82: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_643: "bf16[4096, 16384][16384, 1]cuda:0", permute_647: "bf16[16384, 4096][4096, 1]cuda:0", permute_653: "bf16[4096, 4096][4096, 1]cuda:0", permute_656: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_657: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_658: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_659: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_666: "bf16[4096, 4096][4096, 1]cuda:0", permute_670: "bf16[4096, 4096][4096, 1]cuda:0", permute_674: "bf16[4096, 4096][4096, 1]cuda:0", div_84: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_676: "bf16[4096, 16384][16384, 1]cuda:0", permute_680: "bf16[16384, 4096][4096, 1]cuda:0", permute_686: "bf16[4096, 4096][4096, 1]cuda:0", permute_689: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_690: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_691: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_692: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_699: "bf16[4096, 4096][4096, 1]cuda:0", permute_703: "bf16[4096, 4096][4096, 1]cuda:0", permute_707: "bf16[4096, 4096][4096, 1]cuda:0", div_86: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_709: "bf16[4096, 16384][16384, 1]cuda:0", permute_713: "bf16[16384, 4096][4096, 1]cuda:0", permute_719: "bf16[4096, 4096][4096, 1]cuda:0", permute_722: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_723: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_724: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_725: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_732: "bf16[4096, 4096][4096, 1]cuda:0", permute_736: "bf16[4096, 4096][4096, 1]cuda:0", permute_740: "bf16[4096, 4096][4096, 1]cuda:0", div_88: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_742: "bf16[4096, 16384][16384, 1]cuda:0", permute_746: "bf16[16384, 4096][4096, 1]cuda:0", permute_752: "bf16[4096, 4096][4096, 1]cuda:0", permute_755: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_756: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_757: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_758: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_765: "bf16[4096, 4096][4096, 1]cuda:0", permute_769: "bf16[4096, 4096][4096, 1]cuda:0", permute_773: "bf16[4096, 4096][4096, 1]cuda:0", div_90: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_775: "bf16[4096, 16384][16384, 1]cuda:0", permute_779: "bf16[16384, 4096][4096, 1]cuda:0", permute_785: "bf16[4096, 4096][4096, 1]cuda:0", permute_788: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_789: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_790: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_791: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_798: "bf16[4096, 4096][4096, 1]cuda:0", permute_802: "bf16[4096, 4096][4096, 1]cuda:0", permute_806: "bf16[4096, 4096][4096, 1]cuda:0", div_92: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_808: "bf16[4096, 16384][16384, 1]cuda:0", permute_812: "bf16[16384, 4096][4096, 1]cuda:0", permute_818: "bf16[4096, 4096][4096, 1]cuda:0", permute_821: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_822: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_823: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_824: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_831: "bf16[4096, 4096][4096, 1]cuda:0", permute_835: "bf16[4096, 4096][4096, 1]cuda:0", permute_839: "bf16[4096, 4096][4096, 1]cuda:0", div_94: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_841: "bf16[4096, 16384][16384, 1]cuda:0", permute_845: "bf16[16384, 4096][4096, 1]cuda:0", permute_851: "bf16[4096, 4096][4096, 1]cuda:0", permute_854: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_855: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_856: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_857: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_864: "bf16[4096, 4096][4096, 1]cuda:0", permute_868: "bf16[4096, 4096][4096, 1]cuda:0", permute_872: "bf16[4096, 4096][4096, 1]cuda:0", div_96: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_874: "bf16[4096, 16384][16384, 1]cuda:0", permute_878: "bf16[16384, 4096][4096, 1]cuda:0", permute_884: "bf16[4096, 4096][4096, 1]cuda:0", permute_887: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_888: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_889: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_890: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_897: "bf16[4096, 4096][4096, 1]cuda:0", permute_901: "bf16[4096, 4096][4096, 1]cuda:0", permute_905: "bf16[4096, 4096][4096, 1]cuda:0", div_98: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_907: "bf16[4096, 16384][16384, 1]cuda:0", permute_911: "bf16[16384, 4096][4096, 1]cuda:0", permute_917: "bf16[4096, 4096][4096, 1]cuda:0", permute_920: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_921: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_922: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_923: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_930: "bf16[4096, 4096][4096, 1]cuda:0", permute_934: "bf16[4096, 4096][4096, 1]cuda:0", permute_938: "bf16[4096, 4096][4096, 1]cuda:0", div_100: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_940: "bf16[4096, 16384][16384, 1]cuda:0", permute_944: "bf16[16384, 4096][4096, 1]cuda:0", permute_950: "bf16[4096, 4096][4096, 1]cuda:0", permute_953: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_954: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_955: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_956: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_963: "bf16[4096, 4096][4096, 1]cuda:0", permute_967: "bf16[4096, 4096][4096, 1]cuda:0", permute_971: "bf16[4096, 4096][4096, 1]cuda:0", div_102: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_973: "bf16[4096, 16384][16384, 1]cuda:0", permute_977: "bf16[16384, 4096][4096, 1]cuda:0", permute_983: "bf16[4096, 4096][4096, 1]cuda:0", permute_986: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_987: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_988: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_989: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_996: "bf16[4096, 4096][4096, 1]cuda:0", permute_1000: "bf16[4096, 4096][4096, 1]cuda:0", permute_1004: "bf16[4096, 4096][4096, 1]cuda:0", div_104: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_1006: "bf16[4096, 16384][16384, 1]cuda:0", permute_1010: "bf16[16384, 4096][4096, 1]cuda:0", permute_1016: "bf16[4096, 4096][4096, 1]cuda:0", permute_1019: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_1020: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1021: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1022: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_1029: "bf16[4096, 4096][4096, 1]cuda:0", permute_1033: "bf16[4096, 4096][4096, 1]cuda:0", permute_1037: "bf16[4096, 4096][4096, 1]cuda:0", div_106: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_1039: "bf16[4096, 16384][16384, 1]cuda:0", permute_1043: "bf16[16384, 4096][4096, 1]cuda:0", permute_1049: "bf16[4096, 4096][4096, 1]cuda:0", permute_1052: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_1053: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1054: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1055: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_1062: "bf16[4096, 4096][4096, 1]cuda:0", permute_1066: "bf16[4096, 4096][4096, 1]cuda:0", permute_1070: "bf16[4096, 4096][4096, 1]cuda:0", div_108: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_1072: "bf16[4096, 16384][16384, 1]cuda:0", permute_1076: "bf16[16384, 4096][4096, 1]cuda:0", permute_1082: "bf16[4096, 4096][4096, 1]cuda:0", permute_1085: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_1086: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1087: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1088: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_1095: "bf16[4096, 4096][4096, 1]cuda:0", permute_1099: "bf16[4096, 4096][4096, 1]cuda:0", permute_1103: "bf16[4096, 4096][4096, 1]cuda:0", div_110: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_1105: "bf16[4096, 16384][16384, 1]cuda:0", permute_1109: "bf16[16384, 4096][4096, 1]cuda:0", permute_1115: "bf16[4096, 4096][4096, 1]cuda:0", permute_1118: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_1119: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1120: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1121: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_1128: "bf16[4096, 4096][4096, 1]cuda:0", permute_1132: "bf16[4096, 4096][4096, 1]cuda:0", permute_1136: "bf16[4096, 4096][4096, 1]cuda:0", div_112: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_1138: "bf16[4096, 16384][16384, 1]cuda:0", permute_1142: "bf16[16384, 4096][4096, 1]cuda:0", permute_1148: "bf16[4096, 4096][4096, 1]cuda:0", permute_1151: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_1152: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1153: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1154: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_1161: "bf16[4096, 4096][4096, 1]cuda:0", permute_1165: "bf16[4096, 4096][4096, 1]cuda:0", permute_1169: "bf16[4096, 4096][4096, 1]cuda:0", div_114: "f32[1, 128, 1][128, 1, 1]cuda:0", permute_1171: "bf16[4096, 16384][16384, 1]cuda:0", permute_1175: "bf16[16384, 4096][4096, 1]cuda:0", permute_1181: "bf16[4096, 4096][4096, 1]cuda:0", permute_1184: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_1185: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1186: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1187: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_1194: "bf16[4096, 4096][4096, 1]cuda:0", permute_1198: "bf16[4096, 4096][4096, 1]cuda:0", permute_1202: "bf16[4096, 4096][4096, 1]cuda:0", permute_1204: "bf16[4096, 16384][16384, 1]cuda:0", permute_1208: "bf16[16384, 4096][4096, 1]cuda:0", permute_1214: "bf16[4096, 4096][4096, 1]cuda:0", permute_1217: "bf16[16, 128, 128][16384, 1, 128]cuda:0", permute_1218: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1219: "bf16[16, 256, 128][256, 1, 4096]cuda:0", permute_1220: "bf16[16, 128, 256][256, 4096, 1]cuda:0", permute_1227: "bf16[4096, 4096][4096, 1]cuda:0", permute_1231: "bf16[4096, 4096][4096, 1]cuda:0", permute_1235: "bf16[4096, 4096][4096, 1]cuda:0", tangents_1: "f32[][]cuda:0", tangents_2: "bf16[1, 128][128, 1]cuda:0", tangents_3: "bf16[1, 128][128, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:835 in forward, code: total_loss = (start_loss + end_loss) / 2
        div_59: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:834 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        sum_33: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_4);  ne_4 = None
        convert_element_type_1077: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_33, torch.float32);  sum_33 = None
        div_60: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(div_59, convert_element_type_1077);  convert_element_type_1077 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:830 in forward, code: end_positions = end_positions.clamp(0, ignored_index)
        clamp_min_1: "i64[1][1]cuda:0" = torch.ops.aten.clamp_min.default(primals_316, 0);  primals_316 = None
        clamp_max_1: "i64[1][1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_1, 128);  clamp_min_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:834 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        unsqueeze_376: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clamp_max_1, 1);  clamp_max_1 = None
        ne_7: "b8[1, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_376, 128)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:833 in forward, code: start_loss = loss_fct(start_logits, start_positions)
        full_default_2: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:834 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        where_5: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_7, unsqueeze_376, full_default_2);  unsqueeze_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:502 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota: "i64[128][1]cuda:0" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # No stacktrace found for following nodes
        view_default_1: "i64[1, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(iota, [1, 128]);  iota = None
        expand_default_1: "i64[1, 128][1, 0]cuda:0" = torch.ops.aten.expand.default(where_5, [1, 128]);  where_5 = None
        eq_tensor_1: "b8[1, 128][128, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_default_1, view_default_1);  expand_default_1 = None
        scalar_tensor_default_2: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_3: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:834 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        where_self_1: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor_1, scalar_tensor_default_3, scalar_tensor_default_2);  eq_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:834 in forward, code: end_loss = loss_fct(end_logits, end_positions)
        where_6: "f32[1, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_7, div_60, full_default);  ne_7 = div_60 = None
        mul_282: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self_1, where_6);  where_self_1 = where_6 = None
        convert_element_type_1078: "bf16[1, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_282, torch.bfloat16);  mul_282 = None
        convert_element_type_1079: "f32[1, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1078, torch.float32);  convert_element_type_1078 = None
        convert_element_type_1074: "f32[1, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_226, torch.float32);  clone_226 = None
        sub_61: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1074, amax_29);  convert_element_type_1074 = amax_29 = None
        sub_62: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_61, log_1);  sub_61 = log_1 = None
        convert_element_type_1075: "bf16[1, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_62, torch.bfloat16);  sub_62 = None
        convert_element_type_1076: "f32[1, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1075, torch.float32);  convert_element_type_1075 = None
        exp_30: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.exp.default(convert_element_type_1076);  convert_element_type_1076 = None
        sum_35: "f32[1, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1079, [1], True)
        mul_283: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_30, sum_35);  exp_30 = sum_35 = None
        sub_63: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1079, mul_283);  convert_element_type_1079 = mul_283 = None
        convert_element_type_1081: "bf16[1, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_63, torch.bfloat16);  sub_63 = None
        add_258: "bf16[1, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_3, convert_element_type_1081);  tangents_3 = convert_element_type_1081 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:833 in forward, code: start_loss = loss_fct(start_logits, start_positions)
        sum_30: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne_1);  ne_1 = None
        convert_element_type_1073: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_30, torch.float32);  sum_30 = None
        div_61: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(div_59, convert_element_type_1073);  div_59 = convert_element_type_1073 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:829 in forward, code: start_positions = start_positions.clamp(0, ignored_index)
        clamp_min: "i64[1][1]cuda:0" = torch.ops.aten.clamp_min.default(primals_315, 0);  primals_315 = None
        clamp_max: "i64[1][1]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min, 128);  clamp_min = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:833 in forward, code: start_loss = loss_fct(start_logits, start_positions)
        unsqueeze_377: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clamp_max, 1);  clamp_max = None
        ne_9: "b8[1, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_377, 128)
        where_7: "i64[1, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_9, unsqueeze_377, full_default_2);  unsqueeze_377 = full_default_2 = None

        # No stacktrace found for following nodes
        expand_default: "i64[1, 128][1, 0]cuda:0" = torch.ops.aten.expand.default(where_7, [1, 128]);  where_7 = None
        eq_tensor: "b8[1, 128][128, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_default, view_default_1);  expand_default = view_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:833 in forward, code: start_loss = loss_fct(start_logits, start_positions)
        where_self: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_3, scalar_tensor_default_2);  eq_tensor = scalar_tensor_default_3 = scalar_tensor_default_2 = None
        where_8: "f32[1, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_9, div_61, full_default);  ne_9 = div_61 = full_default = None
        mul_284: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self, where_8);  where_self = where_8 = None
        convert_element_type_1082: "bf16[1, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_284, torch.bfloat16);  mul_284 = None
        convert_element_type_1083: "f32[1, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1082, torch.float32);  convert_element_type_1082 = None
        convert_element_type_1070: "f32[1, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_225, torch.float32);  clone_225 = None
        sub_59: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1070, amax_28);  convert_element_type_1070 = amax_28 = None
        sub_60: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_59, log);  sub_59 = log = None
        convert_element_type_1071: "bf16[1, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_60, torch.bfloat16);  sub_60 = None
        convert_element_type_1072: "f32[1, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1071, torch.float32);  convert_element_type_1071 = None
        exp_31: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.exp.default(convert_element_type_1072);  convert_element_type_1072 = None
        sum_36: "f32[1, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1083, [1], True)
        mul_285: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_31, sum_36);  exp_31 = sum_36 = None
        sub_64: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1083, mul_285);  convert_element_type_1083 = mul_285 = None
        convert_element_type_1085: "bf16[1, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_64, torch.bfloat16);  sub_64 = None
        add_259: "bf16[1, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, convert_element_type_1085);  tangents_2 = convert_element_type_1085 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:818 in forward, code: end_logits = end_logits.squeeze(-1).contiguous()
        unsqueeze_378: "bf16[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_258, 2);  add_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:817 in forward, code: start_logits = start_logits.squeeze(-1).contiguous()
        unsqueeze_379: "bf16[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_259, 2);  add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:816 in forward, code: start_logits, end_logits = logits.split(1, dim=-1)
        cat_113: "bf16[1, 128, 2][256, 2, 1]cuda:0" = torch.ops.aten.cat.default([unsqueeze_379, unsqueeze_378], 2);  unsqueeze_379 = unsqueeze_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:815 in forward, code: logits = self.qa_outputs(sequence_output)
        view_787: "bf16[128, 2][2, 1]cuda:0" = torch.ops.aten.reshape.default(cat_113, [128, 2]);  cat_113 = None
        constant_pad_nd_default: "bf16[128, 8][8, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_787, [0, 6, 0, 0])
        constant_pad_nd_default_1: "bf16[8, 4096][4096, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_309, [0, 0, 0, 6]);  permute_309 = None
        mm_default: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        permute_310: "bf16[2, 128][1, 2]cuda:0" = torch.ops.aten.permute.default(view_787, [1, 0])
        mm_113: "bf16[2, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_310, view_785);  permute_310 = view_785 = None
        sum_37: "f32[1, 2][2, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_787, [0], True, dtype = torch.float32);  view_787 = None
        view_788: "f32[2][1]cuda:0" = torch.ops.aten.reshape.default(sum_37, [2]);  sum_37 = None
        convert_element_type_1090: "bf16[2][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_788, torch.bfloat16);  view_788 = None
        view_789: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_default, [1, 128, 4096]);  mm_default = None
        convert_element_type_1091: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_789, torch.float32);  view_789 = None
        convert_element_type_1092: "f32[2, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_1093: "f32[2][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1090, torch.float32);  convert_element_type_1090 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:542 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_287: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1091, primals_311);  primals_311 = None
        mul_288: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, 4096)
        sum_38: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True)
        mul_289: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, mul_280);  mul_287 = None
        sum_39: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_289, [2], True);  mul_289 = None
        mul_290: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_280, sum_39);  sum_39 = None
        sub_66: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_288, sum_38);  mul_288 = sum_38 = None
        sub_67: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_66, mul_290);  sub_66 = mul_290 = None
        mul_291: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_62, sub_67);  div_62 = sub_67 = None
        mul_292: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1091, mul_280);  mul_280 = None
        sum_40: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_292, [0, 1]);  mul_292 = None
        sum_41: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1091, [0, 1]);  convert_element_type_1091 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1094: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_291, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_791: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1094, [128, 4096]);  convert_element_type_1094 = None
        mm_114: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_791, permute_313);  permute_313 = None
        permute_314: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_791, [1, 0])
        mm_115: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_314, view_782);  view_782 = None
        sum_42: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_791, [0], True, dtype = torch.float32)
        view_792: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_42, [4096]);  sum_42 = None
        convert_element_type_1099: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_792, torch.bfloat16);  view_792 = None
        view_793: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_114, [1, 128, 16384]);  mm_114 = None
        convert_element_type_1100: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_793, torch.float32);  view_793 = None
        convert_element_type_1101: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_1102: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1099, torch.float32);  convert_element_type_1099 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_781: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [1, 128, 16384]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_276: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_781, 0.5)
        mul_293: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1100, mul_276);  mul_276 = None
        convert_element_type_1057: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_781, torch.float32)
        pow_28: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1057, 3.0)
        mul_277: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_28, 0.044715);  pow_28 = None
        add_251: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_781, mul_277);  view_781 = mul_277 = None
        mul_278: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_251, 0.7978845608028654);  add_251 = None
        tanh_27: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_278);  mul_278 = None
        add_252: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_27, 1.0)
        mul_294: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1100, add_252);  convert_element_type_1100 = add_252 = None
        convert_element_type_1103: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_294, torch.bfloat16);  mul_294 = None
        mul_295: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_27, tanh_27);  tanh_27 = None
        sub_68: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_295);  mul_295 = None
        mul_296: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_293, sub_68);  mul_293 = sub_68 = None
        mul_297: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_296, 0.7978845608028654);  mul_296 = None
        convert_element_type_1104: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_297, torch.bfloat16)
        mul_298: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_297, 0.044715);  mul_297 = None
        pow_29: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1057, 2.0);  convert_element_type_1057 = None
        mul_299: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_29, 3.0);  pow_29 = None
        mul_300: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
        convert_element_type_1105: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_300, torch.bfloat16);  mul_300 = None
        add_260: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1104, convert_element_type_1105);  convert_element_type_1104 = convert_element_type_1105 = None
        mul_301: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1103, 0.5);  convert_element_type_1103 = None
        add_261: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_260, mul_301);  add_260 = mul_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_794: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_261, [128, 16384]);  add_261 = None
        mm_116: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_794, permute_317);  permute_317 = None
        permute_318: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_794, [1, 0])
        mm_117: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_318, view_756);  permute_318 = None
        sum_43: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_794, [0], True, dtype = torch.float32);  view_794 = None
        view_795: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_43, [16384]);  sum_43 = None
        convert_element_type_1110: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_795, torch.bfloat16);  view_795 = None
        view_796: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_116, [1, 128, 4096]);  mm_116 = None
        convert_element_type_1111: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_796, torch.float32);  view_796 = None
        convert_element_type_1112: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_1113: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1110, torch.float32);  convert_element_type_1110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_118: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_314, view_778);  permute_314 = view_778 = None
        mm_119: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_791, permute_323);  view_791 = permute_323 = None
        view_798: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_119, [1, 128, 4096]);  mm_119 = None
        convert_element_type_1118: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_118, torch.float32);  mm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_799: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_798, [1, 128, 16, 256]);  view_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_325: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_799, [0, 2, 1, 3]);  view_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_800: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_325, [16, 128, 256]);  permute_325 = None
        bmm_56: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_326, view_800);  permute_326 = None
        bmm_57: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_800, permute_327);  view_800 = permute_327 = None
        view_801: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_56, [1, 16, 128, 256]);  bmm_56 = None
        view_802: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_57, [1, 16, 128, 128]);  bmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1123: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_802, torch.float32);  view_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_302: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1123, div_55);  convert_element_type_1123 = None
        sum_44: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_302, [-1], True)
        neg_58: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_55);  div_55 = None
        fma: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_58, sum_44, mul_302);  neg_58 = sum_44 = mul_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1124: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_63: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1124, 16.0);  convert_element_type_1124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_803: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_63, [16, 128, 128]);  div_63 = None
        bmm_58: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_328, view_803);  permute_328 = None
        bmm_59: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_803, permute_329);  view_803 = permute_329 = None
        view_804: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_58, [1, 16, 256, 128]);  bmm_58 = None
        view_805: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_59, [1, 16, 128, 256]);  bmm_59 = None
        convert_element_type_1130: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_804, torch.float32);  view_804 = None
        permute_330: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1130, [0, 1, 3, 2]);  convert_element_type_1130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1131: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_330, torch.bfloat16);  permute_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_27: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_805, torch.bfloat16);  view_805 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_331: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_27, [0, 2, 1, 3]);  convert_element_type_default_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_332: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1131, [0, 2, 1, 3]);  convert_element_type_1131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_228: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_331, 3, 0, 64)
        slice_229: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_331, 3, 64, 256);  permute_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_230: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_332, 3, 0, 64)
        slice_231: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_332, 3, 64, 256);  permute_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_217: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_363, [1, 128, 1, 32, 2]);  unsqueeze_363 = None
        clone_217: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_217, memory_format = torch.contiguous_format);  expand_217 = None
        view_765: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_217, [1, 128, 1, 64]);  clone_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_303: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_228, view_765)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_806: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_303, [1, 128, 16, 32, 2]);  mul_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_806, -1, 0)
        select_1: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_806, -1, 1);  view_806 = None
        neg_59: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select);  select = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        full_default_12: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.full.default([1, 128, 16, 64], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_59, 3, 1, 9223372036854775807, 2);  neg_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_1: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_1, 3, 0, 9223372036854775807, 2);  select_1 = None
        add_262: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter, slice_scatter_1);  slice_scatter = slice_scatter_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_218: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_365, [1, 128, 1, 32, 2]);  unsqueeze_365 = None
        clone_218: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_218, memory_format = torch.contiguous_format);  expand_218 = None
        view_766: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_218, [1, 128, 1, 64]);  clone_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_304: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_228, view_766);  slice_228 = None
        add_263: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_262, mul_304);  add_262 = mul_304 = None
        mul_305: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_230, view_765);  view_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_807: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_305, [1, 128, 16, 32, 2]);  mul_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_2: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_807, -1, 0)
        select_3: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_807, -1, 1);  view_807 = None
        neg_60: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_2);  select_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_2: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_60, 3, 1, 9223372036854775807, 2);  neg_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_3: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_3, 3, 0, 9223372036854775807, 2);  select_3 = None
        add_264: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_2, slice_scatter_3);  slice_scatter_2 = slice_scatter_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_306: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_230, view_766);  slice_230 = view_766 = None
        add_265: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_264, mul_306);  add_264 = mul_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        full_default_16: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.full.default([1, 128, 16, 256], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_4: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_229, 3, 64, 9223372036854775807);  slice_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_5: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_263, 3, 0, 64);  add_263 = None
        add_266: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_4, slice_scatter_5);  slice_scatter_4 = slice_scatter_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_6: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_231, 3, 64, 9223372036854775807);  slice_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_7: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_265, 3, 0, 64);  add_265 = None
        add_267: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_6, slice_scatter_7);  slice_scatter_6 = slice_scatter_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_333: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_801, [0, 2, 1, 3]);  view_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_227: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_333, memory_format = torch.contiguous_format);  permute_333 = None
        view_808: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_227, [1, 128, 4096]);  clone_227 = None
        view_809: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_267, [1, 128, 4096]);  add_267 = None
        view_810: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_266, [1, 128, 4096]);  add_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_811: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_808, [128, 4096]);  view_808 = None
        permute_334: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_811, [1, 0])
        mm_120: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_334, view_756);  permute_334 = None
        mm_121: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_811, permute_336);  view_811 = permute_336 = None
        view_812: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_121, [1, 128, 4096]);  mm_121 = None
        convert_element_type_1137: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_812, torch.float32);  view_812 = None
        add_268: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1111, convert_element_type_1137);  convert_element_type_1111 = convert_element_type_1137 = None
        convert_element_type_1138: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_120, torch.float32);  mm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_813: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_809, [128, 4096]);  view_809 = None
        permute_338: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_813, [1, 0])
        mm_122: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_338, view_756);  permute_338 = None
        mm_123: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_813, permute_340);  view_813 = permute_340 = None
        view_814: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_123, [1, 128, 4096]);  mm_123 = None
        convert_element_type_1143: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_814, torch.float32);  view_814 = None
        add_269: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_268, convert_element_type_1143);  add_268 = convert_element_type_1143 = None
        convert_element_type_1144: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_122, torch.float32);  mm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_815: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_810, [128, 4096]);  view_810 = None
        permute_342: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_815, [1, 0])
        mm_124: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_342, view_756);  permute_342 = view_756 = None
        mm_125: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_815, permute_344);  view_815 = permute_344 = None
        view_816: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_125, [1, 128, 4096]);  mm_125 = None
        convert_element_type_1149: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_816, torch.float32);  view_816 = None
        add_270: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_269, convert_element_type_1149);  add_269 = convert_element_type_1149 = None
        convert_element_type_1150: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_124, torch.float32);  mm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_308: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_270, primals_300);  primals_300 = None
        mul_309: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_308, 4096)
        sum_45: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_308, [2], True)
        mul_310: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_308, mul_270);  mul_308 = None
        sum_46: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_310, [2], True);  mul_310 = None
        mul_311: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_270, sum_46);  sum_46 = None
        sub_70: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_309, sum_45);  mul_309 = sum_45 = None
        sub_71: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_70, mul_311);  sub_70 = mul_311 = None
        mul_312: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_64, sub_71);  div_64 = sub_71 = None
        mul_313: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_270, mul_270);  mul_270 = None
        sum_47: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_313, [0, 1]);  mul_313 = None
        sum_48: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_270, [0, 1]);  add_270 = None
        add_271: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_291, mul_312);  mul_291 = mul_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1151: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_271, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_817: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1151, [128, 4096]);  convert_element_type_1151 = None
        mm_126: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_817, permute_346);  permute_346 = None
        permute_347: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_817, [1, 0])
        mm_127: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_347, view_754);  view_754 = None
        sum_49: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_817, [0], True, dtype = torch.float32)
        view_818: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_49, [4096]);  sum_49 = None
        convert_element_type_1156: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_818, torch.bfloat16);  view_818 = None
        view_819: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_126, [1, 128, 16384]);  mm_126 = None
        convert_element_type_1157: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_819, torch.float32);  view_819 = None
        convert_element_type_1158: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_1159: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1156, torch.float32);  convert_element_type_1156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_753: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [1, 128, 16384]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_266: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_753, 0.5)
        mul_314: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1157, mul_266);  mul_266 = None
        convert_element_type_1019: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_753, torch.float32)
        pow_27: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1019, 3.0)
        mul_267: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
        add_242: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_753, mul_267);  view_753 = mul_267 = None
        mul_268: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_242, 0.7978845608028654);  add_242 = None
        tanh_26: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_268);  mul_268 = None
        add_243: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_26, 1.0)
        mul_315: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1157, add_243);  convert_element_type_1157 = add_243 = None
        convert_element_type_1160: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_315, torch.bfloat16);  mul_315 = None
        mul_316: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_26, tanh_26);  tanh_26 = None
        sub_72: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_316);  mul_316 = None
        mul_317: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_314, sub_72);  mul_314 = sub_72 = None
        mul_318: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_317, 0.7978845608028654);  mul_317 = None
        convert_element_type_1161: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_318, torch.bfloat16)
        mul_319: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, 0.044715);  mul_318 = None
        pow_30: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1019, 2.0);  convert_element_type_1019 = None
        mul_320: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_30, 3.0);  pow_30 = None
        mul_321: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
        convert_element_type_1162: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_321, torch.bfloat16);  mul_321 = None
        add_272: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1161, convert_element_type_1162);  convert_element_type_1161 = convert_element_type_1162 = None
        mul_322: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1160, 0.5);  convert_element_type_1160 = None
        add_273: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_272, mul_322);  add_272 = mul_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_820: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_273, [128, 16384]);  add_273 = None
        mm_128: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_820, permute_350);  permute_350 = None
        permute_351: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_820, [1, 0])
        mm_129: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_351, view_728);  permute_351 = None
        sum_50: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_820, [0], True, dtype = torch.float32);  view_820 = None
        view_821: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_50, [16384]);  sum_50 = None
        convert_element_type_1167: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_821, torch.bfloat16);  view_821 = None
        view_822: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_128, [1, 128, 4096]);  mm_128 = None
        convert_element_type_1168: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_822, torch.float32);  view_822 = None
        convert_element_type_1169: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        convert_element_type_1170: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1167, torch.float32);  convert_element_type_1167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_130: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_347, view_750);  permute_347 = view_750 = None
        mm_131: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_817, permute_356);  view_817 = permute_356 = None
        view_824: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_131, [1, 128, 4096]);  mm_131 = None
        convert_element_type_1175: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_130, torch.float32);  mm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_825: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_824, [1, 128, 16, 256]);  view_824 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_358: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_825, [0, 2, 1, 3]);  view_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_826: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_358, [16, 128, 256]);  permute_358 = None
        bmm_60: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_359, view_826);  permute_359 = None
        bmm_61: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_826, permute_360);  view_826 = permute_360 = None
        view_827: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_60, [1, 16, 128, 256]);  bmm_60 = None
        view_828: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_61, [1, 16, 128, 128]);  bmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1180: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_828, torch.float32);  view_828 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_323: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1180, div_53);  convert_element_type_1180 = None
        sum_51: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_323, [-1], True)
        neg_61: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_53);  div_53 = None
        fma_1: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_61, sum_51, mul_323);  neg_61 = sum_51 = mul_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1181: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_65: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1181, 16.0);  convert_element_type_1181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_829: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_65, [16, 128, 128]);  div_65 = None
        bmm_62: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_361, view_829);  permute_361 = None
        bmm_63: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_829, permute_362);  view_829 = permute_362 = None
        view_830: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_62, [1, 16, 256, 128]);  bmm_62 = None
        view_831: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_63, [1, 16, 128, 256]);  bmm_63 = None
        convert_element_type_1187: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_830, torch.float32);  view_830 = None
        permute_363: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1187, [0, 1, 3, 2]);  convert_element_type_1187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1188: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_363, torch.bfloat16);  permute_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_26: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_831, torch.bfloat16);  view_831 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_364: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_26, [0, 2, 1, 3]);  convert_element_type_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_365: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1188, [0, 2, 1, 3]);  convert_element_type_1188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_232: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_364, 3, 0, 64)
        slice_233: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_364, 3, 64, 256);  permute_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_234: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_365, 3, 0, 64)
        slice_235: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_365, 3, 64, 256);  permute_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_209: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_350, [1, 128, 1, 32, 2]);  unsqueeze_350 = None
        clone_209: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_209, memory_format = torch.contiguous_format);  expand_209 = None
        view_737: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_209, [1, 128, 1, 64]);  clone_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_324: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_232, view_737)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_832: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_324, [1, 128, 16, 32, 2]);  mul_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_4: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_832, -1, 0)
        select_5: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_832, -1, 1);  view_832 = None
        neg_62: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_4);  select_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_8: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_62, 3, 1, 9223372036854775807, 2);  neg_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_9: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_5, 3, 0, 9223372036854775807, 2);  select_5 = None
        add_274: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_8, slice_scatter_9);  slice_scatter_8 = slice_scatter_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_210: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_352, [1, 128, 1, 32, 2]);  unsqueeze_352 = None
        clone_210: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_210, memory_format = torch.contiguous_format);  expand_210 = None
        view_738: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_210, [1, 128, 1, 64]);  clone_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_325: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_232, view_738);  slice_232 = None
        add_275: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_274, mul_325);  add_274 = mul_325 = None
        mul_326: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_234, view_737);  view_737 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_833: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_326, [1, 128, 16, 32, 2]);  mul_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_6: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_833, -1, 0)
        select_7: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_833, -1, 1);  view_833 = None
        neg_63: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_6);  select_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_10: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_63, 3, 1, 9223372036854775807, 2);  neg_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_11: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_7, 3, 0, 9223372036854775807, 2);  select_7 = None
        add_276: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_10, slice_scatter_11);  slice_scatter_10 = slice_scatter_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_327: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_234, view_738);  slice_234 = view_738 = None
        add_277: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_276, mul_327);  add_276 = mul_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_12: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_233, 3, 64, 9223372036854775807);  slice_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_13: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_275, 3, 0, 64);  add_275 = None
        add_278: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_12, slice_scatter_13);  slice_scatter_12 = slice_scatter_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_14: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_235, 3, 64, 9223372036854775807);  slice_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_15: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_277, 3, 0, 64);  add_277 = None
        add_279: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_14, slice_scatter_15);  slice_scatter_14 = slice_scatter_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_366: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_827, [0, 2, 1, 3]);  view_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_228: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_366, memory_format = torch.contiguous_format);  permute_366 = None
        view_834: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_228, [1, 128, 4096]);  clone_228 = None
        view_835: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_279, [1, 128, 4096]);  add_279 = None
        view_836: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_278, [1, 128, 4096]);  add_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_837: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_834, [128, 4096]);  view_834 = None
        permute_367: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_837, [1, 0])
        mm_132: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_367, view_728);  permute_367 = None
        mm_133: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_837, permute_369);  view_837 = permute_369 = None
        view_838: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_133, [1, 128, 4096]);  mm_133 = None
        convert_element_type_1194: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_838, torch.float32);  view_838 = None
        add_280: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1168, convert_element_type_1194);  convert_element_type_1168 = convert_element_type_1194 = None
        convert_element_type_1195: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_132, torch.float32);  mm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_839: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_835, [128, 4096]);  view_835 = None
        permute_371: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_839, [1, 0])
        mm_134: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_371, view_728);  permute_371 = None
        mm_135: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_839, permute_373);  view_839 = permute_373 = None
        view_840: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_135, [1, 128, 4096]);  mm_135 = None
        convert_element_type_1200: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_840, torch.float32);  view_840 = None
        add_281: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_280, convert_element_type_1200);  add_280 = convert_element_type_1200 = None
        convert_element_type_1201: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_134, torch.float32);  mm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_841: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_836, [128, 4096]);  view_836 = None
        permute_375: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_841, [1, 0])
        mm_136: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_375, view_728);  permute_375 = view_728 = None
        mm_137: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_841, permute_377);  view_841 = permute_377 = None
        view_842: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_137, [1, 128, 4096]);  mm_137 = None
        convert_element_type_1206: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_842, torch.float32);  view_842 = None
        add_282: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_281, convert_element_type_1206);  add_281 = convert_element_type_1206 = None
        convert_element_type_1207: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_136, torch.float32);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_329: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_282, primals_289);  primals_289 = None
        mul_330: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_329, 4096)
        sum_52: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True)
        mul_331: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_329, mul_260);  mul_329 = None
        sum_53: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_331, [2], True);  mul_331 = None
        mul_332: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_260, sum_53);  sum_53 = None
        sub_74: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_330, sum_52);  mul_330 = sum_52 = None
        sub_75: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_74, mul_332);  sub_74 = mul_332 = None
        mul_333: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_66, sub_75);  div_66 = sub_75 = None
        mul_334: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_282, mul_260);  mul_260 = None
        sum_54: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 1]);  mul_334 = None
        sum_55: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_282, [0, 1]);  add_282 = None
        add_283: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_271, mul_333);  add_271 = mul_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1208: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_283, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_843: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1208, [128, 4096]);  convert_element_type_1208 = None
        mm_138: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_843, permute_379);  permute_379 = None
        permute_380: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_843, [1, 0])
        mm_139: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_380, view_726);  view_726 = None
        sum_56: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_843, [0], True, dtype = torch.float32)
        view_844: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_56, [4096]);  sum_56 = None
        convert_element_type_1213: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_844, torch.bfloat16);  view_844 = None
        view_845: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_138, [1, 128, 16384]);  mm_138 = None
        convert_element_type_1214: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_845, torch.float32);  view_845 = None
        convert_element_type_1215: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_1216: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1213, torch.float32);  convert_element_type_1213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_725: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [1, 128, 16384]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_256: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_725, 0.5)
        mul_335: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1214, mul_256);  mul_256 = None
        convert_element_type_981: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_725, torch.float32)
        pow_26: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_981, 3.0)
        mul_257: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_26, 0.044715);  pow_26 = None
        add_233: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_725, mul_257);  view_725 = mul_257 = None
        mul_258: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_233, 0.7978845608028654);  add_233 = None
        tanh_25: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_258);  mul_258 = None
        add_234: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_25, 1.0)
        mul_336: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1214, add_234);  convert_element_type_1214 = add_234 = None
        convert_element_type_1217: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_336, torch.bfloat16);  mul_336 = None
        mul_337: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_25, tanh_25);  tanh_25 = None
        sub_76: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_337);  mul_337 = None
        mul_338: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_335, sub_76);  mul_335 = sub_76 = None
        mul_339: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_338, 0.7978845608028654);  mul_338 = None
        convert_element_type_1218: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_339, torch.bfloat16)
        mul_340: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_339, 0.044715);  mul_339 = None
        pow_31: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_981, 2.0);  convert_element_type_981 = None
        mul_341: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_31, 3.0);  pow_31 = None
        mul_342: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        convert_element_type_1219: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_342, torch.bfloat16);  mul_342 = None
        add_284: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1218, convert_element_type_1219);  convert_element_type_1218 = convert_element_type_1219 = None
        mul_343: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1217, 0.5);  convert_element_type_1217 = None
        add_285: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_284, mul_343);  add_284 = mul_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_846: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_285, [128, 16384]);  add_285 = None
        mm_140: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_846, permute_383);  permute_383 = None
        permute_384: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_846, [1, 0])
        mm_141: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_384, view_700);  permute_384 = None
        sum_57: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_846, [0], True, dtype = torch.float32);  view_846 = None
        view_847: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_57, [16384]);  sum_57 = None
        convert_element_type_1224: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_847, torch.bfloat16);  view_847 = None
        view_848: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_140, [1, 128, 4096]);  mm_140 = None
        convert_element_type_1225: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_848, torch.float32);  view_848 = None
        convert_element_type_1226: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_1227: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1224, torch.float32);  convert_element_type_1224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_142: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_380, view_722);  permute_380 = view_722 = None
        mm_143: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_843, permute_389);  view_843 = permute_389 = None
        view_850: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_143, [1, 128, 4096]);  mm_143 = None
        convert_element_type_1232: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_142, torch.float32);  mm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_851: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_850, [1, 128, 16, 256]);  view_850 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_391: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_851, [0, 2, 1, 3]);  view_851 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_852: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_391, [16, 128, 256]);  permute_391 = None
        bmm_64: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_392, view_852);  permute_392 = None
        bmm_65: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_852, permute_393);  view_852 = permute_393 = None
        view_853: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_64, [1, 16, 128, 256]);  bmm_64 = None
        view_854: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_65, [1, 16, 128, 128]);  bmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1237: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_854, torch.float32);  view_854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_344: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1237, div_51);  convert_element_type_1237 = None
        sum_58: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_344, [-1], True)
        neg_64: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_51);  div_51 = None
        fma_2: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_64, sum_58, mul_344);  neg_64 = sum_58 = mul_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1238: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_67: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1238, 16.0);  convert_element_type_1238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_855: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_67, [16, 128, 128]);  div_67 = None
        bmm_66: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_394, view_855);  permute_394 = None
        bmm_67: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_855, permute_395);  view_855 = permute_395 = None
        view_856: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_66, [1, 16, 256, 128]);  bmm_66 = None
        view_857: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_67, [1, 16, 128, 256]);  bmm_67 = None
        convert_element_type_1244: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_856, torch.float32);  view_856 = None
        permute_396: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1244, [0, 1, 3, 2]);  convert_element_type_1244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1245: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_396, torch.bfloat16);  permute_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_25: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_857, torch.bfloat16);  view_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_397: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_25, [0, 2, 1, 3]);  convert_element_type_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_398: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1245, [0, 2, 1, 3]);  convert_element_type_1245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_236: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_397, 3, 0, 64)
        slice_237: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_397, 3, 64, 256);  permute_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_238: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_398, 3, 0, 64)
        slice_239: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_398, 3, 64, 256);  permute_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_201: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_337, [1, 128, 1, 32, 2]);  unsqueeze_337 = None
        clone_201: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_201, memory_format = torch.contiguous_format);  expand_201 = None
        view_709: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_201, [1, 128, 1, 64]);  clone_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_345: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_236, view_709)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_858: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_345, [1, 128, 16, 32, 2]);  mul_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_8: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_858, -1, 0)
        select_9: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_858, -1, 1);  view_858 = None
        neg_65: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_8);  select_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_16: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_65, 3, 1, 9223372036854775807, 2);  neg_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_17: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_9, 3, 0, 9223372036854775807, 2);  select_9 = None
        add_286: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_16, slice_scatter_17);  slice_scatter_16 = slice_scatter_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_202: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_339, [1, 128, 1, 32, 2]);  unsqueeze_339 = None
        clone_202: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_202, memory_format = torch.contiguous_format);  expand_202 = None
        view_710: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_202, [1, 128, 1, 64]);  clone_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_346: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_236, view_710);  slice_236 = None
        add_287: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_286, mul_346);  add_286 = mul_346 = None
        mul_347: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_238, view_709);  view_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_859: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_347, [1, 128, 16, 32, 2]);  mul_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_10: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_859, -1, 0)
        select_11: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_859, -1, 1);  view_859 = None
        neg_66: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_10);  select_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_18: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_66, 3, 1, 9223372036854775807, 2);  neg_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_19: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_11, 3, 0, 9223372036854775807, 2);  select_11 = None
        add_288: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_18, slice_scatter_19);  slice_scatter_18 = slice_scatter_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_348: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_238, view_710);  slice_238 = view_710 = None
        add_289: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_288, mul_348);  add_288 = mul_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_20: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_237, 3, 64, 9223372036854775807);  slice_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_21: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_287, 3, 0, 64);  add_287 = None
        add_290: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_20, slice_scatter_21);  slice_scatter_20 = slice_scatter_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_22: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_239, 3, 64, 9223372036854775807);  slice_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_23: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_289, 3, 0, 64);  add_289 = None
        add_291: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_22, slice_scatter_23);  slice_scatter_22 = slice_scatter_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_399: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_853, [0, 2, 1, 3]);  view_853 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_229: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_399, memory_format = torch.contiguous_format);  permute_399 = None
        view_860: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_229, [1, 128, 4096]);  clone_229 = None
        view_861: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_291, [1, 128, 4096]);  add_291 = None
        view_862: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_290, [1, 128, 4096]);  add_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_863: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_860, [128, 4096]);  view_860 = None
        permute_400: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_863, [1, 0])
        mm_144: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_400, view_700);  permute_400 = None
        mm_145: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_863, permute_402);  view_863 = permute_402 = None
        view_864: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_145, [1, 128, 4096]);  mm_145 = None
        convert_element_type_1251: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_864, torch.float32);  view_864 = None
        add_292: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1225, convert_element_type_1251);  convert_element_type_1225 = convert_element_type_1251 = None
        convert_element_type_1252: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_144, torch.float32);  mm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_865: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_861, [128, 4096]);  view_861 = None
        permute_404: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_865, [1, 0])
        mm_146: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_404, view_700);  permute_404 = None
        mm_147: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_865, permute_406);  view_865 = permute_406 = None
        view_866: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_147, [1, 128, 4096]);  mm_147 = None
        convert_element_type_1257: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_866, torch.float32);  view_866 = None
        add_293: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_292, convert_element_type_1257);  add_292 = convert_element_type_1257 = None
        convert_element_type_1258: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_146, torch.float32);  mm_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_867: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_862, [128, 4096]);  view_862 = None
        permute_408: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_867, [1, 0])
        mm_148: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_408, view_700);  permute_408 = view_700 = None
        mm_149: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_867, permute_410);  view_867 = permute_410 = None
        view_868: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_149, [1, 128, 4096]);  mm_149 = None
        convert_element_type_1263: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_868, torch.float32);  view_868 = None
        add_294: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_293, convert_element_type_1263);  add_293 = convert_element_type_1263 = None
        convert_element_type_1264: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_148, torch.float32);  mm_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_350: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_294, primals_278);  primals_278 = None
        mul_351: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_350, 4096)
        sum_59: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_350, [2], True)
        mul_352: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_350, mul_250);  mul_350 = None
        sum_60: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_352, [2], True);  mul_352 = None
        mul_353: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_250, sum_60);  sum_60 = None
        sub_78: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_351, sum_59);  mul_351 = sum_59 = None
        sub_79: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_78, mul_353);  sub_78 = mul_353 = None
        mul_354: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_68, sub_79);  div_68 = sub_79 = None
        mul_355: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_294, mul_250);  mul_250 = None
        sum_61: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_355, [0, 1]);  mul_355 = None
        sum_62: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_294, [0, 1]);  add_294 = None
        add_295: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_283, mul_354);  add_283 = mul_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1265: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_295, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_869: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1265, [128, 4096]);  convert_element_type_1265 = None
        mm_150: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_869, permute_412);  permute_412 = None
        permute_413: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_869, [1, 0])
        mm_151: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_413, view_698);  view_698 = None
        sum_63: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_869, [0], True, dtype = torch.float32)
        view_870: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [4096]);  sum_63 = None
        convert_element_type_1270: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_870, torch.bfloat16);  view_870 = None
        view_871: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_150, [1, 128, 16384]);  mm_150 = None
        convert_element_type_1271: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_871, torch.float32);  view_871 = None
        convert_element_type_1272: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_151, torch.float32);  mm_151 = None
        convert_element_type_1273: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1270, torch.float32);  convert_element_type_1270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_697: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [1, 128, 16384]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_246: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_697, 0.5)
        mul_356: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1271, mul_246);  mul_246 = None
        convert_element_type_943: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_697, torch.float32)
        pow_25: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_943, 3.0)
        mul_247: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
        add_224: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_697, mul_247);  view_697 = mul_247 = None
        mul_248: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_224, 0.7978845608028654);  add_224 = None
        tanh_24: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_248);  mul_248 = None
        add_225: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_24, 1.0)
        mul_357: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1271, add_225);  convert_element_type_1271 = add_225 = None
        convert_element_type_1274: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_357, torch.bfloat16);  mul_357 = None
        mul_358: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_24, tanh_24);  tanh_24 = None
        sub_80: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_358);  mul_358 = None
        mul_359: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_356, sub_80);  mul_356 = sub_80 = None
        mul_360: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_359, 0.7978845608028654);  mul_359 = None
        convert_element_type_1275: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_360, torch.bfloat16)
        mul_361: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_360, 0.044715);  mul_360 = None
        pow_32: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_943, 2.0);  convert_element_type_943 = None
        mul_362: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_32, 3.0);  pow_32 = None
        mul_363: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
        convert_element_type_1276: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_363, torch.bfloat16);  mul_363 = None
        add_296: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1275, convert_element_type_1276);  convert_element_type_1275 = convert_element_type_1276 = None
        mul_364: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1274, 0.5);  convert_element_type_1274 = None
        add_297: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_296, mul_364);  add_296 = mul_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_872: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_297, [128, 16384]);  add_297 = None
        mm_152: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_872, permute_416);  permute_416 = None
        permute_417: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_872, [1, 0])
        mm_153: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_417, view_672);  permute_417 = None
        sum_64: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_872, [0], True, dtype = torch.float32);  view_872 = None
        view_873: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_64, [16384]);  sum_64 = None
        convert_element_type_1281: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_873, torch.bfloat16);  view_873 = None
        view_874: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_152, [1, 128, 4096]);  mm_152 = None
        convert_element_type_1282: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_874, torch.float32);  view_874 = None
        convert_element_type_1283: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_153, torch.float32);  mm_153 = None
        convert_element_type_1284: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1281, torch.float32);  convert_element_type_1281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_154: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_413, view_694);  permute_413 = view_694 = None
        mm_155: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_869, permute_422);  view_869 = permute_422 = None
        view_876: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_155, [1, 128, 4096]);  mm_155 = None
        convert_element_type_1289: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_154, torch.float32);  mm_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_877: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_876, [1, 128, 16, 256]);  view_876 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_424: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_877, [0, 2, 1, 3]);  view_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_878: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_424, [16, 128, 256]);  permute_424 = None
        bmm_68: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_425, view_878);  permute_425 = None
        bmm_69: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_878, permute_426);  view_878 = permute_426 = None
        view_879: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_68, [1, 16, 128, 256]);  bmm_68 = None
        view_880: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_69, [1, 16, 128, 128]);  bmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1294: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_880, torch.float32);  view_880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_365: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1294, div_49);  convert_element_type_1294 = None
        sum_65: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_365, [-1], True)
        neg_67: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_49);  div_49 = None
        fma_3: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_67, sum_65, mul_365);  neg_67 = sum_65 = mul_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1295: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_69: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1295, 16.0);  convert_element_type_1295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_881: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_69, [16, 128, 128]);  div_69 = None
        bmm_70: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_427, view_881);  permute_427 = None
        bmm_71: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_881, permute_428);  view_881 = permute_428 = None
        view_882: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_70, [1, 16, 256, 128]);  bmm_70 = None
        view_883: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_71, [1, 16, 128, 256]);  bmm_71 = None
        convert_element_type_1301: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_882, torch.float32);  view_882 = None
        permute_429: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1301, [0, 1, 3, 2]);  convert_element_type_1301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1302: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_429, torch.bfloat16);  permute_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_24: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_883, torch.bfloat16);  view_883 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_430: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_24, [0, 2, 1, 3]);  convert_element_type_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_431: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1302, [0, 2, 1, 3]);  convert_element_type_1302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_240: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_430, 3, 0, 64)
        slice_241: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_430, 3, 64, 256);  permute_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_242: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_431, 3, 0, 64)
        slice_243: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_431, 3, 64, 256);  permute_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_193: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_324, [1, 128, 1, 32, 2]);  unsqueeze_324 = None
        clone_193: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_193, memory_format = torch.contiguous_format);  expand_193 = None
        view_681: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_193, [1, 128, 1, 64]);  clone_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_366: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_240, view_681)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_884: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_366, [1, 128, 16, 32, 2]);  mul_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_12: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_884, -1, 0)
        select_13: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_884, -1, 1);  view_884 = None
        neg_68: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_12);  select_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_24: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_68, 3, 1, 9223372036854775807, 2);  neg_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_25: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_13, 3, 0, 9223372036854775807, 2);  select_13 = None
        add_298: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_24, slice_scatter_25);  slice_scatter_24 = slice_scatter_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_194: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_326, [1, 128, 1, 32, 2]);  unsqueeze_326 = None
        clone_194: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_194, memory_format = torch.contiguous_format);  expand_194 = None
        view_682: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_194, [1, 128, 1, 64]);  clone_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_367: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_240, view_682);  slice_240 = None
        add_299: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_298, mul_367);  add_298 = mul_367 = None
        mul_368: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_242, view_681);  view_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_885: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_368, [1, 128, 16, 32, 2]);  mul_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_14: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_885, -1, 0)
        select_15: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_885, -1, 1);  view_885 = None
        neg_69: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_14);  select_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_26: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_69, 3, 1, 9223372036854775807, 2);  neg_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_27: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_15, 3, 0, 9223372036854775807, 2);  select_15 = None
        add_300: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_26, slice_scatter_27);  slice_scatter_26 = slice_scatter_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_369: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_242, view_682);  slice_242 = view_682 = None
        add_301: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_300, mul_369);  add_300 = mul_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_28: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_241, 3, 64, 9223372036854775807);  slice_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_29: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_299, 3, 0, 64);  add_299 = None
        add_302: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_28, slice_scatter_29);  slice_scatter_28 = slice_scatter_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_30: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_243, 3, 64, 9223372036854775807);  slice_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_31: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_301, 3, 0, 64);  add_301 = None
        add_303: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_30, slice_scatter_31);  slice_scatter_30 = slice_scatter_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_432: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_879, [0, 2, 1, 3]);  view_879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_230: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_432, memory_format = torch.contiguous_format);  permute_432 = None
        view_886: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_230, [1, 128, 4096]);  clone_230 = None
        view_887: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_303, [1, 128, 4096]);  add_303 = None
        view_888: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_302, [1, 128, 4096]);  add_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_889: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_886, [128, 4096]);  view_886 = None
        permute_433: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_889, [1, 0])
        mm_156: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_433, view_672);  permute_433 = None
        mm_157: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_889, permute_435);  view_889 = permute_435 = None
        view_890: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_157, [1, 128, 4096]);  mm_157 = None
        convert_element_type_1308: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_890, torch.float32);  view_890 = None
        add_304: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1282, convert_element_type_1308);  convert_element_type_1282 = convert_element_type_1308 = None
        convert_element_type_1309: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_156, torch.float32);  mm_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_891: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_887, [128, 4096]);  view_887 = None
        permute_437: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_891, [1, 0])
        mm_158: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_437, view_672);  permute_437 = None
        mm_159: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_891, permute_439);  view_891 = permute_439 = None
        view_892: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_159, [1, 128, 4096]);  mm_159 = None
        convert_element_type_1314: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_892, torch.float32);  view_892 = None
        add_305: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_304, convert_element_type_1314);  add_304 = convert_element_type_1314 = None
        convert_element_type_1315: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_158, torch.float32);  mm_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_893: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_888, [128, 4096]);  view_888 = None
        permute_441: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_893, [1, 0])
        mm_160: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_441, view_672);  permute_441 = view_672 = None
        mm_161: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_893, permute_443);  view_893 = permute_443 = None
        view_894: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_161, [1, 128, 4096]);  mm_161 = None
        convert_element_type_1320: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_894, torch.float32);  view_894 = None
        add_306: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_305, convert_element_type_1320);  add_305 = convert_element_type_1320 = None
        convert_element_type_1321: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_160, torch.float32);  mm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_371: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_306, primals_267);  primals_267 = None
        mul_372: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_371, 4096)
        sum_66: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True)
        mul_373: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_371, mul_240);  mul_371 = None
        sum_67: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_373, [2], True);  mul_373 = None
        mul_374: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_240, sum_67);  sum_67 = None
        sub_82: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_372, sum_66);  mul_372 = sum_66 = None
        sub_83: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_82, mul_374);  sub_82 = mul_374 = None
        mul_375: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_70, sub_83);  div_70 = sub_83 = None
        mul_376: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_306, mul_240);  mul_240 = None
        sum_68: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 1]);  mul_376 = None
        sum_69: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_306, [0, 1]);  add_306 = None
        add_307: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_295, mul_375);  add_295 = mul_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1322: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_307, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_895: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1322, [128, 4096]);  convert_element_type_1322 = None
        mm_162: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_895, permute_445);  permute_445 = None
        permute_446: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_895, [1, 0])
        mm_163: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_446, view_670);  view_670 = None
        sum_70: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_895, [0], True, dtype = torch.float32)
        view_896: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [4096]);  sum_70 = None
        convert_element_type_1327: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_896, torch.bfloat16);  view_896 = None
        view_897: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_162, [1, 128, 16384]);  mm_162 = None
        convert_element_type_1328: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_897, torch.float32);  view_897 = None
        convert_element_type_1329: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_163, torch.float32);  mm_163 = None
        convert_element_type_1330: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1327, torch.float32);  convert_element_type_1327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_669: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [1, 128, 16384]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_236: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_669, 0.5)
        mul_377: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1328, mul_236);  mul_236 = None
        convert_element_type_905: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_669, torch.float32)
        pow_24: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_905, 3.0)
        mul_237: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_215: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_669, mul_237);  view_669 = mul_237 = None
        mul_238: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_215, 0.7978845608028654);  add_215 = None
        tanh_23: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_238);  mul_238 = None
        add_216: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_23, 1.0)
        mul_378: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1328, add_216);  convert_element_type_1328 = add_216 = None
        convert_element_type_1331: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_378, torch.bfloat16);  mul_378 = None
        mul_379: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_23, tanh_23);  tanh_23 = None
        sub_84: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_379);  mul_379 = None
        mul_380: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_377, sub_84);  mul_377 = sub_84 = None
        mul_381: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_380, 0.7978845608028654);  mul_380 = None
        convert_element_type_1332: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_381, torch.bfloat16)
        mul_382: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, 0.044715);  mul_381 = None
        pow_33: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_905, 2.0);  convert_element_type_905 = None
        mul_383: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_33, 3.0);  pow_33 = None
        mul_384: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
        convert_element_type_1333: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_384, torch.bfloat16);  mul_384 = None
        add_308: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1332, convert_element_type_1333);  convert_element_type_1332 = convert_element_type_1333 = None
        mul_385: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1331, 0.5);  convert_element_type_1331 = None
        add_309: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_308, mul_385);  add_308 = mul_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_898: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_309, [128, 16384]);  add_309 = None
        mm_164: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_898, permute_449);  permute_449 = None
        permute_450: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_898, [1, 0])
        mm_165: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_450, view_644);  permute_450 = None
        sum_71: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_898, [0], True, dtype = torch.float32);  view_898 = None
        view_899: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_71, [16384]);  sum_71 = None
        convert_element_type_1338: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_899, torch.bfloat16);  view_899 = None
        view_900: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_164, [1, 128, 4096]);  mm_164 = None
        convert_element_type_1339: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_900, torch.float32);  view_900 = None
        convert_element_type_1340: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_165, torch.float32);  mm_165 = None
        convert_element_type_1341: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1338, torch.float32);  convert_element_type_1338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_166: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_446, view_666);  permute_446 = view_666 = None
        mm_167: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_895, permute_455);  view_895 = permute_455 = None
        view_902: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_167, [1, 128, 4096]);  mm_167 = None
        convert_element_type_1346: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_166, torch.float32);  mm_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_903: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_902, [1, 128, 16, 256]);  view_902 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_457: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_903, [0, 2, 1, 3]);  view_903 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_904: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_457, [16, 128, 256]);  permute_457 = None
        bmm_72: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_458, view_904);  permute_458 = None
        bmm_73: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_904, permute_459);  view_904 = permute_459 = None
        view_905: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_72, [1, 16, 128, 256]);  bmm_72 = None
        view_906: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_73, [1, 16, 128, 128]);  bmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1351: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_906, torch.float32);  view_906 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_386: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1351, div_47);  convert_element_type_1351 = None
        sum_72: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_386, [-1], True)
        neg_70: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_47);  div_47 = None
        fma_4: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_70, sum_72, mul_386);  neg_70 = sum_72 = mul_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1352: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_71: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1352, 16.0);  convert_element_type_1352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_907: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_71, [16, 128, 128]);  div_71 = None
        bmm_74: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_460, view_907);  permute_460 = None
        bmm_75: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_907, permute_461);  view_907 = permute_461 = None
        view_908: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_74, [1, 16, 256, 128]);  bmm_74 = None
        view_909: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_75, [1, 16, 128, 256]);  bmm_75 = None
        convert_element_type_1358: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_908, torch.float32);  view_908 = None
        permute_462: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1358, [0, 1, 3, 2]);  convert_element_type_1358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1359: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_462, torch.bfloat16);  permute_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_23: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_909, torch.bfloat16);  view_909 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_463: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_23, [0, 2, 1, 3]);  convert_element_type_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_464: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1359, [0, 2, 1, 3]);  convert_element_type_1359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_244: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_463, 3, 0, 64)
        slice_245: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_463, 3, 64, 256);  permute_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_246: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_464, 3, 0, 64)
        slice_247: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_464, 3, 64, 256);  permute_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_185: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_311, [1, 128, 1, 32, 2]);  unsqueeze_311 = None
        clone_185: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
        view_653: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_185, [1, 128, 1, 64]);  clone_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_387: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_244, view_653)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_910: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_387, [1, 128, 16, 32, 2]);  mul_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_16: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_910, -1, 0)
        select_17: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_910, -1, 1);  view_910 = None
        neg_71: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_16);  select_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_32: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_71, 3, 1, 9223372036854775807, 2);  neg_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_33: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_17, 3, 0, 9223372036854775807, 2);  select_17 = None
        add_310: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_32, slice_scatter_33);  slice_scatter_32 = slice_scatter_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_186: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_313, [1, 128, 1, 32, 2]);  unsqueeze_313 = None
        clone_186: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_186, memory_format = torch.contiguous_format);  expand_186 = None
        view_654: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_186, [1, 128, 1, 64]);  clone_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_388: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_244, view_654);  slice_244 = None
        add_311: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_310, mul_388);  add_310 = mul_388 = None
        mul_389: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_246, view_653);  view_653 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_911: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_389, [1, 128, 16, 32, 2]);  mul_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_18: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_911, -1, 0)
        select_19: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_911, -1, 1);  view_911 = None
        neg_72: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_18);  select_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_34: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_72, 3, 1, 9223372036854775807, 2);  neg_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_35: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_19, 3, 0, 9223372036854775807, 2);  select_19 = None
        add_312: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_34, slice_scatter_35);  slice_scatter_34 = slice_scatter_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_390: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_246, view_654);  slice_246 = view_654 = None
        add_313: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_312, mul_390);  add_312 = mul_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_36: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_245, 3, 64, 9223372036854775807);  slice_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_37: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_311, 3, 0, 64);  add_311 = None
        add_314: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_36, slice_scatter_37);  slice_scatter_36 = slice_scatter_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_38: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_247, 3, 64, 9223372036854775807);  slice_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_39: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_313, 3, 0, 64);  add_313 = None
        add_315: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_38, slice_scatter_39);  slice_scatter_38 = slice_scatter_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_465: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_905, [0, 2, 1, 3]);  view_905 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_231: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_465, memory_format = torch.contiguous_format);  permute_465 = None
        view_912: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_231, [1, 128, 4096]);  clone_231 = None
        view_913: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_315, [1, 128, 4096]);  add_315 = None
        view_914: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_314, [1, 128, 4096]);  add_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_915: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_912, [128, 4096]);  view_912 = None
        permute_466: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_915, [1, 0])
        mm_168: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_466, view_644);  permute_466 = None
        mm_169: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_915, permute_468);  view_915 = permute_468 = None
        view_916: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_169, [1, 128, 4096]);  mm_169 = None
        convert_element_type_1365: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_916, torch.float32);  view_916 = None
        add_316: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1339, convert_element_type_1365);  convert_element_type_1339 = convert_element_type_1365 = None
        convert_element_type_1366: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_168, torch.float32);  mm_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_917: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_913, [128, 4096]);  view_913 = None
        permute_470: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_917, [1, 0])
        mm_170: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_470, view_644);  permute_470 = None
        mm_171: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_917, permute_472);  view_917 = permute_472 = None
        view_918: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_171, [1, 128, 4096]);  mm_171 = None
        convert_element_type_1371: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_918, torch.float32);  view_918 = None
        add_317: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_316, convert_element_type_1371);  add_316 = convert_element_type_1371 = None
        convert_element_type_1372: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_170, torch.float32);  mm_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_919: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_914, [128, 4096]);  view_914 = None
        permute_474: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_919, [1, 0])
        mm_172: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_474, view_644);  permute_474 = view_644 = None
        mm_173: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_919, permute_476);  view_919 = permute_476 = None
        view_920: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_173, [1, 128, 4096]);  mm_173 = None
        convert_element_type_1377: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_920, torch.float32);  view_920 = None
        add_318: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_317, convert_element_type_1377);  add_317 = convert_element_type_1377 = None
        convert_element_type_1378: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_172, torch.float32);  mm_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_392: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_318, primals_256);  primals_256 = None
        mul_393: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_392, 4096)
        sum_73: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_392, [2], True)
        mul_394: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_392, mul_230);  mul_392 = None
        sum_74: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_394, [2], True);  mul_394 = None
        mul_395: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, sum_74);  sum_74 = None
        sub_86: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_393, sum_73);  mul_393 = sum_73 = None
        sub_87: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_86, mul_395);  sub_86 = mul_395 = None
        mul_396: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_72, sub_87);  div_72 = sub_87 = None
        mul_397: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_318, mul_230);  mul_230 = None
        sum_75: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_397, [0, 1]);  mul_397 = None
        sum_76: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_318, [0, 1]);  add_318 = None
        add_319: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_307, mul_396);  add_307 = mul_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1379: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_319, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_921: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1379, [128, 4096]);  convert_element_type_1379 = None
        mm_174: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_921, permute_478);  permute_478 = None
        permute_479: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_921, [1, 0])
        mm_175: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_479, view_642);  view_642 = None
        sum_77: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_921, [0], True, dtype = torch.float32)
        view_922: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_77, [4096]);  sum_77 = None
        convert_element_type_1384: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_922, torch.bfloat16);  view_922 = None
        view_923: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_174, [1, 128, 16384]);  mm_174 = None
        convert_element_type_1385: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_923, torch.float32);  view_923 = None
        convert_element_type_1386: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_175, torch.float32);  mm_175 = None
        convert_element_type_1387: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1384, torch.float32);  convert_element_type_1384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_641: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [1, 128, 16384]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_226: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_641, 0.5)
        mul_398: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1385, mul_226);  mul_226 = None
        convert_element_type_867: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_641, torch.float32)
        pow_23: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_867, 3.0)
        mul_227: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_206: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_641, mul_227);  view_641 = mul_227 = None
        mul_228: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_206, 0.7978845608028654);  add_206 = None
        tanh_22: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_228);  mul_228 = None
        add_207: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_22, 1.0)
        mul_399: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1385, add_207);  convert_element_type_1385 = add_207 = None
        convert_element_type_1388: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_399, torch.bfloat16);  mul_399 = None
        mul_400: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_22, tanh_22);  tanh_22 = None
        sub_88: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_400);  mul_400 = None
        mul_401: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_398, sub_88);  mul_398 = sub_88 = None
        mul_402: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_401, 0.7978845608028654);  mul_401 = None
        convert_element_type_1389: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_402, torch.bfloat16)
        mul_403: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, 0.044715);  mul_402 = None
        pow_34: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_867, 2.0);  convert_element_type_867 = None
        mul_404: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_34, 3.0);  pow_34 = None
        mul_405: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
        convert_element_type_1390: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_405, torch.bfloat16);  mul_405 = None
        add_320: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1389, convert_element_type_1390);  convert_element_type_1389 = convert_element_type_1390 = None
        mul_406: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1388, 0.5);  convert_element_type_1388 = None
        add_321: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_320, mul_406);  add_320 = mul_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_924: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_321, [128, 16384]);  add_321 = None
        mm_176: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_924, permute_482);  permute_482 = None
        permute_483: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_924, [1, 0])
        mm_177: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_483, view_616);  permute_483 = None
        sum_78: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_924, [0], True, dtype = torch.float32);  view_924 = None
        view_925: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_78, [16384]);  sum_78 = None
        convert_element_type_1395: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_925, torch.bfloat16);  view_925 = None
        view_926: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_176, [1, 128, 4096]);  mm_176 = None
        convert_element_type_1396: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_926, torch.float32);  view_926 = None
        convert_element_type_1397: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_177, torch.float32);  mm_177 = None
        convert_element_type_1398: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1395, torch.float32);  convert_element_type_1395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_178: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_479, view_638);  permute_479 = view_638 = None
        mm_179: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_921, permute_488);  view_921 = permute_488 = None
        view_928: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_179, [1, 128, 4096]);  mm_179 = None
        convert_element_type_1403: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_178, torch.float32);  mm_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_929: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_928, [1, 128, 16, 256]);  view_928 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_490: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_929, [0, 2, 1, 3]);  view_929 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_930: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_490, [16, 128, 256]);  permute_490 = None
        bmm_76: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_491, view_930);  permute_491 = None
        bmm_77: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_930, permute_492);  view_930 = permute_492 = None
        view_931: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_76, [1, 16, 128, 256]);  bmm_76 = None
        view_932: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_77, [1, 16, 128, 128]);  bmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1408: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_932, torch.float32);  view_932 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_407: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1408, div_45);  convert_element_type_1408 = None
        sum_79: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_407, [-1], True)
        neg_73: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_45);  div_45 = None
        fma_5: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_73, sum_79, mul_407);  neg_73 = sum_79 = mul_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1409: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_73: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1409, 16.0);  convert_element_type_1409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_933: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_73, [16, 128, 128]);  div_73 = None
        bmm_78: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_493, view_933);  permute_493 = None
        bmm_79: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_933, permute_494);  view_933 = permute_494 = None
        view_934: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_78, [1, 16, 256, 128]);  bmm_78 = None
        view_935: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_79, [1, 16, 128, 256]);  bmm_79 = None
        convert_element_type_1415: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_934, torch.float32);  view_934 = None
        permute_495: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1415, [0, 1, 3, 2]);  convert_element_type_1415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1416: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_495, torch.bfloat16);  permute_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_22: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_935, torch.bfloat16);  view_935 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_496: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_22, [0, 2, 1, 3]);  convert_element_type_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_497: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1416, [0, 2, 1, 3]);  convert_element_type_1416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_248: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_496, 3, 0, 64)
        slice_249: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_496, 3, 64, 256);  permute_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_250: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_497, 3, 0, 64)
        slice_251: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_497, 3, 64, 256);  permute_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_177: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_298, [1, 128, 1, 32, 2]);  unsqueeze_298 = None
        clone_177: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
        view_625: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_177, [1, 128, 1, 64]);  clone_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_408: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_248, view_625)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_936: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_408, [1, 128, 16, 32, 2]);  mul_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_20: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_936, -1, 0)
        select_21: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_936, -1, 1);  view_936 = None
        neg_74: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_20);  select_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_40: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_74, 3, 1, 9223372036854775807, 2);  neg_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_41: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_21, 3, 0, 9223372036854775807, 2);  select_21 = None
        add_322: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_40, slice_scatter_41);  slice_scatter_40 = slice_scatter_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_178: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_300, [1, 128, 1, 32, 2]);  unsqueeze_300 = None
        clone_178: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_178, memory_format = torch.contiguous_format);  expand_178 = None
        view_626: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_178, [1, 128, 1, 64]);  clone_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_409: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_248, view_626);  slice_248 = None
        add_323: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_322, mul_409);  add_322 = mul_409 = None
        mul_410: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_250, view_625);  view_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_937: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_410, [1, 128, 16, 32, 2]);  mul_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_22: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_937, -1, 0)
        select_23: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_937, -1, 1);  view_937 = None
        neg_75: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_22);  select_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_42: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_75, 3, 1, 9223372036854775807, 2);  neg_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_43: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_23, 3, 0, 9223372036854775807, 2);  select_23 = None
        add_324: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_42, slice_scatter_43);  slice_scatter_42 = slice_scatter_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_411: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_250, view_626);  slice_250 = view_626 = None
        add_325: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_324, mul_411);  add_324 = mul_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_44: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_249, 3, 64, 9223372036854775807);  slice_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_45: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_323, 3, 0, 64);  add_323 = None
        add_326: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_44, slice_scatter_45);  slice_scatter_44 = slice_scatter_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_46: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_251, 3, 64, 9223372036854775807);  slice_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_47: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_325, 3, 0, 64);  add_325 = None
        add_327: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_46, slice_scatter_47);  slice_scatter_46 = slice_scatter_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_498: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_931, [0, 2, 1, 3]);  view_931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_232: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_498, memory_format = torch.contiguous_format);  permute_498 = None
        view_938: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_232, [1, 128, 4096]);  clone_232 = None
        view_939: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_327, [1, 128, 4096]);  add_327 = None
        view_940: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_326, [1, 128, 4096]);  add_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_941: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_938, [128, 4096]);  view_938 = None
        permute_499: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_941, [1, 0])
        mm_180: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_499, view_616);  permute_499 = None
        mm_181: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_941, permute_501);  view_941 = permute_501 = None
        view_942: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_181, [1, 128, 4096]);  mm_181 = None
        convert_element_type_1422: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_942, torch.float32);  view_942 = None
        add_328: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1396, convert_element_type_1422);  convert_element_type_1396 = convert_element_type_1422 = None
        convert_element_type_1423: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_180, torch.float32);  mm_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_943: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_939, [128, 4096]);  view_939 = None
        permute_503: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_943, [1, 0])
        mm_182: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_503, view_616);  permute_503 = None
        mm_183: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_943, permute_505);  view_943 = permute_505 = None
        view_944: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_183, [1, 128, 4096]);  mm_183 = None
        convert_element_type_1428: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_944, torch.float32);  view_944 = None
        add_329: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_328, convert_element_type_1428);  add_328 = convert_element_type_1428 = None
        convert_element_type_1429: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_182, torch.float32);  mm_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_945: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_940, [128, 4096]);  view_940 = None
        permute_507: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_945, [1, 0])
        mm_184: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_507, view_616);  permute_507 = view_616 = None
        mm_185: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_945, permute_509);  view_945 = permute_509 = None
        view_946: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_185, [1, 128, 4096]);  mm_185 = None
        convert_element_type_1434: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_946, torch.float32);  view_946 = None
        add_330: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_329, convert_element_type_1434);  add_329 = convert_element_type_1434 = None
        convert_element_type_1435: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_184, torch.float32);  mm_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_413: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_330, primals_245);  primals_245 = None
        mul_414: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_413, 4096)
        sum_80: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True)
        mul_415: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_413, mul_220);  mul_413 = None
        sum_81: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_415, [2], True);  mul_415 = None
        mul_416: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_220, sum_81);  sum_81 = None
        sub_90: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_414, sum_80);  mul_414 = sum_80 = None
        sub_91: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, mul_416);  sub_90 = mul_416 = None
        mul_417: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_74, sub_91);  div_74 = sub_91 = None
        mul_418: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_330, mul_220);  mul_220 = None
        sum_82: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 1]);  mul_418 = None
        sum_83: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_330, [0, 1]);  add_330 = None
        add_331: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_319, mul_417);  add_319 = mul_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1436: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_331, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_947: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1436, [128, 4096]);  convert_element_type_1436 = None
        mm_186: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_947, permute_511);  permute_511 = None
        permute_512: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_947, [1, 0])
        mm_187: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_512, view_614);  view_614 = None
        sum_84: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_947, [0], True, dtype = torch.float32)
        view_948: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_84, [4096]);  sum_84 = None
        convert_element_type_1441: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_948, torch.bfloat16);  view_948 = None
        view_949: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_186, [1, 128, 16384]);  mm_186 = None
        convert_element_type_1442: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_949, torch.float32);  view_949 = None
        convert_element_type_1443: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_187, torch.float32);  mm_187 = None
        convert_element_type_1444: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1441, torch.float32);  convert_element_type_1441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_613: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [1, 128, 16384]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_216: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_613, 0.5)
        mul_419: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1442, mul_216);  mul_216 = None
        convert_element_type_829: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_613, torch.float32)
        pow_22: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_829, 3.0)
        mul_217: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_197: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_613, mul_217);  view_613 = mul_217 = None
        mul_218: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_197, 0.7978845608028654);  add_197 = None
        tanh_21: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_218);  mul_218 = None
        add_198: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_21, 1.0)
        mul_420: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1442, add_198);  convert_element_type_1442 = add_198 = None
        convert_element_type_1445: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_420, torch.bfloat16);  mul_420 = None
        mul_421: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_21, tanh_21);  tanh_21 = None
        sub_92: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_421);  mul_421 = None
        mul_422: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_419, sub_92);  mul_419 = sub_92 = None
        mul_423: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_422, 0.7978845608028654);  mul_422 = None
        convert_element_type_1446: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_423, torch.bfloat16)
        mul_424: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_423, 0.044715);  mul_423 = None
        pow_35: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_829, 2.0);  convert_element_type_829 = None
        mul_425: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_35, 3.0);  pow_35 = None
        mul_426: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
        convert_element_type_1447: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_426, torch.bfloat16);  mul_426 = None
        add_332: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1446, convert_element_type_1447);  convert_element_type_1446 = convert_element_type_1447 = None
        mul_427: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1445, 0.5);  convert_element_type_1445 = None
        add_333: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_332, mul_427);  add_332 = mul_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_950: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_333, [128, 16384]);  add_333 = None
        mm_188: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_950, permute_515);  permute_515 = None
        permute_516: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_950, [1, 0])
        mm_189: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_516, view_588);  permute_516 = None
        sum_85: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_950, [0], True, dtype = torch.float32);  view_950 = None
        view_951: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_85, [16384]);  sum_85 = None
        convert_element_type_1452: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_951, torch.bfloat16);  view_951 = None
        view_952: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_188, [1, 128, 4096]);  mm_188 = None
        convert_element_type_1453: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_952, torch.float32);  view_952 = None
        convert_element_type_1454: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_189, torch.float32);  mm_189 = None
        convert_element_type_1455: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1452, torch.float32);  convert_element_type_1452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_190: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_512, view_610);  permute_512 = view_610 = None
        mm_191: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_947, permute_521);  view_947 = permute_521 = None
        view_954: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_191, [1, 128, 4096]);  mm_191 = None
        convert_element_type_1460: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_190, torch.float32);  mm_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_955: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_954, [1, 128, 16, 256]);  view_954 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_523: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_955, [0, 2, 1, 3]);  view_955 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_956: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_523, [16, 128, 256]);  permute_523 = None
        bmm_80: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_524, view_956);  permute_524 = None
        bmm_81: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_956, permute_525);  view_956 = permute_525 = None
        view_957: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_80, [1, 16, 128, 256]);  bmm_80 = None
        view_958: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_81, [1, 16, 128, 128]);  bmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1465: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_958, torch.float32);  view_958 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_428: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1465, div_43);  convert_element_type_1465 = None
        sum_86: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_428, [-1], True)
        neg_76: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_43);  div_43 = None
        fma_6: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_76, sum_86, mul_428);  neg_76 = sum_86 = mul_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1466: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_75: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1466, 16.0);  convert_element_type_1466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_959: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_75, [16, 128, 128]);  div_75 = None
        bmm_82: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_526, view_959);  permute_526 = None
        bmm_83: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_959, permute_527);  view_959 = permute_527 = None
        view_960: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_82, [1, 16, 256, 128]);  bmm_82 = None
        view_961: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_83, [1, 16, 128, 256]);  bmm_83 = None
        convert_element_type_1472: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_960, torch.float32);  view_960 = None
        permute_528: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1472, [0, 1, 3, 2]);  convert_element_type_1472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1473: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_528, torch.bfloat16);  permute_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_21: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_961, torch.bfloat16);  view_961 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_529: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_21, [0, 2, 1, 3]);  convert_element_type_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_530: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1473, [0, 2, 1, 3]);  convert_element_type_1473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_252: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_529, 3, 0, 64)
        slice_253: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_529, 3, 64, 256);  permute_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_254: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_530, 3, 0, 64)
        slice_255: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_530, 3, 64, 256);  permute_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_169: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_285, [1, 128, 1, 32, 2]);  unsqueeze_285 = None
        clone_169: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
        view_597: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_169, [1, 128, 1, 64]);  clone_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_429: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_252, view_597)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_962: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_429, [1, 128, 16, 32, 2]);  mul_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_24: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_962, -1, 0)
        select_25: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_962, -1, 1);  view_962 = None
        neg_77: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_24);  select_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_48: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_77, 3, 1, 9223372036854775807, 2);  neg_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_49: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_25, 3, 0, 9223372036854775807, 2);  select_25 = None
        add_334: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_48, slice_scatter_49);  slice_scatter_48 = slice_scatter_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_170: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_287, [1, 128, 1, 32, 2]);  unsqueeze_287 = None
        clone_170: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_170, memory_format = torch.contiguous_format);  expand_170 = None
        view_598: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_170, [1, 128, 1, 64]);  clone_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_430: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_252, view_598);  slice_252 = None
        add_335: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_334, mul_430);  add_334 = mul_430 = None
        mul_431: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_254, view_597);  view_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_963: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_431, [1, 128, 16, 32, 2]);  mul_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_26: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_963, -1, 0)
        select_27: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_963, -1, 1);  view_963 = None
        neg_78: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_26);  select_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_50: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_78, 3, 1, 9223372036854775807, 2);  neg_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_51: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_27, 3, 0, 9223372036854775807, 2);  select_27 = None
        add_336: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_50, slice_scatter_51);  slice_scatter_50 = slice_scatter_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_432: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_254, view_598);  slice_254 = view_598 = None
        add_337: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_336, mul_432);  add_336 = mul_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_52: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_253, 3, 64, 9223372036854775807);  slice_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_53: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_335, 3, 0, 64);  add_335 = None
        add_338: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_52, slice_scatter_53);  slice_scatter_52 = slice_scatter_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_54: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_255, 3, 64, 9223372036854775807);  slice_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_55: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_337, 3, 0, 64);  add_337 = None
        add_339: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_54, slice_scatter_55);  slice_scatter_54 = slice_scatter_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_531: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_957, [0, 2, 1, 3]);  view_957 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_233: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_531, memory_format = torch.contiguous_format);  permute_531 = None
        view_964: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_233, [1, 128, 4096]);  clone_233 = None
        view_965: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_339, [1, 128, 4096]);  add_339 = None
        view_966: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_338, [1, 128, 4096]);  add_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_967: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_964, [128, 4096]);  view_964 = None
        permute_532: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_967, [1, 0])
        mm_192: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_532, view_588);  permute_532 = None
        mm_193: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_967, permute_534);  view_967 = permute_534 = None
        view_968: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_193, [1, 128, 4096]);  mm_193 = None
        convert_element_type_1479: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_968, torch.float32);  view_968 = None
        add_340: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1453, convert_element_type_1479);  convert_element_type_1453 = convert_element_type_1479 = None
        convert_element_type_1480: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_192, torch.float32);  mm_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_969: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_965, [128, 4096]);  view_965 = None
        permute_536: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_969, [1, 0])
        mm_194: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_536, view_588);  permute_536 = None
        mm_195: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_969, permute_538);  view_969 = permute_538 = None
        view_970: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_195, [1, 128, 4096]);  mm_195 = None
        convert_element_type_1485: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_970, torch.float32);  view_970 = None
        add_341: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_340, convert_element_type_1485);  add_340 = convert_element_type_1485 = None
        convert_element_type_1486: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_194, torch.float32);  mm_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_971: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_966, [128, 4096]);  view_966 = None
        permute_540: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_971, [1, 0])
        mm_196: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_540, view_588);  permute_540 = view_588 = None
        mm_197: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_971, permute_542);  view_971 = permute_542 = None
        view_972: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_197, [1, 128, 4096]);  mm_197 = None
        convert_element_type_1491: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_972, torch.float32);  view_972 = None
        add_342: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_341, convert_element_type_1491);  add_341 = convert_element_type_1491 = None
        convert_element_type_1492: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_196, torch.float32);  mm_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_434: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_342, primals_234);  primals_234 = None
        mul_435: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_434, 4096)
        sum_87: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_434, [2], True)
        mul_436: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_434, mul_210);  mul_434 = None
        sum_88: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_436, [2], True);  mul_436 = None
        mul_437: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, sum_88);  sum_88 = None
        sub_94: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_435, sum_87);  mul_435 = sum_87 = None
        sub_95: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_94, mul_437);  sub_94 = mul_437 = None
        mul_438: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_76, sub_95);  div_76 = sub_95 = None
        mul_439: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_342, mul_210);  mul_210 = None
        sum_89: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_439, [0, 1]);  mul_439 = None
        sum_90: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_342, [0, 1]);  add_342 = None
        add_343: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_331, mul_438);  add_331 = mul_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1493: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_343, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_973: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1493, [128, 4096]);  convert_element_type_1493 = None
        mm_198: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_973, permute_544);  permute_544 = None
        permute_545: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_973, [1, 0])
        mm_199: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_545, view_586);  view_586 = None
        sum_91: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_973, [0], True, dtype = torch.float32)
        view_974: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_91, [4096]);  sum_91 = None
        convert_element_type_1498: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_974, torch.bfloat16);  view_974 = None
        view_975: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_198, [1, 128, 16384]);  mm_198 = None
        convert_element_type_1499: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_975, torch.float32);  view_975 = None
        convert_element_type_1500: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_199, torch.float32);  mm_199 = None
        convert_element_type_1501: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1498, torch.float32);  convert_element_type_1498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_585: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [1, 128, 16384]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_206: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_585, 0.5)
        mul_440: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1499, mul_206);  mul_206 = None
        convert_element_type_791: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_585, torch.float32)
        pow_21: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_791, 3.0)
        mul_207: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_188: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_585, mul_207);  view_585 = mul_207 = None
        mul_208: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_188, 0.7978845608028654);  add_188 = None
        tanh_20: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_208);  mul_208 = None
        add_189: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_20, 1.0)
        mul_441: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1499, add_189);  convert_element_type_1499 = add_189 = None
        convert_element_type_1502: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_441, torch.bfloat16);  mul_441 = None
        mul_442: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_20, tanh_20);  tanh_20 = None
        sub_96: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_442);  mul_442 = None
        mul_443: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_440, sub_96);  mul_440 = sub_96 = None
        mul_444: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_443, 0.7978845608028654);  mul_443 = None
        convert_element_type_1503: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_444, torch.bfloat16)
        mul_445: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_444, 0.044715);  mul_444 = None
        pow_36: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_791, 2.0);  convert_element_type_791 = None
        mul_446: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_36, 3.0);  pow_36 = None
        mul_447: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
        convert_element_type_1504: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_447, torch.bfloat16);  mul_447 = None
        add_344: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1503, convert_element_type_1504);  convert_element_type_1503 = convert_element_type_1504 = None
        mul_448: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1502, 0.5);  convert_element_type_1502 = None
        add_345: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_344, mul_448);  add_344 = mul_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_976: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_345, [128, 16384]);  add_345 = None
        mm_200: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_976, permute_548);  permute_548 = None
        permute_549: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_976, [1, 0])
        mm_201: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_549, view_560);  permute_549 = None
        sum_92: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_976, [0], True, dtype = torch.float32);  view_976 = None
        view_977: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_92, [16384]);  sum_92 = None
        convert_element_type_1509: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_977, torch.bfloat16);  view_977 = None
        view_978: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_200, [1, 128, 4096]);  mm_200 = None
        convert_element_type_1510: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_978, torch.float32);  view_978 = None
        convert_element_type_1511: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_201, torch.float32);  mm_201 = None
        convert_element_type_1512: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1509, torch.float32);  convert_element_type_1509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_202: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_545, view_582);  permute_545 = view_582 = None
        mm_203: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_973, permute_554);  view_973 = permute_554 = None
        view_980: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_203, [1, 128, 4096]);  mm_203 = None
        convert_element_type_1517: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_202, torch.float32);  mm_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_981: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_980, [1, 128, 16, 256]);  view_980 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_556: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_981, [0, 2, 1, 3]);  view_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_982: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_556, [16, 128, 256]);  permute_556 = None
        bmm_84: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_557, view_982);  permute_557 = None
        bmm_85: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_982, permute_558);  view_982 = permute_558 = None
        view_983: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_84, [1, 16, 128, 256]);  bmm_84 = None
        view_984: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_85, [1, 16, 128, 128]);  bmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1522: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_984, torch.float32);  view_984 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_449: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1522, div_41);  convert_element_type_1522 = None
        sum_93: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_449, [-1], True)
        neg_79: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_41);  div_41 = None
        fma_7: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_79, sum_93, mul_449);  neg_79 = sum_93 = mul_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1523: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_77: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1523, 16.0);  convert_element_type_1523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_985: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_77, [16, 128, 128]);  div_77 = None
        bmm_86: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_559, view_985);  permute_559 = None
        bmm_87: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_985, permute_560);  view_985 = permute_560 = None
        view_986: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_86, [1, 16, 256, 128]);  bmm_86 = None
        view_987: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_87, [1, 16, 128, 256]);  bmm_87 = None
        convert_element_type_1529: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_986, torch.float32);  view_986 = None
        permute_561: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1529, [0, 1, 3, 2]);  convert_element_type_1529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1530: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_561, torch.bfloat16);  permute_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_20: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_987, torch.bfloat16);  view_987 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_562: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_20, [0, 2, 1, 3]);  convert_element_type_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_563: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1530, [0, 2, 1, 3]);  convert_element_type_1530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_256: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_562, 3, 0, 64)
        slice_257: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_562, 3, 64, 256);  permute_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_258: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_563, 3, 0, 64)
        slice_259: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_563, 3, 64, 256);  permute_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_161: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_272, [1, 128, 1, 32, 2]);  unsqueeze_272 = None
        clone_161: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
        view_569: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_161, [1, 128, 1, 64]);  clone_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_450: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_256, view_569)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_988: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_450, [1, 128, 16, 32, 2]);  mul_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_28: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_988, -1, 0)
        select_29: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_988, -1, 1);  view_988 = None
        neg_80: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_28);  select_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_56: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_80, 3, 1, 9223372036854775807, 2);  neg_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_57: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_29, 3, 0, 9223372036854775807, 2);  select_29 = None
        add_346: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_56, slice_scatter_57);  slice_scatter_56 = slice_scatter_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_162: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_274, [1, 128, 1, 32, 2]);  unsqueeze_274 = None
        clone_162: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_162, memory_format = torch.contiguous_format);  expand_162 = None
        view_570: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_162, [1, 128, 1, 64]);  clone_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_451: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_256, view_570);  slice_256 = None
        add_347: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_346, mul_451);  add_346 = mul_451 = None
        mul_452: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_258, view_569);  view_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_989: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_452, [1, 128, 16, 32, 2]);  mul_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_30: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_989, -1, 0)
        select_31: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_989, -1, 1);  view_989 = None
        neg_81: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_30);  select_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_58: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_81, 3, 1, 9223372036854775807, 2);  neg_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_59: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_31, 3, 0, 9223372036854775807, 2);  select_31 = None
        add_348: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_58, slice_scatter_59);  slice_scatter_58 = slice_scatter_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_453: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_258, view_570);  slice_258 = view_570 = None
        add_349: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_348, mul_453);  add_348 = mul_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_60: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_257, 3, 64, 9223372036854775807);  slice_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_61: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_347, 3, 0, 64);  add_347 = None
        add_350: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_60, slice_scatter_61);  slice_scatter_60 = slice_scatter_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_62: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_259, 3, 64, 9223372036854775807);  slice_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_63: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_349, 3, 0, 64);  add_349 = None
        add_351: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_62, slice_scatter_63);  slice_scatter_62 = slice_scatter_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_564: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_983, [0, 2, 1, 3]);  view_983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_234: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_564, memory_format = torch.contiguous_format);  permute_564 = None
        view_990: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_234, [1, 128, 4096]);  clone_234 = None
        view_991: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_351, [1, 128, 4096]);  add_351 = None
        view_992: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_350, [1, 128, 4096]);  add_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_993: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_990, [128, 4096]);  view_990 = None
        permute_565: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_993, [1, 0])
        mm_204: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_565, view_560);  permute_565 = None
        mm_205: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_993, permute_567);  view_993 = permute_567 = None
        view_994: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_205, [1, 128, 4096]);  mm_205 = None
        convert_element_type_1536: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_994, torch.float32);  view_994 = None
        add_352: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1510, convert_element_type_1536);  convert_element_type_1510 = convert_element_type_1536 = None
        convert_element_type_1537: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_204, torch.float32);  mm_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_995: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_991, [128, 4096]);  view_991 = None
        permute_569: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_995, [1, 0])
        mm_206: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_569, view_560);  permute_569 = None
        mm_207: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_995, permute_571);  view_995 = permute_571 = None
        view_996: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_207, [1, 128, 4096]);  mm_207 = None
        convert_element_type_1542: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_996, torch.float32);  view_996 = None
        add_353: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_352, convert_element_type_1542);  add_352 = convert_element_type_1542 = None
        convert_element_type_1543: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_206, torch.float32);  mm_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_997: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_992, [128, 4096]);  view_992 = None
        permute_573: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_997, [1, 0])
        mm_208: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_573, view_560);  permute_573 = view_560 = None
        mm_209: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_997, permute_575);  view_997 = permute_575 = None
        view_998: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_209, [1, 128, 4096]);  mm_209 = None
        convert_element_type_1548: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_998, torch.float32);  view_998 = None
        add_354: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_353, convert_element_type_1548);  add_353 = convert_element_type_1548 = None
        convert_element_type_1549: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_208, torch.float32);  mm_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_455: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_354, primals_223);  primals_223 = None
        mul_456: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_455, 4096)
        sum_94: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_455, [2], True)
        mul_457: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_455, mul_200);  mul_455 = None
        sum_95: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_457, [2], True);  mul_457 = None
        mul_458: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_200, sum_95);  sum_95 = None
        sub_98: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_456, sum_94);  mul_456 = sum_94 = None
        sub_99: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_98, mul_458);  sub_98 = mul_458 = None
        mul_459: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_78, sub_99);  div_78 = sub_99 = None
        mul_460: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_354, mul_200);  mul_200 = None
        sum_96: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_460, [0, 1]);  mul_460 = None
        sum_97: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_354, [0, 1]);  add_354 = None
        add_355: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_343, mul_459);  add_343 = mul_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1550: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_355, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_999: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1550, [128, 4096]);  convert_element_type_1550 = None
        mm_210: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_999, permute_577);  permute_577 = None
        permute_578: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_999, [1, 0])
        mm_211: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_578, view_558);  view_558 = None
        sum_98: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_999, [0], True, dtype = torch.float32)
        view_1000: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_98, [4096]);  sum_98 = None
        convert_element_type_1555: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1000, torch.bfloat16);  view_1000 = None
        view_1001: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_210, [1, 128, 16384]);  mm_210 = None
        convert_element_type_1556: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1001, torch.float32);  view_1001 = None
        convert_element_type_1557: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_211, torch.float32);  mm_211 = None
        convert_element_type_1558: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1555, torch.float32);  convert_element_type_1555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_557: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [1, 128, 16384]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_196: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_557, 0.5)
        mul_461: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1556, mul_196);  mul_196 = None
        convert_element_type_753: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_557, torch.float32)
        pow_20: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_753, 3.0)
        mul_197: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_179: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_557, mul_197);  view_557 = mul_197 = None
        mul_198: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_179, 0.7978845608028654);  add_179 = None
        tanh_19: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_198);  mul_198 = None
        add_180: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_19, 1.0)
        mul_462: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1556, add_180);  convert_element_type_1556 = add_180 = None
        convert_element_type_1559: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_462, torch.bfloat16);  mul_462 = None
        mul_463: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_19, tanh_19);  tanh_19 = None
        sub_100: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_463);  mul_463 = None
        mul_464: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_461, sub_100);  mul_461 = sub_100 = None
        mul_465: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_464, 0.7978845608028654);  mul_464 = None
        convert_element_type_1560: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_465, torch.bfloat16)
        mul_466: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_465, 0.044715);  mul_465 = None
        pow_37: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_753, 2.0);  convert_element_type_753 = None
        mul_467: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_37, 3.0);  pow_37 = None
        mul_468: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
        convert_element_type_1561: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_468, torch.bfloat16);  mul_468 = None
        add_356: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1560, convert_element_type_1561);  convert_element_type_1560 = convert_element_type_1561 = None
        mul_469: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1559, 0.5);  convert_element_type_1559 = None
        add_357: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_356, mul_469);  add_356 = mul_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1002: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_357, [128, 16384]);  add_357 = None
        mm_212: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1002, permute_581);  permute_581 = None
        permute_582: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1002, [1, 0])
        mm_213: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_582, view_532);  permute_582 = None
        sum_99: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1002, [0], True, dtype = torch.float32);  view_1002 = None
        view_1003: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_99, [16384]);  sum_99 = None
        convert_element_type_1566: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1003, torch.bfloat16);  view_1003 = None
        view_1004: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_212, [1, 128, 4096]);  mm_212 = None
        convert_element_type_1567: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1004, torch.float32);  view_1004 = None
        convert_element_type_1568: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_213, torch.float32);  mm_213 = None
        convert_element_type_1569: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1566, torch.float32);  convert_element_type_1566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_214: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_578, view_554);  permute_578 = view_554 = None
        mm_215: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_999, permute_587);  view_999 = permute_587 = None
        view_1006: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_215, [1, 128, 4096]);  mm_215 = None
        convert_element_type_1574: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_214, torch.float32);  mm_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1007: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1006, [1, 128, 16, 256]);  view_1006 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_589: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1007, [0, 2, 1, 3]);  view_1007 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1008: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_589, [16, 128, 256]);  permute_589 = None
        bmm_88: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_590, view_1008);  permute_590 = None
        bmm_89: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1008, permute_591);  view_1008 = permute_591 = None
        view_1009: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_88, [1, 16, 128, 256]);  bmm_88 = None
        view_1010: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_89, [1, 16, 128, 128]);  bmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1579: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1010, torch.float32);  view_1010 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_470: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1579, div_39);  convert_element_type_1579 = None
        sum_100: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_470, [-1], True)
        neg_82: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_39);  div_39 = None
        fma_8: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_82, sum_100, mul_470);  neg_82 = sum_100 = mul_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1580: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_79: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1580, 16.0);  convert_element_type_1580 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1011: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_79, [16, 128, 128]);  div_79 = None
        bmm_90: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_592, view_1011);  permute_592 = None
        bmm_91: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1011, permute_593);  view_1011 = permute_593 = None
        view_1012: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_90, [1, 16, 256, 128]);  bmm_90 = None
        view_1013: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_91, [1, 16, 128, 256]);  bmm_91 = None
        convert_element_type_1586: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1012, torch.float32);  view_1012 = None
        permute_594: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1586, [0, 1, 3, 2]);  convert_element_type_1586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1587: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_594, torch.bfloat16);  permute_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_19: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1013, torch.bfloat16);  view_1013 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_595: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_19, [0, 2, 1, 3]);  convert_element_type_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_596: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1587, [0, 2, 1, 3]);  convert_element_type_1587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_260: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_595, 3, 0, 64)
        slice_261: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_595, 3, 64, 256);  permute_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_262: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_596, 3, 0, 64)
        slice_263: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_596, 3, 64, 256);  permute_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_153: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_259, [1, 128, 1, 32, 2]);  unsqueeze_259 = None
        clone_153: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
        view_541: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_153, [1, 128, 1, 64]);  clone_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_471: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_260, view_541)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1014: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_471, [1, 128, 16, 32, 2]);  mul_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_32: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1014, -1, 0)
        select_33: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1014, -1, 1);  view_1014 = None
        neg_83: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_32);  select_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_64: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_83, 3, 1, 9223372036854775807, 2);  neg_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_65: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_33, 3, 0, 9223372036854775807, 2);  select_33 = None
        add_358: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_64, slice_scatter_65);  slice_scatter_64 = slice_scatter_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_154: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_261, [1, 128, 1, 32, 2]);  unsqueeze_261 = None
        clone_154: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_154, memory_format = torch.contiguous_format);  expand_154 = None
        view_542: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_154, [1, 128, 1, 64]);  clone_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_472: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_260, view_542);  slice_260 = None
        add_359: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_358, mul_472);  add_358 = mul_472 = None
        mul_473: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_262, view_541);  view_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1015: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_473, [1, 128, 16, 32, 2]);  mul_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_34: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1015, -1, 0)
        select_35: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1015, -1, 1);  view_1015 = None
        neg_84: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_34);  select_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_66: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_84, 3, 1, 9223372036854775807, 2);  neg_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_67: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_35, 3, 0, 9223372036854775807, 2);  select_35 = None
        add_360: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_66, slice_scatter_67);  slice_scatter_66 = slice_scatter_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_474: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_262, view_542);  slice_262 = view_542 = None
        add_361: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_360, mul_474);  add_360 = mul_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_68: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_261, 3, 64, 9223372036854775807);  slice_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_69: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_359, 3, 0, 64);  add_359 = None
        add_362: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_68, slice_scatter_69);  slice_scatter_68 = slice_scatter_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_70: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_263, 3, 64, 9223372036854775807);  slice_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_71: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_361, 3, 0, 64);  add_361 = None
        add_363: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_70, slice_scatter_71);  slice_scatter_70 = slice_scatter_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_597: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1009, [0, 2, 1, 3]);  view_1009 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_235: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_597, memory_format = torch.contiguous_format);  permute_597 = None
        view_1016: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_235, [1, 128, 4096]);  clone_235 = None
        view_1017: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_363, [1, 128, 4096]);  add_363 = None
        view_1018: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_362, [1, 128, 4096]);  add_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1019: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1016, [128, 4096]);  view_1016 = None
        permute_598: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1019, [1, 0])
        mm_216: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_598, view_532);  permute_598 = None
        mm_217: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1019, permute_600);  view_1019 = permute_600 = None
        view_1020: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_217, [1, 128, 4096]);  mm_217 = None
        convert_element_type_1593: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1020, torch.float32);  view_1020 = None
        add_364: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1567, convert_element_type_1593);  convert_element_type_1567 = convert_element_type_1593 = None
        convert_element_type_1594: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_216, torch.float32);  mm_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1021: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1017, [128, 4096]);  view_1017 = None
        permute_602: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1021, [1, 0])
        mm_218: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_602, view_532);  permute_602 = None
        mm_219: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1021, permute_604);  view_1021 = permute_604 = None
        view_1022: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_219, [1, 128, 4096]);  mm_219 = None
        convert_element_type_1599: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1022, torch.float32);  view_1022 = None
        add_365: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_364, convert_element_type_1599);  add_364 = convert_element_type_1599 = None
        convert_element_type_1600: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_218, torch.float32);  mm_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1023: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1018, [128, 4096]);  view_1018 = None
        permute_606: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1023, [1, 0])
        mm_220: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_606, view_532);  permute_606 = view_532 = None
        mm_221: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1023, permute_608);  view_1023 = permute_608 = None
        view_1024: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_221, [1, 128, 4096]);  mm_221 = None
        convert_element_type_1605: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1024, torch.float32);  view_1024 = None
        add_366: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_365, convert_element_type_1605);  add_365 = convert_element_type_1605 = None
        convert_element_type_1606: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_220, torch.float32);  mm_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_476: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_366, primals_212);  primals_212 = None
        mul_477: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_476, 4096)
        sum_101: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True)
        mul_478: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_476, mul_190);  mul_476 = None
        sum_102: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_478, [2], True);  mul_478 = None
        mul_479: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_190, sum_102);  sum_102 = None
        sub_102: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_477, sum_101);  mul_477 = sum_101 = None
        sub_103: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_102, mul_479);  sub_102 = mul_479 = None
        mul_480: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_80, sub_103);  div_80 = sub_103 = None
        mul_481: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_366, mul_190);  mul_190 = None
        sum_103: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 1]);  mul_481 = None
        sum_104: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_366, [0, 1]);  add_366 = None
        add_367: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_355, mul_480);  add_355 = mul_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1607: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_367, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1025: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1607, [128, 4096]);  convert_element_type_1607 = None
        mm_222: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1025, permute_610);  permute_610 = None
        permute_611: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1025, [1, 0])
        mm_223: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_611, view_530);  view_530 = None
        sum_105: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1025, [0], True, dtype = torch.float32)
        view_1026: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_105, [4096]);  sum_105 = None
        convert_element_type_1612: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1026, torch.bfloat16);  view_1026 = None
        view_1027: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_222, [1, 128, 16384]);  mm_222 = None
        convert_element_type_1613: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1027, torch.float32);  view_1027 = None
        convert_element_type_1614: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_223, torch.float32);  mm_223 = None
        convert_element_type_1615: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1612, torch.float32);  convert_element_type_1612 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_529: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [1, 128, 16384]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_186: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_529, 0.5)
        mul_482: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1613, mul_186);  mul_186 = None
        convert_element_type_715: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_529, torch.float32)
        pow_19: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_715, 3.0)
        mul_187: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_170: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_529, mul_187);  view_529 = mul_187 = None
        mul_188: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_170, 0.7978845608028654);  add_170 = None
        tanh_18: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_188);  mul_188 = None
        add_171: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_18, 1.0)
        mul_483: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1613, add_171);  convert_element_type_1613 = add_171 = None
        convert_element_type_1616: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_483, torch.bfloat16);  mul_483 = None
        mul_484: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_18, tanh_18);  tanh_18 = None
        sub_104: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_484);  mul_484 = None
        mul_485: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_482, sub_104);  mul_482 = sub_104 = None
        mul_486: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_485, 0.7978845608028654);  mul_485 = None
        convert_element_type_1617: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_486, torch.bfloat16)
        mul_487: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_486, 0.044715);  mul_486 = None
        pow_38: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_715, 2.0);  convert_element_type_715 = None
        mul_488: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_38, 3.0);  pow_38 = None
        mul_489: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_487, mul_488);  mul_487 = mul_488 = None
        convert_element_type_1618: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_489, torch.bfloat16);  mul_489 = None
        add_368: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1617, convert_element_type_1618);  convert_element_type_1617 = convert_element_type_1618 = None
        mul_490: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1616, 0.5);  convert_element_type_1616 = None
        add_369: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_368, mul_490);  add_368 = mul_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1028: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_369, [128, 16384]);  add_369 = None
        mm_224: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1028, permute_614);  permute_614 = None
        permute_615: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1028, [1, 0])
        mm_225: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_615, view_504);  permute_615 = None
        sum_106: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1028, [0], True, dtype = torch.float32);  view_1028 = None
        view_1029: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_106, [16384]);  sum_106 = None
        convert_element_type_1623: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1029, torch.bfloat16);  view_1029 = None
        view_1030: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_224, [1, 128, 4096]);  mm_224 = None
        convert_element_type_1624: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1030, torch.float32);  view_1030 = None
        convert_element_type_1625: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_225, torch.float32);  mm_225 = None
        convert_element_type_1626: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1623, torch.float32);  convert_element_type_1623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_226: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_611, view_526);  permute_611 = view_526 = None
        mm_227: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1025, permute_620);  view_1025 = permute_620 = None
        view_1032: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_227, [1, 128, 4096]);  mm_227 = None
        convert_element_type_1631: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_226, torch.float32);  mm_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1033: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1032, [1, 128, 16, 256]);  view_1032 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_622: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1033, [0, 2, 1, 3]);  view_1033 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1034: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_622, [16, 128, 256]);  permute_622 = None
        bmm_92: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_623, view_1034);  permute_623 = None
        bmm_93: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1034, permute_624);  view_1034 = permute_624 = None
        view_1035: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_92, [1, 16, 128, 256]);  bmm_92 = None
        view_1036: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_93, [1, 16, 128, 128]);  bmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1636: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1036, torch.float32);  view_1036 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_491: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1636, div_37);  convert_element_type_1636 = None
        sum_107: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_491, [-1], True)
        neg_85: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_37);  div_37 = None
        fma_9: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_85, sum_107, mul_491);  neg_85 = sum_107 = mul_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1637: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_81: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1637, 16.0);  convert_element_type_1637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1037: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_81, [16, 128, 128]);  div_81 = None
        bmm_94: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_625, view_1037);  permute_625 = None
        bmm_95: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1037, permute_626);  view_1037 = permute_626 = None
        view_1038: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_94, [1, 16, 256, 128]);  bmm_94 = None
        view_1039: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_95, [1, 16, 128, 256]);  bmm_95 = None
        convert_element_type_1643: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1038, torch.float32);  view_1038 = None
        permute_627: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1643, [0, 1, 3, 2]);  convert_element_type_1643 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1644: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_627, torch.bfloat16);  permute_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_18: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1039, torch.bfloat16);  view_1039 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_628: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_18, [0, 2, 1, 3]);  convert_element_type_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_629: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1644, [0, 2, 1, 3]);  convert_element_type_1644 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_264: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_628, 3, 0, 64)
        slice_265: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_628, 3, 64, 256);  permute_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_266: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_629, 3, 0, 64)
        slice_267: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_629, 3, 64, 256);  permute_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_145: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_246, [1, 128, 1, 32, 2]);  unsqueeze_246 = None
        clone_145: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
        view_513: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_145, [1, 128, 1, 64]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_492: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_264, view_513)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1040: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_492, [1, 128, 16, 32, 2]);  mul_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_36: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1040, -1, 0)
        select_37: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1040, -1, 1);  view_1040 = None
        neg_86: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_36);  select_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_72: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_86, 3, 1, 9223372036854775807, 2);  neg_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_73: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_37, 3, 0, 9223372036854775807, 2);  select_37 = None
        add_370: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_72, slice_scatter_73);  slice_scatter_72 = slice_scatter_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_146: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_248, [1, 128, 1, 32, 2]);  unsqueeze_248 = None
        clone_146: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_146, memory_format = torch.contiguous_format);  expand_146 = None
        view_514: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_146, [1, 128, 1, 64]);  clone_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_493: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_264, view_514);  slice_264 = None
        add_371: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_370, mul_493);  add_370 = mul_493 = None
        mul_494: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_266, view_513);  view_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1041: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_494, [1, 128, 16, 32, 2]);  mul_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_38: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1041, -1, 0)
        select_39: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1041, -1, 1);  view_1041 = None
        neg_87: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_38);  select_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_74: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_87, 3, 1, 9223372036854775807, 2);  neg_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_75: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_39, 3, 0, 9223372036854775807, 2);  select_39 = None
        add_372: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_74, slice_scatter_75);  slice_scatter_74 = slice_scatter_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_495: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_266, view_514);  slice_266 = view_514 = None
        add_373: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_372, mul_495);  add_372 = mul_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_76: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_265, 3, 64, 9223372036854775807);  slice_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_77: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_371, 3, 0, 64);  add_371 = None
        add_374: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_76, slice_scatter_77);  slice_scatter_76 = slice_scatter_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_78: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_267, 3, 64, 9223372036854775807);  slice_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_79: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_373, 3, 0, 64);  add_373 = None
        add_375: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_78, slice_scatter_79);  slice_scatter_78 = slice_scatter_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_630: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1035, [0, 2, 1, 3]);  view_1035 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_236: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_630, memory_format = torch.contiguous_format);  permute_630 = None
        view_1042: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_236, [1, 128, 4096]);  clone_236 = None
        view_1043: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_375, [1, 128, 4096]);  add_375 = None
        view_1044: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_374, [1, 128, 4096]);  add_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1045: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1042, [128, 4096]);  view_1042 = None
        permute_631: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1045, [1, 0])
        mm_228: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_631, view_504);  permute_631 = None
        mm_229: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1045, permute_633);  view_1045 = permute_633 = None
        view_1046: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_229, [1, 128, 4096]);  mm_229 = None
        convert_element_type_1650: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1046, torch.float32);  view_1046 = None
        add_376: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1624, convert_element_type_1650);  convert_element_type_1624 = convert_element_type_1650 = None
        convert_element_type_1651: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_228, torch.float32);  mm_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1047: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1043, [128, 4096]);  view_1043 = None
        permute_635: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1047, [1, 0])
        mm_230: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_635, view_504);  permute_635 = None
        mm_231: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1047, permute_637);  view_1047 = permute_637 = None
        view_1048: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_231, [1, 128, 4096]);  mm_231 = None
        convert_element_type_1656: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1048, torch.float32);  view_1048 = None
        add_377: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_376, convert_element_type_1656);  add_376 = convert_element_type_1656 = None
        convert_element_type_1657: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_230, torch.float32);  mm_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1049: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1044, [128, 4096]);  view_1044 = None
        permute_639: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1049, [1, 0])
        mm_232: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_639, view_504);  permute_639 = view_504 = None
        mm_233: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1049, permute_641);  view_1049 = permute_641 = None
        view_1050: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_233, [1, 128, 4096]);  mm_233 = None
        convert_element_type_1662: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1050, torch.float32);  view_1050 = None
        add_378: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_377, convert_element_type_1662);  add_377 = convert_element_type_1662 = None
        convert_element_type_1663: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_232, torch.float32);  mm_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_497: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_378, primals_201);  primals_201 = None
        mul_498: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_497, 4096)
        sum_108: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True)
        mul_499: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_497, mul_180);  mul_497 = None
        sum_109: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_499, [2], True);  mul_499 = None
        mul_500: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_180, sum_109);  sum_109 = None
        sub_106: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_498, sum_108);  mul_498 = sum_108 = None
        sub_107: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_106, mul_500);  sub_106 = mul_500 = None
        mul_501: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_82, sub_107);  div_82 = sub_107 = None
        mul_502: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_378, mul_180);  mul_180 = None
        sum_110: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_502, [0, 1]);  mul_502 = None
        sum_111: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_378, [0, 1]);  add_378 = None
        add_379: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_367, mul_501);  add_367 = mul_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1664: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_379, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1051: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1664, [128, 4096]);  convert_element_type_1664 = None
        mm_234: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1051, permute_643);  permute_643 = None
        permute_644: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1051, [1, 0])
        mm_235: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_644, view_502);  view_502 = None
        sum_112: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1051, [0], True, dtype = torch.float32)
        view_1052: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_112, [4096]);  sum_112 = None
        convert_element_type_1669: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1052, torch.bfloat16);  view_1052 = None
        view_1053: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_234, [1, 128, 16384]);  mm_234 = None
        convert_element_type_1670: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1053, torch.float32);  view_1053 = None
        convert_element_type_1671: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_235, torch.float32);  mm_235 = None
        convert_element_type_1672: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1669, torch.float32);  convert_element_type_1669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_501: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [1, 128, 16384]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_176: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_501, 0.5)
        mul_503: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1670, mul_176);  mul_176 = None
        convert_element_type_677: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_501, torch.float32)
        pow_18: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_677, 3.0)
        mul_177: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_161: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_501, mul_177);  view_501 = mul_177 = None
        mul_178: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_161, 0.7978845608028654);  add_161 = None
        tanh_17: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_178);  mul_178 = None
        add_162: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_17, 1.0)
        mul_504: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1670, add_162);  convert_element_type_1670 = add_162 = None
        convert_element_type_1673: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_504, torch.bfloat16);  mul_504 = None
        mul_505: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_17, tanh_17);  tanh_17 = None
        sub_108: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_505);  mul_505 = None
        mul_506: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_503, sub_108);  mul_503 = sub_108 = None
        mul_507: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_506, 0.7978845608028654);  mul_506 = None
        convert_element_type_1674: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_507, torch.bfloat16)
        mul_508: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_507, 0.044715);  mul_507 = None
        pow_39: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_677, 2.0);  convert_element_type_677 = None
        mul_509: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_39, 3.0);  pow_39 = None
        mul_510: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
        convert_element_type_1675: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_510, torch.bfloat16);  mul_510 = None
        add_380: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1674, convert_element_type_1675);  convert_element_type_1674 = convert_element_type_1675 = None
        mul_511: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1673, 0.5);  convert_element_type_1673 = None
        add_381: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_380, mul_511);  add_380 = mul_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1054: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_381, [128, 16384]);  add_381 = None
        mm_236: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1054, permute_647);  permute_647 = None
        permute_648: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1054, [1, 0])
        mm_237: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_648, view_476);  permute_648 = None
        sum_113: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1054, [0], True, dtype = torch.float32);  view_1054 = None
        view_1055: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_113, [16384]);  sum_113 = None
        convert_element_type_1680: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1055, torch.bfloat16);  view_1055 = None
        view_1056: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_236, [1, 128, 4096]);  mm_236 = None
        convert_element_type_1681: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1056, torch.float32);  view_1056 = None
        convert_element_type_1682: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_237, torch.float32);  mm_237 = None
        convert_element_type_1683: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1680, torch.float32);  convert_element_type_1680 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_238: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_644, view_498);  permute_644 = view_498 = None
        mm_239: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1051, permute_653);  view_1051 = permute_653 = None
        view_1058: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_239, [1, 128, 4096]);  mm_239 = None
        convert_element_type_1688: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_238, torch.float32);  mm_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1059: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1058, [1, 128, 16, 256]);  view_1058 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_655: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1059, [0, 2, 1, 3]);  view_1059 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1060: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_655, [16, 128, 256]);  permute_655 = None
        bmm_96: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_656, view_1060);  permute_656 = None
        bmm_97: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1060, permute_657);  view_1060 = permute_657 = None
        view_1061: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_96, [1, 16, 128, 256]);  bmm_96 = None
        view_1062: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_97, [1, 16, 128, 128]);  bmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1693: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1062, torch.float32);  view_1062 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_512: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1693, div_35);  convert_element_type_1693 = None
        sum_114: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_512, [-1], True)
        neg_88: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_35);  div_35 = None
        fma_10: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_88, sum_114, mul_512);  neg_88 = sum_114 = mul_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1694: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_83: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1694, 16.0);  convert_element_type_1694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1063: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_83, [16, 128, 128]);  div_83 = None
        bmm_98: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_658, view_1063);  permute_658 = None
        bmm_99: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1063, permute_659);  view_1063 = permute_659 = None
        view_1064: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_98, [1, 16, 256, 128]);  bmm_98 = None
        view_1065: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_99, [1, 16, 128, 256]);  bmm_99 = None
        convert_element_type_1700: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1064, torch.float32);  view_1064 = None
        permute_660: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1700, [0, 1, 3, 2]);  convert_element_type_1700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1701: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_660, torch.bfloat16);  permute_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_17: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1065, torch.bfloat16);  view_1065 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_661: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_17, [0, 2, 1, 3]);  convert_element_type_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_662: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1701, [0, 2, 1, 3]);  convert_element_type_1701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_268: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_661, 3, 0, 64)
        slice_269: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_661, 3, 64, 256);  permute_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_270: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_662, 3, 0, 64)
        slice_271: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_662, 3, 64, 256);  permute_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_137: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_233, [1, 128, 1, 32, 2]);  unsqueeze_233 = None
        clone_137: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
        view_485: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_137, [1, 128, 1, 64]);  clone_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_513: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_268, view_485)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1066: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_513, [1, 128, 16, 32, 2]);  mul_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_40: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1066, -1, 0)
        select_41: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1066, -1, 1);  view_1066 = None
        neg_89: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_40);  select_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_80: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_89, 3, 1, 9223372036854775807, 2);  neg_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_81: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_41, 3, 0, 9223372036854775807, 2);  select_41 = None
        add_382: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_80, slice_scatter_81);  slice_scatter_80 = slice_scatter_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_138: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_235, [1, 128, 1, 32, 2]);  unsqueeze_235 = None
        clone_138: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_138, memory_format = torch.contiguous_format);  expand_138 = None
        view_486: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_138, [1, 128, 1, 64]);  clone_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_514: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_268, view_486);  slice_268 = None
        add_383: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_382, mul_514);  add_382 = mul_514 = None
        mul_515: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_270, view_485);  view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1067: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_515, [1, 128, 16, 32, 2]);  mul_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_42: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1067, -1, 0)
        select_43: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1067, -1, 1);  view_1067 = None
        neg_90: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_42);  select_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_82: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_90, 3, 1, 9223372036854775807, 2);  neg_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_83: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_43, 3, 0, 9223372036854775807, 2);  select_43 = None
        add_384: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_82, slice_scatter_83);  slice_scatter_82 = slice_scatter_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_516: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_270, view_486);  slice_270 = view_486 = None
        add_385: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_384, mul_516);  add_384 = mul_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_84: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_269, 3, 64, 9223372036854775807);  slice_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_85: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_383, 3, 0, 64);  add_383 = None
        add_386: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_84, slice_scatter_85);  slice_scatter_84 = slice_scatter_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_86: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_271, 3, 64, 9223372036854775807);  slice_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_87: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_385, 3, 0, 64);  add_385 = None
        add_387: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_86, slice_scatter_87);  slice_scatter_86 = slice_scatter_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_663: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1061, [0, 2, 1, 3]);  view_1061 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_237: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_663, memory_format = torch.contiguous_format);  permute_663 = None
        view_1068: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_237, [1, 128, 4096]);  clone_237 = None
        view_1069: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_387, [1, 128, 4096]);  add_387 = None
        view_1070: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_386, [1, 128, 4096]);  add_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1071: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1068, [128, 4096]);  view_1068 = None
        permute_664: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1071, [1, 0])
        mm_240: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_664, view_476);  permute_664 = None
        mm_241: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1071, permute_666);  view_1071 = permute_666 = None
        view_1072: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_241, [1, 128, 4096]);  mm_241 = None
        convert_element_type_1707: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1072, torch.float32);  view_1072 = None
        add_388: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1681, convert_element_type_1707);  convert_element_type_1681 = convert_element_type_1707 = None
        convert_element_type_1708: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_240, torch.float32);  mm_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1073: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1069, [128, 4096]);  view_1069 = None
        permute_668: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1073, [1, 0])
        mm_242: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_668, view_476);  permute_668 = None
        mm_243: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1073, permute_670);  view_1073 = permute_670 = None
        view_1074: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_243, [1, 128, 4096]);  mm_243 = None
        convert_element_type_1713: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1074, torch.float32);  view_1074 = None
        add_389: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_388, convert_element_type_1713);  add_388 = convert_element_type_1713 = None
        convert_element_type_1714: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_242, torch.float32);  mm_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1075: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1070, [128, 4096]);  view_1070 = None
        permute_672: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1075, [1, 0])
        mm_244: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_672, view_476);  permute_672 = view_476 = None
        mm_245: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1075, permute_674);  view_1075 = permute_674 = None
        view_1076: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_245, [1, 128, 4096]);  mm_245 = None
        convert_element_type_1719: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1076, torch.float32);  view_1076 = None
        add_390: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_389, convert_element_type_1719);  add_389 = convert_element_type_1719 = None
        convert_element_type_1720: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_244, torch.float32);  mm_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_518: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_390, primals_190);  primals_190 = None
        mul_519: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_518, 4096)
        sum_115: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_518, [2], True)
        mul_520: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_518, mul_170);  mul_518 = None
        sum_116: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_520, [2], True);  mul_520 = None
        mul_521: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_170, sum_116);  sum_116 = None
        sub_110: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_519, sum_115);  mul_519 = sum_115 = None
        sub_111: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_110, mul_521);  sub_110 = mul_521 = None
        mul_522: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_84, sub_111);  div_84 = sub_111 = None
        mul_523: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_390, mul_170);  mul_170 = None
        sum_117: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_523, [0, 1]);  mul_523 = None
        sum_118: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_390, [0, 1]);  add_390 = None
        add_391: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_379, mul_522);  add_379 = mul_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1721: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_391, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1077: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1721, [128, 4096]);  convert_element_type_1721 = None
        mm_246: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1077, permute_676);  permute_676 = None
        permute_677: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1077, [1, 0])
        mm_247: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_677, view_474);  view_474 = None
        sum_119: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1077, [0], True, dtype = torch.float32)
        view_1078: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_119, [4096]);  sum_119 = None
        convert_element_type_1726: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1078, torch.bfloat16);  view_1078 = None
        view_1079: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_246, [1, 128, 16384]);  mm_246 = None
        convert_element_type_1727: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1079, torch.float32);  view_1079 = None
        convert_element_type_1728: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_247, torch.float32);  mm_247 = None
        convert_element_type_1729: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1726, torch.float32);  convert_element_type_1726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_473: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [1, 128, 16384]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_166: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_473, 0.5)
        mul_524: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1727, mul_166);  mul_166 = None
        convert_element_type_639: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_473, torch.float32)
        pow_17: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_639, 3.0)
        mul_167: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_152: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_473, mul_167);  view_473 = mul_167 = None
        mul_168: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_152, 0.7978845608028654);  add_152 = None
        tanh_16: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_168);  mul_168 = None
        add_153: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_16, 1.0)
        mul_525: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1727, add_153);  convert_element_type_1727 = add_153 = None
        convert_element_type_1730: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_525, torch.bfloat16);  mul_525 = None
        mul_526: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_16, tanh_16);  tanh_16 = None
        sub_112: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_526);  mul_526 = None
        mul_527: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_524, sub_112);  mul_524 = sub_112 = None
        mul_528: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_527, 0.7978845608028654);  mul_527 = None
        convert_element_type_1731: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_528, torch.bfloat16)
        mul_529: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_528, 0.044715);  mul_528 = None
        pow_40: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_639, 2.0);  convert_element_type_639 = None
        mul_530: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_40, 3.0);  pow_40 = None
        mul_531: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
        convert_element_type_1732: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_531, torch.bfloat16);  mul_531 = None
        add_392: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1731, convert_element_type_1732);  convert_element_type_1731 = convert_element_type_1732 = None
        mul_532: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1730, 0.5);  convert_element_type_1730 = None
        add_393: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_392, mul_532);  add_392 = mul_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1080: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_393, [128, 16384]);  add_393 = None
        mm_248: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1080, permute_680);  permute_680 = None
        permute_681: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1080, [1, 0])
        mm_249: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_681, view_448);  permute_681 = None
        sum_120: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1080, [0], True, dtype = torch.float32);  view_1080 = None
        view_1081: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_120, [16384]);  sum_120 = None
        convert_element_type_1737: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1081, torch.bfloat16);  view_1081 = None
        view_1082: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_248, [1, 128, 4096]);  mm_248 = None
        convert_element_type_1738: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1082, torch.float32);  view_1082 = None
        convert_element_type_1739: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_249, torch.float32);  mm_249 = None
        convert_element_type_1740: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1737, torch.float32);  convert_element_type_1737 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_250: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_677, view_470);  permute_677 = view_470 = None
        mm_251: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1077, permute_686);  view_1077 = permute_686 = None
        view_1084: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_251, [1, 128, 4096]);  mm_251 = None
        convert_element_type_1745: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_250, torch.float32);  mm_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1085: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1084, [1, 128, 16, 256]);  view_1084 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_688: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1085, [0, 2, 1, 3]);  view_1085 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1086: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_688, [16, 128, 256]);  permute_688 = None
        bmm_100: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_689, view_1086);  permute_689 = None
        bmm_101: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1086, permute_690);  view_1086 = permute_690 = None
        view_1087: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_100, [1, 16, 128, 256]);  bmm_100 = None
        view_1088: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_101, [1, 16, 128, 128]);  bmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1750: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1088, torch.float32);  view_1088 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_533: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1750, div_33);  convert_element_type_1750 = None
        sum_121: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_533, [-1], True)
        neg_91: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_33);  div_33 = None
        fma_11: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_91, sum_121, mul_533);  neg_91 = sum_121 = mul_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1751: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_85: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1751, 16.0);  convert_element_type_1751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1089: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_85, [16, 128, 128]);  div_85 = None
        bmm_102: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_691, view_1089);  permute_691 = None
        bmm_103: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1089, permute_692);  view_1089 = permute_692 = None
        view_1090: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_102, [1, 16, 256, 128]);  bmm_102 = None
        view_1091: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_103, [1, 16, 128, 256]);  bmm_103 = None
        convert_element_type_1757: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1090, torch.float32);  view_1090 = None
        permute_693: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1757, [0, 1, 3, 2]);  convert_element_type_1757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1758: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_693, torch.bfloat16);  permute_693 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_16: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1091, torch.bfloat16);  view_1091 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_694: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_16, [0, 2, 1, 3]);  convert_element_type_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_695: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1758, [0, 2, 1, 3]);  convert_element_type_1758 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_272: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_694, 3, 0, 64)
        slice_273: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_694, 3, 64, 256);  permute_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_274: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_695, 3, 0, 64)
        slice_275: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_695, 3, 64, 256);  permute_695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_129: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_220, [1, 128, 1, 32, 2]);  unsqueeze_220 = None
        clone_129: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
        view_457: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [1, 128, 1, 64]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_534: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_272, view_457)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1092: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_534, [1, 128, 16, 32, 2]);  mul_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_44: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1092, -1, 0)
        select_45: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1092, -1, 1);  view_1092 = None
        neg_92: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_44);  select_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_88: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_92, 3, 1, 9223372036854775807, 2);  neg_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_89: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_45, 3, 0, 9223372036854775807, 2);  select_45 = None
        add_394: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_88, slice_scatter_89);  slice_scatter_88 = slice_scatter_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_130: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_222, [1, 128, 1, 32, 2]);  unsqueeze_222 = None
        clone_130: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_130, memory_format = torch.contiguous_format);  expand_130 = None
        view_458: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [1, 128, 1, 64]);  clone_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_535: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_272, view_458);  slice_272 = None
        add_395: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_394, mul_535);  add_394 = mul_535 = None
        mul_536: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_274, view_457);  view_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1093: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_536, [1, 128, 16, 32, 2]);  mul_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_46: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1093, -1, 0)
        select_47: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1093, -1, 1);  view_1093 = None
        neg_93: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_46);  select_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_90: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_93, 3, 1, 9223372036854775807, 2);  neg_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_91: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_47, 3, 0, 9223372036854775807, 2);  select_47 = None
        add_396: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_90, slice_scatter_91);  slice_scatter_90 = slice_scatter_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_537: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_274, view_458);  slice_274 = view_458 = None
        add_397: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_396, mul_537);  add_396 = mul_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_92: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_273, 3, 64, 9223372036854775807);  slice_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_93: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_395, 3, 0, 64);  add_395 = None
        add_398: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_92, slice_scatter_93);  slice_scatter_92 = slice_scatter_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_94: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_275, 3, 64, 9223372036854775807);  slice_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_95: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_397, 3, 0, 64);  add_397 = None
        add_399: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_94, slice_scatter_95);  slice_scatter_94 = slice_scatter_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_696: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1087, [0, 2, 1, 3]);  view_1087 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_238: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_696, memory_format = torch.contiguous_format);  permute_696 = None
        view_1094: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_238, [1, 128, 4096]);  clone_238 = None
        view_1095: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_399, [1, 128, 4096]);  add_399 = None
        view_1096: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_398, [1, 128, 4096]);  add_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1097: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1094, [128, 4096]);  view_1094 = None
        permute_697: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1097, [1, 0])
        mm_252: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_697, view_448);  permute_697 = None
        mm_253: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1097, permute_699);  view_1097 = permute_699 = None
        view_1098: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_253, [1, 128, 4096]);  mm_253 = None
        convert_element_type_1764: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1098, torch.float32);  view_1098 = None
        add_400: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1738, convert_element_type_1764);  convert_element_type_1738 = convert_element_type_1764 = None
        convert_element_type_1765: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_252, torch.float32);  mm_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1099: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1095, [128, 4096]);  view_1095 = None
        permute_701: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1099, [1, 0])
        mm_254: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_701, view_448);  permute_701 = None
        mm_255: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1099, permute_703);  view_1099 = permute_703 = None
        view_1100: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_255, [1, 128, 4096]);  mm_255 = None
        convert_element_type_1770: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1100, torch.float32);  view_1100 = None
        add_401: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_400, convert_element_type_1770);  add_400 = convert_element_type_1770 = None
        convert_element_type_1771: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_254, torch.float32);  mm_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1101: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1096, [128, 4096]);  view_1096 = None
        permute_705: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1101, [1, 0])
        mm_256: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_705, view_448);  permute_705 = view_448 = None
        mm_257: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1101, permute_707);  view_1101 = permute_707 = None
        view_1102: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_257, [1, 128, 4096]);  mm_257 = None
        convert_element_type_1776: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1102, torch.float32);  view_1102 = None
        add_402: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_401, convert_element_type_1776);  add_401 = convert_element_type_1776 = None
        convert_element_type_1777: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_256, torch.float32);  mm_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_539: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_402, primals_179);  primals_179 = None
        mul_540: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_539, 4096)
        sum_122: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_539, [2], True)
        mul_541: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_539, mul_160);  mul_539 = None
        sum_123: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_541, [2], True);  mul_541 = None
        mul_542: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, sum_123);  sum_123 = None
        sub_114: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_540, sum_122);  mul_540 = sum_122 = None
        sub_115: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_114, mul_542);  sub_114 = mul_542 = None
        mul_543: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_86, sub_115);  div_86 = sub_115 = None
        mul_544: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_402, mul_160);  mul_160 = None
        sum_124: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_544, [0, 1]);  mul_544 = None
        sum_125: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_402, [0, 1]);  add_402 = None
        add_403: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_391, mul_543);  add_391 = mul_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1778: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_403, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1103: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1778, [128, 4096]);  convert_element_type_1778 = None
        mm_258: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1103, permute_709);  permute_709 = None
        permute_710: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1103, [1, 0])
        mm_259: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_710, view_446);  view_446 = None
        sum_126: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1103, [0], True, dtype = torch.float32)
        view_1104: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_126, [4096]);  sum_126 = None
        convert_element_type_1783: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1104, torch.bfloat16);  view_1104 = None
        view_1105: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_258, [1, 128, 16384]);  mm_258 = None
        convert_element_type_1784: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1105, torch.float32);  view_1105 = None
        convert_element_type_1785: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_259, torch.float32);  mm_259 = None
        convert_element_type_1786: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1783, torch.float32);  convert_element_type_1783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_445: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [1, 128, 16384]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_156: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_445, 0.5)
        mul_545: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1784, mul_156);  mul_156 = None
        convert_element_type_601: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_445, torch.float32)
        pow_16: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_601, 3.0)
        mul_157: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_143: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_445, mul_157);  view_445 = mul_157 = None
        mul_158: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_143, 0.7978845608028654);  add_143 = None
        tanh_15: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
        add_144: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
        mul_546: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1784, add_144);  convert_element_type_1784 = add_144 = None
        convert_element_type_1787: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_546, torch.bfloat16);  mul_546 = None
        mul_547: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_15, tanh_15);  tanh_15 = None
        sub_116: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_547);  mul_547 = None
        mul_548: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_545, sub_116);  mul_545 = sub_116 = None
        mul_549: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_548, 0.7978845608028654);  mul_548 = None
        convert_element_type_1788: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_549, torch.bfloat16)
        mul_550: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_549, 0.044715);  mul_549 = None
        pow_41: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_601, 2.0);  convert_element_type_601 = None
        mul_551: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_41, 3.0);  pow_41 = None
        mul_552: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_550, mul_551);  mul_550 = mul_551 = None
        convert_element_type_1789: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_552, torch.bfloat16);  mul_552 = None
        add_404: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1788, convert_element_type_1789);  convert_element_type_1788 = convert_element_type_1789 = None
        mul_553: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1787, 0.5);  convert_element_type_1787 = None
        add_405: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_404, mul_553);  add_404 = mul_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1106: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_405, [128, 16384]);  add_405 = None
        mm_260: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1106, permute_713);  permute_713 = None
        permute_714: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1106, [1, 0])
        mm_261: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_714, view_420);  permute_714 = None
        sum_127: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1106, [0], True, dtype = torch.float32);  view_1106 = None
        view_1107: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_127, [16384]);  sum_127 = None
        convert_element_type_1794: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1107, torch.bfloat16);  view_1107 = None
        view_1108: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_260, [1, 128, 4096]);  mm_260 = None
        convert_element_type_1795: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1108, torch.float32);  view_1108 = None
        convert_element_type_1796: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_261, torch.float32);  mm_261 = None
        convert_element_type_1797: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1794, torch.float32);  convert_element_type_1794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_262: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_710, view_442);  permute_710 = view_442 = None
        mm_263: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1103, permute_719);  view_1103 = permute_719 = None
        view_1110: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_263, [1, 128, 4096]);  mm_263 = None
        convert_element_type_1802: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_262, torch.float32);  mm_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1111: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1110, [1, 128, 16, 256]);  view_1110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_721: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1111, [0, 2, 1, 3]);  view_1111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1112: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_721, [16, 128, 256]);  permute_721 = None
        bmm_104: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_722, view_1112);  permute_722 = None
        bmm_105: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1112, permute_723);  view_1112 = permute_723 = None
        view_1113: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_104, [1, 16, 128, 256]);  bmm_104 = None
        view_1114: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_105, [1, 16, 128, 128]);  bmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1807: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1114, torch.float32);  view_1114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_554: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1807, div_31);  convert_element_type_1807 = None
        sum_128: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_554, [-1], True)
        neg_94: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_31);  div_31 = None
        fma_12: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_94, sum_128, mul_554);  neg_94 = sum_128 = mul_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1808: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_12, torch.bfloat16);  fma_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_87: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1808, 16.0);  convert_element_type_1808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1115: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_87, [16, 128, 128]);  div_87 = None
        bmm_106: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_724, view_1115);  permute_724 = None
        bmm_107: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1115, permute_725);  view_1115 = permute_725 = None
        view_1116: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_106, [1, 16, 256, 128]);  bmm_106 = None
        view_1117: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_107, [1, 16, 128, 256]);  bmm_107 = None
        convert_element_type_1814: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1116, torch.float32);  view_1116 = None
        permute_726: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1814, [0, 1, 3, 2]);  convert_element_type_1814 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1815: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_726, torch.bfloat16);  permute_726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_15: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1117, torch.bfloat16);  view_1117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_727: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_15, [0, 2, 1, 3]);  convert_element_type_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_728: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1815, [0, 2, 1, 3]);  convert_element_type_1815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_276: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_727, 3, 0, 64)
        slice_277: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_727, 3, 64, 256);  permute_727 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_278: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_728, 3, 0, 64)
        slice_279: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_728, 3, 64, 256);  permute_728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_121: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_207, [1, 128, 1, 32, 2]);  unsqueeze_207 = None
        clone_121: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
        view_429: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_121, [1, 128, 1, 64]);  clone_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_555: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_276, view_429)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1118: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_555, [1, 128, 16, 32, 2]);  mul_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_48: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1118, -1, 0)
        select_49: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1118, -1, 1);  view_1118 = None
        neg_95: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_48);  select_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_96: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_95, 3, 1, 9223372036854775807, 2);  neg_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_97: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_49, 3, 0, 9223372036854775807, 2);  select_49 = None
        add_406: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_96, slice_scatter_97);  slice_scatter_96 = slice_scatter_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_122: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_209, [1, 128, 1, 32, 2]);  unsqueeze_209 = None
        clone_122: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
        view_430: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_122, [1, 128, 1, 64]);  clone_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_556: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_276, view_430);  slice_276 = None
        add_407: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_406, mul_556);  add_406 = mul_556 = None
        mul_557: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_278, view_429);  view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1119: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_557, [1, 128, 16, 32, 2]);  mul_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_50: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1119, -1, 0)
        select_51: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1119, -1, 1);  view_1119 = None
        neg_96: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_50);  select_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_98: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_96, 3, 1, 9223372036854775807, 2);  neg_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_99: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_51, 3, 0, 9223372036854775807, 2);  select_51 = None
        add_408: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_98, slice_scatter_99);  slice_scatter_98 = slice_scatter_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_558: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_278, view_430);  slice_278 = view_430 = None
        add_409: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_408, mul_558);  add_408 = mul_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_100: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_277, 3, 64, 9223372036854775807);  slice_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_101: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_407, 3, 0, 64);  add_407 = None
        add_410: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_100, slice_scatter_101);  slice_scatter_100 = slice_scatter_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_102: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_279, 3, 64, 9223372036854775807);  slice_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_103: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_409, 3, 0, 64);  add_409 = None
        add_411: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_102, slice_scatter_103);  slice_scatter_102 = slice_scatter_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_729: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1113, [0, 2, 1, 3]);  view_1113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_239: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_729, memory_format = torch.contiguous_format);  permute_729 = None
        view_1120: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_239, [1, 128, 4096]);  clone_239 = None
        view_1121: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_411, [1, 128, 4096]);  add_411 = None
        view_1122: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_410, [1, 128, 4096]);  add_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1123: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1120, [128, 4096]);  view_1120 = None
        permute_730: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1123, [1, 0])
        mm_264: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_730, view_420);  permute_730 = None
        mm_265: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1123, permute_732);  view_1123 = permute_732 = None
        view_1124: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_265, [1, 128, 4096]);  mm_265 = None
        convert_element_type_1821: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1124, torch.float32);  view_1124 = None
        add_412: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1795, convert_element_type_1821);  convert_element_type_1795 = convert_element_type_1821 = None
        convert_element_type_1822: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_264, torch.float32);  mm_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1125: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1121, [128, 4096]);  view_1121 = None
        permute_734: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1125, [1, 0])
        mm_266: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_734, view_420);  permute_734 = None
        mm_267: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1125, permute_736);  view_1125 = permute_736 = None
        view_1126: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_267, [1, 128, 4096]);  mm_267 = None
        convert_element_type_1827: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1126, torch.float32);  view_1126 = None
        add_413: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_412, convert_element_type_1827);  add_412 = convert_element_type_1827 = None
        convert_element_type_1828: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_266, torch.float32);  mm_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1127: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1122, [128, 4096]);  view_1122 = None
        permute_738: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1127, [1, 0])
        mm_268: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_738, view_420);  permute_738 = view_420 = None
        mm_269: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1127, permute_740);  view_1127 = permute_740 = None
        view_1128: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_269, [1, 128, 4096]);  mm_269 = None
        convert_element_type_1833: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1128, torch.float32);  view_1128 = None
        add_414: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_413, convert_element_type_1833);  add_413 = convert_element_type_1833 = None
        convert_element_type_1834: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_268, torch.float32);  mm_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_560: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_414, primals_168);  primals_168 = None
        mul_561: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_560, 4096)
        sum_129: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_560, [2], True)
        mul_562: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_560, mul_150);  mul_560 = None
        sum_130: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_562, [2], True);  mul_562 = None
        mul_563: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, sum_130);  sum_130 = None
        sub_118: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_561, sum_129);  mul_561 = sum_129 = None
        sub_119: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_118, mul_563);  sub_118 = mul_563 = None
        mul_564: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_88, sub_119);  div_88 = sub_119 = None
        mul_565: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_414, mul_150);  mul_150 = None
        sum_131: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_565, [0, 1]);  mul_565 = None
        sum_132: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_414, [0, 1]);  add_414 = None
        add_415: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_403, mul_564);  add_403 = mul_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1835: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_415, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1129: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1835, [128, 4096]);  convert_element_type_1835 = None
        mm_270: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1129, permute_742);  permute_742 = None
        permute_743: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1129, [1, 0])
        mm_271: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_743, view_418);  view_418 = None
        sum_133: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1129, [0], True, dtype = torch.float32)
        view_1130: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_133, [4096]);  sum_133 = None
        convert_element_type_1840: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1130, torch.bfloat16);  view_1130 = None
        view_1131: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_270, [1, 128, 16384]);  mm_270 = None
        convert_element_type_1841: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1131, torch.float32);  view_1131 = None
        convert_element_type_1842: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_271, torch.float32);  mm_271 = None
        convert_element_type_1843: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1840, torch.float32);  convert_element_type_1840 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_417: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [1, 128, 16384]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_146: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_417, 0.5)
        mul_566: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1841, mul_146);  mul_146 = None
        convert_element_type_563: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_417, torch.float32)
        pow_15: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_563, 3.0)
        mul_147: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_134: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_417, mul_147);  view_417 = mul_147 = None
        mul_148: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_134, 0.7978845608028654);  add_134 = None
        tanh_14: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_148);  mul_148 = None
        add_135: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
        mul_567: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1841, add_135);  convert_element_type_1841 = add_135 = None
        convert_element_type_1844: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_567, torch.bfloat16);  mul_567 = None
        mul_568: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_14, tanh_14);  tanh_14 = None
        sub_120: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_568);  mul_568 = None
        mul_569: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_566, sub_120);  mul_566 = sub_120 = None
        mul_570: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_569, 0.7978845608028654);  mul_569 = None
        convert_element_type_1845: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_570, torch.bfloat16)
        mul_571: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_570, 0.044715);  mul_570 = None
        pow_42: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_563, 2.0);  convert_element_type_563 = None
        mul_572: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_42, 3.0);  pow_42 = None
        mul_573: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
        convert_element_type_1846: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_573, torch.bfloat16);  mul_573 = None
        add_416: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1845, convert_element_type_1846);  convert_element_type_1845 = convert_element_type_1846 = None
        mul_574: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1844, 0.5);  convert_element_type_1844 = None
        add_417: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_416, mul_574);  add_416 = mul_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1132: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_417, [128, 16384]);  add_417 = None
        mm_272: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1132, permute_746);  permute_746 = None
        permute_747: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1132, [1, 0])
        mm_273: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_747, view_392);  permute_747 = None
        sum_134: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1132, [0], True, dtype = torch.float32);  view_1132 = None
        view_1133: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_134, [16384]);  sum_134 = None
        convert_element_type_1851: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1133, torch.bfloat16);  view_1133 = None
        view_1134: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_272, [1, 128, 4096]);  mm_272 = None
        convert_element_type_1852: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1134, torch.float32);  view_1134 = None
        convert_element_type_1853: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_273, torch.float32);  mm_273 = None
        convert_element_type_1854: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1851, torch.float32);  convert_element_type_1851 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_274: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_743, view_414);  permute_743 = view_414 = None
        mm_275: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1129, permute_752);  view_1129 = permute_752 = None
        view_1136: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_275, [1, 128, 4096]);  mm_275 = None
        convert_element_type_1859: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_274, torch.float32);  mm_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1137: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1136, [1, 128, 16, 256]);  view_1136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_754: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1137, [0, 2, 1, 3]);  view_1137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1138: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_754, [16, 128, 256]);  permute_754 = None
        bmm_108: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_755, view_1138);  permute_755 = None
        bmm_109: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1138, permute_756);  view_1138 = permute_756 = None
        view_1139: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_108, [1, 16, 128, 256]);  bmm_108 = None
        view_1140: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_109, [1, 16, 128, 128]);  bmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1864: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1140, torch.float32);  view_1140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_575: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1864, div_29);  convert_element_type_1864 = None
        sum_135: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_575, [-1], True)
        neg_97: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_29);  div_29 = None
        fma_13: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_97, sum_135, mul_575);  neg_97 = sum_135 = mul_575 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1865: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_13, torch.bfloat16);  fma_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_89: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1865, 16.0);  convert_element_type_1865 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1141: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_89, [16, 128, 128]);  div_89 = None
        bmm_110: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_757, view_1141);  permute_757 = None
        bmm_111: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1141, permute_758);  view_1141 = permute_758 = None
        view_1142: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_110, [1, 16, 256, 128]);  bmm_110 = None
        view_1143: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_111, [1, 16, 128, 256]);  bmm_111 = None
        convert_element_type_1871: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1142, torch.float32);  view_1142 = None
        permute_759: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1871, [0, 1, 3, 2]);  convert_element_type_1871 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1872: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_759, torch.bfloat16);  permute_759 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_14: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1143, torch.bfloat16);  view_1143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_760: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_14, [0, 2, 1, 3]);  convert_element_type_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_761: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1872, [0, 2, 1, 3]);  convert_element_type_1872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_280: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_760, 3, 0, 64)
        slice_281: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_760, 3, 64, 256);  permute_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_282: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_761, 3, 0, 64)
        slice_283: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_761, 3, 64, 256);  permute_761 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_113: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_194, [1, 128, 1, 32, 2]);  unsqueeze_194 = None
        clone_113: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
        view_401: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [1, 128, 1, 64]);  clone_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_576: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_280, view_401)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1144: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_576, [1, 128, 16, 32, 2]);  mul_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_52: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1144, -1, 0)
        select_53: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1144, -1, 1);  view_1144 = None
        neg_98: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_52);  select_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_104: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_98, 3, 1, 9223372036854775807, 2);  neg_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_105: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_53, 3, 0, 9223372036854775807, 2);  select_53 = None
        add_418: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_104, slice_scatter_105);  slice_scatter_104 = slice_scatter_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_114: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_196, [1, 128, 1, 32, 2]);  unsqueeze_196 = None
        clone_114: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_114, memory_format = torch.contiguous_format);  expand_114 = None
        view_402: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_114, [1, 128, 1, 64]);  clone_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_577: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_280, view_402);  slice_280 = None
        add_419: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_418, mul_577);  add_418 = mul_577 = None
        mul_578: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_282, view_401);  view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1145: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_578, [1, 128, 16, 32, 2]);  mul_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_54: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1145, -1, 0)
        select_55: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1145, -1, 1);  view_1145 = None
        neg_99: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_54);  select_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_106: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_99, 3, 1, 9223372036854775807, 2);  neg_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_107: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_55, 3, 0, 9223372036854775807, 2);  select_55 = None
        add_420: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_106, slice_scatter_107);  slice_scatter_106 = slice_scatter_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_579: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_282, view_402);  slice_282 = view_402 = None
        add_421: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_420, mul_579);  add_420 = mul_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_108: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_281, 3, 64, 9223372036854775807);  slice_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_109: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_419, 3, 0, 64);  add_419 = None
        add_422: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_108, slice_scatter_109);  slice_scatter_108 = slice_scatter_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_110: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_283, 3, 64, 9223372036854775807);  slice_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_111: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_421, 3, 0, 64);  add_421 = None
        add_423: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_110, slice_scatter_111);  slice_scatter_110 = slice_scatter_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_762: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1139, [0, 2, 1, 3]);  view_1139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_240: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_762, memory_format = torch.contiguous_format);  permute_762 = None
        view_1146: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_240, [1, 128, 4096]);  clone_240 = None
        view_1147: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_423, [1, 128, 4096]);  add_423 = None
        view_1148: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_422, [1, 128, 4096]);  add_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1149: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1146, [128, 4096]);  view_1146 = None
        permute_763: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1149, [1, 0])
        mm_276: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_763, view_392);  permute_763 = None
        mm_277: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1149, permute_765);  view_1149 = permute_765 = None
        view_1150: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_277, [1, 128, 4096]);  mm_277 = None
        convert_element_type_1878: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1150, torch.float32);  view_1150 = None
        add_424: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1852, convert_element_type_1878);  convert_element_type_1852 = convert_element_type_1878 = None
        convert_element_type_1879: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_276, torch.float32);  mm_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1151: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1147, [128, 4096]);  view_1147 = None
        permute_767: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1151, [1, 0])
        mm_278: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_767, view_392);  permute_767 = None
        mm_279: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1151, permute_769);  view_1151 = permute_769 = None
        view_1152: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_279, [1, 128, 4096]);  mm_279 = None
        convert_element_type_1884: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1152, torch.float32);  view_1152 = None
        add_425: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_424, convert_element_type_1884);  add_424 = convert_element_type_1884 = None
        convert_element_type_1885: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_278, torch.float32);  mm_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1153: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1148, [128, 4096]);  view_1148 = None
        permute_771: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1153, [1, 0])
        mm_280: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_771, view_392);  permute_771 = view_392 = None
        mm_281: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1153, permute_773);  view_1153 = permute_773 = None
        view_1154: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_281, [1, 128, 4096]);  mm_281 = None
        convert_element_type_1890: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1154, torch.float32);  view_1154 = None
        add_426: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_425, convert_element_type_1890);  add_425 = convert_element_type_1890 = None
        convert_element_type_1891: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_280, torch.float32);  mm_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_581: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_426, primals_157);  primals_157 = None
        mul_582: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_581, 4096)
        sum_136: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_581, [2], True)
        mul_583: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_581, mul_140);  mul_581 = None
        sum_137: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_583, [2], True);  mul_583 = None
        mul_584: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, sum_137);  sum_137 = None
        sub_122: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_582, sum_136);  mul_582 = sum_136 = None
        sub_123: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_122, mul_584);  sub_122 = mul_584 = None
        mul_585: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_90, sub_123);  div_90 = sub_123 = None
        mul_586: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_426, mul_140);  mul_140 = None
        sum_138: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_586, [0, 1]);  mul_586 = None
        sum_139: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_426, [0, 1]);  add_426 = None
        add_427: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_415, mul_585);  add_415 = mul_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1892: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_427, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1155: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1892, [128, 4096]);  convert_element_type_1892 = None
        mm_282: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1155, permute_775);  permute_775 = None
        permute_776: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1155, [1, 0])
        mm_283: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_776, view_390);  view_390 = None
        sum_140: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1155, [0], True, dtype = torch.float32)
        view_1156: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_140, [4096]);  sum_140 = None
        convert_element_type_1897: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1156, torch.bfloat16);  view_1156 = None
        view_1157: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_282, [1, 128, 16384]);  mm_282 = None
        convert_element_type_1898: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1157, torch.float32);  view_1157 = None
        convert_element_type_1899: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_283, torch.float32);  mm_283 = None
        convert_element_type_1900: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1897, torch.float32);  convert_element_type_1897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_389: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [1, 128, 16384]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_136: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_389, 0.5)
        mul_587: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1898, mul_136);  mul_136 = None
        convert_element_type_525: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_389, torch.float32)
        pow_14: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_525, 3.0)
        mul_137: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_125: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_389, mul_137);  view_389 = mul_137 = None
        mul_138: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_125, 0.7978845608028654);  add_125 = None
        tanh_13: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_138);  mul_138 = None
        add_126: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
        mul_588: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1898, add_126);  convert_element_type_1898 = add_126 = None
        convert_element_type_1901: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_588, torch.bfloat16);  mul_588 = None
        mul_589: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
        sub_124: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_589);  mul_589 = None
        mul_590: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_587, sub_124);  mul_587 = sub_124 = None
        mul_591: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_590, 0.7978845608028654);  mul_590 = None
        convert_element_type_1902: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_591, torch.bfloat16)
        mul_592: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_591, 0.044715);  mul_591 = None
        pow_43: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_525, 2.0);  convert_element_type_525 = None
        mul_593: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_43, 3.0);  pow_43 = None
        mul_594: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
        convert_element_type_1903: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_594, torch.bfloat16);  mul_594 = None
        add_428: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1902, convert_element_type_1903);  convert_element_type_1902 = convert_element_type_1903 = None
        mul_595: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1901, 0.5);  convert_element_type_1901 = None
        add_429: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_428, mul_595);  add_428 = mul_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1158: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_429, [128, 16384]);  add_429 = None
        mm_284: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1158, permute_779);  permute_779 = None
        permute_780: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1158, [1, 0])
        mm_285: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_780, view_364);  permute_780 = None
        sum_141: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1158, [0], True, dtype = torch.float32);  view_1158 = None
        view_1159: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_141, [16384]);  sum_141 = None
        convert_element_type_1908: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1159, torch.bfloat16);  view_1159 = None
        view_1160: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_284, [1, 128, 4096]);  mm_284 = None
        convert_element_type_1909: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1160, torch.float32);  view_1160 = None
        convert_element_type_1910: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_285, torch.float32);  mm_285 = None
        convert_element_type_1911: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1908, torch.float32);  convert_element_type_1908 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_286: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_776, view_386);  permute_776 = view_386 = None
        mm_287: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1155, permute_785);  view_1155 = permute_785 = None
        view_1162: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_287, [1, 128, 4096]);  mm_287 = None
        convert_element_type_1916: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_286, torch.float32);  mm_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1163: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1162, [1, 128, 16, 256]);  view_1162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_787: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1163, [0, 2, 1, 3]);  view_1163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1164: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_787, [16, 128, 256]);  permute_787 = None
        bmm_112: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_788, view_1164);  permute_788 = None
        bmm_113: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1164, permute_789);  view_1164 = permute_789 = None
        view_1165: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_112, [1, 16, 128, 256]);  bmm_112 = None
        view_1166: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_113, [1, 16, 128, 128]);  bmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1921: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1166, torch.float32);  view_1166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_596: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1921, div_27);  convert_element_type_1921 = None
        sum_142: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_596, [-1], True)
        neg_100: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_27);  div_27 = None
        fma_14: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_100, sum_142, mul_596);  neg_100 = sum_142 = mul_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1922: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_14, torch.bfloat16);  fma_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_91: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1922, 16.0);  convert_element_type_1922 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1167: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_91, [16, 128, 128]);  div_91 = None
        bmm_114: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_790, view_1167);  permute_790 = None
        bmm_115: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1167, permute_791);  view_1167 = permute_791 = None
        view_1168: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_114, [1, 16, 256, 128]);  bmm_114 = None
        view_1169: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_115, [1, 16, 128, 256]);  bmm_115 = None
        convert_element_type_1928: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1168, torch.float32);  view_1168 = None
        permute_792: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1928, [0, 1, 3, 2]);  convert_element_type_1928 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1929: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_792, torch.bfloat16);  permute_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_13: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1169, torch.bfloat16);  view_1169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_793: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_13, [0, 2, 1, 3]);  convert_element_type_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_794: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1929, [0, 2, 1, 3]);  convert_element_type_1929 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_284: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_793, 3, 0, 64)
        slice_285: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_793, 3, 64, 256);  permute_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_286: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_794, 3, 0, 64)
        slice_287: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_794, 3, 64, 256);  permute_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_105: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_181, [1, 128, 1, 32, 2]);  unsqueeze_181 = None
        clone_105: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_373: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_105, [1, 128, 1, 64]);  clone_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_597: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_284, view_373)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1170: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_597, [1, 128, 16, 32, 2]);  mul_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_56: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1170, -1, 0)
        select_57: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1170, -1, 1);  view_1170 = None
        neg_101: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_56);  select_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_112: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_101, 3, 1, 9223372036854775807, 2);  neg_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_113: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_57, 3, 0, 9223372036854775807, 2);  select_57 = None
        add_430: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_112, slice_scatter_113);  slice_scatter_112 = slice_scatter_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_106: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_183, [1, 128, 1, 32, 2]);  unsqueeze_183 = None
        clone_106: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_106, memory_format = torch.contiguous_format);  expand_106 = None
        view_374: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [1, 128, 1, 64]);  clone_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_598: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_284, view_374);  slice_284 = None
        add_431: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_430, mul_598);  add_430 = mul_598 = None
        mul_599: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_286, view_373);  view_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1171: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_599, [1, 128, 16, 32, 2]);  mul_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_58: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1171, -1, 0)
        select_59: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1171, -1, 1);  view_1171 = None
        neg_102: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_58);  select_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_114: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_102, 3, 1, 9223372036854775807, 2);  neg_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_115: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_59, 3, 0, 9223372036854775807, 2);  select_59 = None
        add_432: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_114, slice_scatter_115);  slice_scatter_114 = slice_scatter_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_600: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_286, view_374);  slice_286 = view_374 = None
        add_433: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_432, mul_600);  add_432 = mul_600 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_116: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_285, 3, 64, 9223372036854775807);  slice_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_117: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_431, 3, 0, 64);  add_431 = None
        add_434: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_116, slice_scatter_117);  slice_scatter_116 = slice_scatter_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_118: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_287, 3, 64, 9223372036854775807);  slice_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_119: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_433, 3, 0, 64);  add_433 = None
        add_435: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_118, slice_scatter_119);  slice_scatter_118 = slice_scatter_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_795: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1165, [0, 2, 1, 3]);  view_1165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_241: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_795, memory_format = torch.contiguous_format);  permute_795 = None
        view_1172: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_241, [1, 128, 4096]);  clone_241 = None
        view_1173: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_435, [1, 128, 4096]);  add_435 = None
        view_1174: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_434, [1, 128, 4096]);  add_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1175: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1172, [128, 4096]);  view_1172 = None
        permute_796: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1175, [1, 0])
        mm_288: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_796, view_364);  permute_796 = None
        mm_289: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1175, permute_798);  view_1175 = permute_798 = None
        view_1176: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_289, [1, 128, 4096]);  mm_289 = None
        convert_element_type_1935: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1176, torch.float32);  view_1176 = None
        add_436: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1909, convert_element_type_1935);  convert_element_type_1909 = convert_element_type_1935 = None
        convert_element_type_1936: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_288, torch.float32);  mm_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1177: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1173, [128, 4096]);  view_1173 = None
        permute_800: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1177, [1, 0])
        mm_290: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_800, view_364);  permute_800 = None
        mm_291: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1177, permute_802);  view_1177 = permute_802 = None
        view_1178: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_291, [1, 128, 4096]);  mm_291 = None
        convert_element_type_1941: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1178, torch.float32);  view_1178 = None
        add_437: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_436, convert_element_type_1941);  add_436 = convert_element_type_1941 = None
        convert_element_type_1942: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_290, torch.float32);  mm_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1179: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1174, [128, 4096]);  view_1174 = None
        permute_804: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1179, [1, 0])
        mm_292: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_804, view_364);  permute_804 = view_364 = None
        mm_293: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1179, permute_806);  view_1179 = permute_806 = None
        view_1180: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_293, [1, 128, 4096]);  mm_293 = None
        convert_element_type_1947: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1180, torch.float32);  view_1180 = None
        add_438: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_437, convert_element_type_1947);  add_437 = convert_element_type_1947 = None
        convert_element_type_1948: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_292, torch.float32);  mm_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_602: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_438, primals_146);  primals_146 = None
        mul_603: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_602, 4096)
        sum_143: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_602, [2], True)
        mul_604: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_602, mul_130);  mul_602 = None
        sum_144: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_604, [2], True);  mul_604 = None
        mul_605: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, sum_144);  sum_144 = None
        sub_126: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_603, sum_143);  mul_603 = sum_143 = None
        sub_127: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_126, mul_605);  sub_126 = mul_605 = None
        mul_606: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_92, sub_127);  div_92 = sub_127 = None
        mul_607: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_438, mul_130);  mul_130 = None
        sum_145: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 1]);  mul_607 = None
        sum_146: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_438, [0, 1]);  add_438 = None
        add_439: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_427, mul_606);  add_427 = mul_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_1949: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_439, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1181: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1949, [128, 4096]);  convert_element_type_1949 = None
        mm_294: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1181, permute_808);  permute_808 = None
        permute_809: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1181, [1, 0])
        mm_295: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_809, view_362);  view_362 = None
        sum_147: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1181, [0], True, dtype = torch.float32)
        view_1182: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [4096]);  sum_147 = None
        convert_element_type_1954: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1182, torch.bfloat16);  view_1182 = None
        view_1183: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_294, [1, 128, 16384]);  mm_294 = None
        convert_element_type_1955: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1183, torch.float32);  view_1183 = None
        convert_element_type_1956: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_295, torch.float32);  mm_295 = None
        convert_element_type_1957: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1954, torch.float32);  convert_element_type_1954 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_361: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [1, 128, 16384]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_126: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_361, 0.5)
        mul_608: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1955, mul_126);  mul_126 = None
        convert_element_type_487: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_361, torch.float32)
        pow_13: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_487, 3.0)
        mul_127: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_116: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_361, mul_127);  view_361 = mul_127 = None
        mul_128: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_116, 0.7978845608028654);  add_116 = None
        tanh_12: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_128);  mul_128 = None
        add_117: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
        mul_609: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1955, add_117);  convert_element_type_1955 = add_117 = None
        convert_element_type_1958: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_609, torch.bfloat16);  mul_609 = None
        mul_610: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
        sub_128: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_610);  mul_610 = None
        mul_611: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_608, sub_128);  mul_608 = sub_128 = None
        mul_612: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_611, 0.7978845608028654);  mul_611 = None
        convert_element_type_1959: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_612, torch.bfloat16)
        mul_613: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_612, 0.044715);  mul_612 = None
        pow_44: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_487, 2.0);  convert_element_type_487 = None
        mul_614: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_44, 3.0);  pow_44 = None
        mul_615: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
        convert_element_type_1960: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_615, torch.bfloat16);  mul_615 = None
        add_440: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1959, convert_element_type_1960);  convert_element_type_1959 = convert_element_type_1960 = None
        mul_616: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1958, 0.5);  convert_element_type_1958 = None
        add_441: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_440, mul_616);  add_440 = mul_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1184: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_441, [128, 16384]);  add_441 = None
        mm_296: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1184, permute_812);  permute_812 = None
        permute_813: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1184, [1, 0])
        mm_297: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_813, view_336);  permute_813 = None
        sum_148: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1184, [0], True, dtype = torch.float32);  view_1184 = None
        view_1185: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_148, [16384]);  sum_148 = None
        convert_element_type_1965: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1185, torch.bfloat16);  view_1185 = None
        view_1186: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_296, [1, 128, 4096]);  mm_296 = None
        convert_element_type_1966: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1186, torch.float32);  view_1186 = None
        convert_element_type_1967: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_297, torch.float32);  mm_297 = None
        convert_element_type_1968: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1965, torch.float32);  convert_element_type_1965 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_298: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_809, view_358);  permute_809 = view_358 = None
        mm_299: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1181, permute_818);  view_1181 = permute_818 = None
        view_1188: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_299, [1, 128, 4096]);  mm_299 = None
        convert_element_type_1973: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_298, torch.float32);  mm_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1189: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1188, [1, 128, 16, 256]);  view_1188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_820: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1189, [0, 2, 1, 3]);  view_1189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1190: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_820, [16, 128, 256]);  permute_820 = None
        bmm_116: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_821, view_1190);  permute_821 = None
        bmm_117: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1190, permute_822);  view_1190 = permute_822 = None
        view_1191: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_116, [1, 16, 128, 256]);  bmm_116 = None
        view_1192: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_117, [1, 16, 128, 128]);  bmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1978: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1192, torch.float32);  view_1192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_617: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1978, div_25);  convert_element_type_1978 = None
        sum_149: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_617, [-1], True)
        neg_103: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_25);  div_25 = None
        fma_15: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_103, sum_149, mul_617);  neg_103 = sum_149 = mul_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1979: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_15, torch.bfloat16);  fma_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_93: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1979, 16.0);  convert_element_type_1979 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1193: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_93, [16, 128, 128]);  div_93 = None
        bmm_118: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_823, view_1193);  permute_823 = None
        bmm_119: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1193, permute_824);  view_1193 = permute_824 = None
        view_1194: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_118, [1, 16, 256, 128]);  bmm_118 = None
        view_1195: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_119, [1, 16, 128, 256]);  bmm_119 = None
        convert_element_type_1985: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1194, torch.float32);  view_1194 = None
        permute_825: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1985, [0, 1, 3, 2]);  convert_element_type_1985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1986: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_825, torch.bfloat16);  permute_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_12: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1195, torch.bfloat16);  view_1195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_826: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_12, [0, 2, 1, 3]);  convert_element_type_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_827: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1986, [0, 2, 1, 3]);  convert_element_type_1986 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_288: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_826, 3, 0, 64)
        slice_289: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_826, 3, 64, 256);  permute_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_290: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_827, 3, 0, 64)
        slice_291: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_827, 3, 64, 256);  permute_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_97: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_168, [1, 128, 1, 32, 2]);  unsqueeze_168 = None
        clone_97: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_345: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_97, [1, 128, 1, 64]);  clone_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_618: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_288, view_345)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1196: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_618, [1, 128, 16, 32, 2]);  mul_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_60: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1196, -1, 0)
        select_61: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1196, -1, 1);  view_1196 = None
        neg_104: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_60);  select_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_120: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_104, 3, 1, 9223372036854775807, 2);  neg_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_121: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_61, 3, 0, 9223372036854775807, 2);  select_61 = None
        add_442: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_120, slice_scatter_121);  slice_scatter_120 = slice_scatter_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_98: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_170, [1, 128, 1, 32, 2]);  unsqueeze_170 = None
        clone_98: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
        view_346: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_98, [1, 128, 1, 64]);  clone_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_619: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_288, view_346);  slice_288 = None
        add_443: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_442, mul_619);  add_442 = mul_619 = None
        mul_620: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_290, view_345);  view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1197: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_620, [1, 128, 16, 32, 2]);  mul_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_62: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1197, -1, 0)
        select_63: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1197, -1, 1);  view_1197 = None
        neg_105: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_62);  select_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_122: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_105, 3, 1, 9223372036854775807, 2);  neg_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_123: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_63, 3, 0, 9223372036854775807, 2);  select_63 = None
        add_444: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_122, slice_scatter_123);  slice_scatter_122 = slice_scatter_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_621: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_290, view_346);  slice_290 = view_346 = None
        add_445: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_444, mul_621);  add_444 = mul_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_124: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_289, 3, 64, 9223372036854775807);  slice_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_125: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_443, 3, 0, 64);  add_443 = None
        add_446: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_124, slice_scatter_125);  slice_scatter_124 = slice_scatter_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_126: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_291, 3, 64, 9223372036854775807);  slice_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_127: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_445, 3, 0, 64);  add_445 = None
        add_447: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_126, slice_scatter_127);  slice_scatter_126 = slice_scatter_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_828: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1191, [0, 2, 1, 3]);  view_1191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_242: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_828, memory_format = torch.contiguous_format);  permute_828 = None
        view_1198: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_242, [1, 128, 4096]);  clone_242 = None
        view_1199: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_447, [1, 128, 4096]);  add_447 = None
        view_1200: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_446, [1, 128, 4096]);  add_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1201: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1198, [128, 4096]);  view_1198 = None
        permute_829: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1201, [1, 0])
        mm_300: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_829, view_336);  permute_829 = None
        mm_301: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1201, permute_831);  view_1201 = permute_831 = None
        view_1202: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_301, [1, 128, 4096]);  mm_301 = None
        convert_element_type_1992: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1202, torch.float32);  view_1202 = None
        add_448: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1966, convert_element_type_1992);  convert_element_type_1966 = convert_element_type_1992 = None
        convert_element_type_1993: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_300, torch.float32);  mm_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1203: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1199, [128, 4096]);  view_1199 = None
        permute_833: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1203, [1, 0])
        mm_302: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_833, view_336);  permute_833 = None
        mm_303: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1203, permute_835);  view_1203 = permute_835 = None
        view_1204: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_303, [1, 128, 4096]);  mm_303 = None
        convert_element_type_1998: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1204, torch.float32);  view_1204 = None
        add_449: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_448, convert_element_type_1998);  add_448 = convert_element_type_1998 = None
        convert_element_type_1999: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_302, torch.float32);  mm_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1205: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1200, [128, 4096]);  view_1200 = None
        permute_837: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1205, [1, 0])
        mm_304: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_837, view_336);  permute_837 = view_336 = None
        mm_305: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1205, permute_839);  view_1205 = permute_839 = None
        view_1206: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_305, [1, 128, 4096]);  mm_305 = None
        convert_element_type_2004: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1206, torch.float32);  view_1206 = None
        add_450: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_449, convert_element_type_2004);  add_449 = convert_element_type_2004 = None
        convert_element_type_2005: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_304, torch.float32);  mm_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_623: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_450, primals_135);  primals_135 = None
        mul_624: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_623, 4096)
        sum_150: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_623, [2], True)
        mul_625: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_623, mul_120);  mul_623 = None
        sum_151: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_625, [2], True);  mul_625 = None
        mul_626: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, sum_151);  sum_151 = None
        sub_130: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_624, sum_150);  mul_624 = sum_150 = None
        sub_131: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_130, mul_626);  sub_130 = mul_626 = None
        mul_627: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_94, sub_131);  div_94 = sub_131 = None
        mul_628: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_450, mul_120);  mul_120 = None
        sum_152: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_628, [0, 1]);  mul_628 = None
        sum_153: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_450, [0, 1]);  add_450 = None
        add_451: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_439, mul_627);  add_439 = mul_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2006: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_451, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1207: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2006, [128, 4096]);  convert_element_type_2006 = None
        mm_306: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1207, permute_841);  permute_841 = None
        permute_842: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1207, [1, 0])
        mm_307: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_842, view_334);  view_334 = None
        sum_154: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1207, [0], True, dtype = torch.float32)
        view_1208: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_154, [4096]);  sum_154 = None
        convert_element_type_2011: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1208, torch.bfloat16);  view_1208 = None
        view_1209: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_306, [1, 128, 16384]);  mm_306 = None
        convert_element_type_2012: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1209, torch.float32);  view_1209 = None
        convert_element_type_2013: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_307, torch.float32);  mm_307 = None
        convert_element_type_2014: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2011, torch.float32);  convert_element_type_2011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_333: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [1, 128, 16384]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_333, 0.5)
        mul_629: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2012, mul_116);  mul_116 = None
        convert_element_type_449: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_333, torch.float32)
        pow_12: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_449, 3.0)
        mul_117: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_107: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_333, mul_117);  view_333 = mul_117 = None
        mul_118: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_107, 0.7978845608028654);  add_107 = None
        tanh_11: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_108: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_630: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2012, add_108);  convert_element_type_2012 = add_108 = None
        convert_element_type_2015: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_630, torch.bfloat16);  mul_630 = None
        mul_631: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_132: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_631);  mul_631 = None
        mul_632: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_629, sub_132);  mul_629 = sub_132 = None
        mul_633: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_632, 0.7978845608028654);  mul_632 = None
        convert_element_type_2016: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_633, torch.bfloat16)
        mul_634: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_633, 0.044715);  mul_633 = None
        pow_45: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_449, 2.0);  convert_element_type_449 = None
        mul_635: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_45, 3.0);  pow_45 = None
        mul_636: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_634, mul_635);  mul_634 = mul_635 = None
        convert_element_type_2017: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_636, torch.bfloat16);  mul_636 = None
        add_452: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2016, convert_element_type_2017);  convert_element_type_2016 = convert_element_type_2017 = None
        mul_637: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2015, 0.5);  convert_element_type_2015 = None
        add_453: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_452, mul_637);  add_452 = mul_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1210: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_453, [128, 16384]);  add_453 = None
        mm_308: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1210, permute_845);  permute_845 = None
        permute_846: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1210, [1, 0])
        mm_309: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_846, view_308);  permute_846 = None
        sum_155: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True, dtype = torch.float32);  view_1210 = None
        view_1211: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_155, [16384]);  sum_155 = None
        convert_element_type_2022: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1211, torch.bfloat16);  view_1211 = None
        view_1212: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_308, [1, 128, 4096]);  mm_308 = None
        convert_element_type_2023: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1212, torch.float32);  view_1212 = None
        convert_element_type_2024: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_309, torch.float32);  mm_309 = None
        convert_element_type_2025: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2022, torch.float32);  convert_element_type_2022 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_310: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_842, view_330);  permute_842 = view_330 = None
        mm_311: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1207, permute_851);  view_1207 = permute_851 = None
        view_1214: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_311, [1, 128, 4096]);  mm_311 = None
        convert_element_type_2030: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_310, torch.float32);  mm_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1215: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1214, [1, 128, 16, 256]);  view_1214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_853: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1215, [0, 2, 1, 3]);  view_1215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1216: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_853, [16, 128, 256]);  permute_853 = None
        bmm_120: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_854, view_1216);  permute_854 = None
        bmm_121: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1216, permute_855);  view_1216 = permute_855 = None
        view_1217: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_120, [1, 16, 128, 256]);  bmm_120 = None
        view_1218: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_121, [1, 16, 128, 128]);  bmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2035: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1218, torch.float32);  view_1218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_638: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2035, div_23);  convert_element_type_2035 = None
        sum_156: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_638, [-1], True)
        neg_106: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_23);  div_23 = None
        fma_16: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_106, sum_156, mul_638);  neg_106 = sum_156 = mul_638 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2036: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_16, torch.bfloat16);  fma_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_95: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2036, 16.0);  convert_element_type_2036 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1219: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_95, [16, 128, 128]);  div_95 = None
        bmm_122: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_856, view_1219);  permute_856 = None
        bmm_123: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1219, permute_857);  view_1219 = permute_857 = None
        view_1220: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_122, [1, 16, 256, 128]);  bmm_122 = None
        view_1221: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_123, [1, 16, 128, 256]);  bmm_123 = None
        convert_element_type_2042: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1220, torch.float32);  view_1220 = None
        permute_858: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2042, [0, 1, 3, 2]);  convert_element_type_2042 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2043: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_858, torch.bfloat16);  permute_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_11: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1221, torch.bfloat16);  view_1221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_859: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_11, [0, 2, 1, 3]);  convert_element_type_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_860: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2043, [0, 2, 1, 3]);  convert_element_type_2043 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_292: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_859, 3, 0, 64)
        slice_293: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_859, 3, 64, 256);  permute_859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_294: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_860, 3, 0, 64)
        slice_295: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_860, 3, 64, 256);  permute_860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_89: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_155, [1, 128, 1, 32, 2]);  unsqueeze_155 = None
        clone_89: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_317: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [1, 128, 1, 64]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_639: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_292, view_317)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1222: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_639, [1, 128, 16, 32, 2]);  mul_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_64: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1222, -1, 0)
        select_65: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1222, -1, 1);  view_1222 = None
        neg_107: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_64);  select_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_128: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_107, 3, 1, 9223372036854775807, 2);  neg_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_129: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_65, 3, 0, 9223372036854775807, 2);  select_65 = None
        add_454: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_128, slice_scatter_129);  slice_scatter_128 = slice_scatter_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_90: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_157, [1, 128, 1, 32, 2]);  unsqueeze_157 = None
        clone_90: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_318: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [1, 128, 1, 64]);  clone_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_640: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_292, view_318);  slice_292 = None
        add_455: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_454, mul_640);  add_454 = mul_640 = None
        mul_641: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_294, view_317);  view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1223: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_641, [1, 128, 16, 32, 2]);  mul_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_66: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1223, -1, 0)
        select_67: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1223, -1, 1);  view_1223 = None
        neg_108: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_66);  select_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_130: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_108, 3, 1, 9223372036854775807, 2);  neg_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_131: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_67, 3, 0, 9223372036854775807, 2);  select_67 = None
        add_456: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_130, slice_scatter_131);  slice_scatter_130 = slice_scatter_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_642: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_294, view_318);  slice_294 = view_318 = None
        add_457: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_456, mul_642);  add_456 = mul_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_132: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_293, 3, 64, 9223372036854775807);  slice_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_133: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_455, 3, 0, 64);  add_455 = None
        add_458: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_132, slice_scatter_133);  slice_scatter_132 = slice_scatter_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_134: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_295, 3, 64, 9223372036854775807);  slice_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_135: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_457, 3, 0, 64);  add_457 = None
        add_459: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_134, slice_scatter_135);  slice_scatter_134 = slice_scatter_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_861: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1217, [0, 2, 1, 3]);  view_1217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_243: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_861, memory_format = torch.contiguous_format);  permute_861 = None
        view_1224: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_243, [1, 128, 4096]);  clone_243 = None
        view_1225: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_459, [1, 128, 4096]);  add_459 = None
        view_1226: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_458, [1, 128, 4096]);  add_458 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1227: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1224, [128, 4096]);  view_1224 = None
        permute_862: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1227, [1, 0])
        mm_312: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_862, view_308);  permute_862 = None
        mm_313: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1227, permute_864);  view_1227 = permute_864 = None
        view_1228: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_313, [1, 128, 4096]);  mm_313 = None
        convert_element_type_2049: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1228, torch.float32);  view_1228 = None
        add_460: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2023, convert_element_type_2049);  convert_element_type_2023 = convert_element_type_2049 = None
        convert_element_type_2050: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_312, torch.float32);  mm_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1229: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1225, [128, 4096]);  view_1225 = None
        permute_866: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1229, [1, 0])
        mm_314: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_866, view_308);  permute_866 = None
        mm_315: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1229, permute_868);  view_1229 = permute_868 = None
        view_1230: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_315, [1, 128, 4096]);  mm_315 = None
        convert_element_type_2055: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1230, torch.float32);  view_1230 = None
        add_461: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_460, convert_element_type_2055);  add_460 = convert_element_type_2055 = None
        convert_element_type_2056: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_314, torch.float32);  mm_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1231: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1226, [128, 4096]);  view_1226 = None
        permute_870: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1231, [1, 0])
        mm_316: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_870, view_308);  permute_870 = view_308 = None
        mm_317: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1231, permute_872);  view_1231 = permute_872 = None
        view_1232: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_317, [1, 128, 4096]);  mm_317 = None
        convert_element_type_2061: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1232, torch.float32);  view_1232 = None
        add_462: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_461, convert_element_type_2061);  add_461 = convert_element_type_2061 = None
        convert_element_type_2062: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_316, torch.float32);  mm_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_644: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_462, primals_124);  primals_124 = None
        mul_645: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_644, 4096)
        sum_157: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_644, [2], True)
        mul_646: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_644, mul_110);  mul_644 = None
        sum_158: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_646, [2], True);  mul_646 = None
        mul_647: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, sum_158);  sum_158 = None
        sub_134: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_645, sum_157);  mul_645 = sum_157 = None
        sub_135: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_134, mul_647);  sub_134 = mul_647 = None
        mul_648: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_96, sub_135);  div_96 = sub_135 = None
        mul_649: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_462, mul_110);  mul_110 = None
        sum_159: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_649, [0, 1]);  mul_649 = None
        sum_160: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_462, [0, 1]);  add_462 = None
        add_463: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_451, mul_648);  add_451 = mul_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2063: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_463, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1233: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2063, [128, 4096]);  convert_element_type_2063 = None
        mm_318: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1233, permute_874);  permute_874 = None
        permute_875: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1233, [1, 0])
        mm_319: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_875, view_306);  view_306 = None
        sum_161: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1233, [0], True, dtype = torch.float32)
        view_1234: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_161, [4096]);  sum_161 = None
        convert_element_type_2068: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1234, torch.bfloat16);  view_1234 = None
        view_1235: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_318, [1, 128, 16384]);  mm_318 = None
        convert_element_type_2069: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1235, torch.float32);  view_1235 = None
        convert_element_type_2070: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_319, torch.float32);  mm_319 = None
        convert_element_type_2071: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2068, torch.float32);  convert_element_type_2068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_305: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [1, 128, 16384]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_106: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_305, 0.5)
        mul_650: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2069, mul_106);  mul_106 = None
        convert_element_type_411: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_305, torch.float32)
        pow_11: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_411, 3.0)
        mul_107: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_98: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_305, mul_107);  view_305 = mul_107 = None
        mul_108: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_98, 0.7978845608028654);  add_98 = None
        tanh_10: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_108);  mul_108 = None
        add_99: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_651: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2069, add_99);  convert_element_type_2069 = add_99 = None
        convert_element_type_2072: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_651, torch.bfloat16);  mul_651 = None
        mul_652: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_136: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_652);  mul_652 = None
        mul_653: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_650, sub_136);  mul_650 = sub_136 = None
        mul_654: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_653, 0.7978845608028654);  mul_653 = None
        convert_element_type_2073: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_654, torch.bfloat16)
        mul_655: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_654, 0.044715);  mul_654 = None
        pow_46: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_411, 2.0);  convert_element_type_411 = None
        mul_656: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_46, 3.0);  pow_46 = None
        mul_657: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
        convert_element_type_2074: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_657, torch.bfloat16);  mul_657 = None
        add_464: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2073, convert_element_type_2074);  convert_element_type_2073 = convert_element_type_2074 = None
        mul_658: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2072, 0.5);  convert_element_type_2072 = None
        add_465: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_464, mul_658);  add_464 = mul_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1236: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_465, [128, 16384]);  add_465 = None
        mm_320: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1236, permute_878);  permute_878 = None
        permute_879: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1236, [1, 0])
        mm_321: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_879, view_280);  permute_879 = None
        sum_162: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1236, [0], True, dtype = torch.float32);  view_1236 = None
        view_1237: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_162, [16384]);  sum_162 = None
        convert_element_type_2079: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1237, torch.bfloat16);  view_1237 = None
        view_1238: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_320, [1, 128, 4096]);  mm_320 = None
        convert_element_type_2080: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1238, torch.float32);  view_1238 = None
        convert_element_type_2081: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_321, torch.float32);  mm_321 = None
        convert_element_type_2082: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2079, torch.float32);  convert_element_type_2079 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_322: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_875, view_302);  permute_875 = view_302 = None
        mm_323: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1233, permute_884);  view_1233 = permute_884 = None
        view_1240: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_323, [1, 128, 4096]);  mm_323 = None
        convert_element_type_2087: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_322, torch.float32);  mm_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1241: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1240, [1, 128, 16, 256]);  view_1240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_886: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1241, [0, 2, 1, 3]);  view_1241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1242: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_886, [16, 128, 256]);  permute_886 = None
        bmm_124: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_887, view_1242);  permute_887 = None
        bmm_125: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1242, permute_888);  view_1242 = permute_888 = None
        view_1243: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_124, [1, 16, 128, 256]);  bmm_124 = None
        view_1244: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_125, [1, 16, 128, 128]);  bmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2092: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1244, torch.float32);  view_1244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_659: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2092, div_21);  convert_element_type_2092 = None
        sum_163: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_659, [-1], True)
        neg_109: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_21);  div_21 = None
        fma_17: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_109, sum_163, mul_659);  neg_109 = sum_163 = mul_659 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2093: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_17, torch.bfloat16);  fma_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_97: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2093, 16.0);  convert_element_type_2093 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1245: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_97, [16, 128, 128]);  div_97 = None
        bmm_126: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_889, view_1245);  permute_889 = None
        bmm_127: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1245, permute_890);  view_1245 = permute_890 = None
        view_1246: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_126, [1, 16, 256, 128]);  bmm_126 = None
        view_1247: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_127, [1, 16, 128, 256]);  bmm_127 = None
        convert_element_type_2099: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1246, torch.float32);  view_1246 = None
        permute_891: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2099, [0, 1, 3, 2]);  convert_element_type_2099 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2100: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_891, torch.bfloat16);  permute_891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_10: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1247, torch.bfloat16);  view_1247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_892: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_10, [0, 2, 1, 3]);  convert_element_type_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_893: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2100, [0, 2, 1, 3]);  convert_element_type_2100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_296: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_892, 3, 0, 64)
        slice_297: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_892, 3, 64, 256);  permute_892 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_298: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_893, 3, 0, 64)
        slice_299: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_893, 3, 64, 256);  permute_893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_81: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_142, [1, 128, 1, 32, 2]);  unsqueeze_142 = None
        clone_81: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_289: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [1, 128, 1, 64]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_660: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_296, view_289)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1248: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_660, [1, 128, 16, 32, 2]);  mul_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_68: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1248, -1, 0)
        select_69: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1248, -1, 1);  view_1248 = None
        neg_110: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_68);  select_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_136: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_110, 3, 1, 9223372036854775807, 2);  neg_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_137: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_69, 3, 0, 9223372036854775807, 2);  select_69 = None
        add_466: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_136, slice_scatter_137);  slice_scatter_136 = slice_scatter_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_82: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_144, [1, 128, 1, 32, 2]);  unsqueeze_144 = None
        clone_82: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_290: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [1, 128, 1, 64]);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_661: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_296, view_290);  slice_296 = None
        add_467: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_466, mul_661);  add_466 = mul_661 = None
        mul_662: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_298, view_289);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1249: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_662, [1, 128, 16, 32, 2]);  mul_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_70: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1249, -1, 0)
        select_71: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1249, -1, 1);  view_1249 = None
        neg_111: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_70);  select_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_138: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_111, 3, 1, 9223372036854775807, 2);  neg_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_139: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_71, 3, 0, 9223372036854775807, 2);  select_71 = None
        add_468: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_138, slice_scatter_139);  slice_scatter_138 = slice_scatter_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_663: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_298, view_290);  slice_298 = view_290 = None
        add_469: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_468, mul_663);  add_468 = mul_663 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_140: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_297, 3, 64, 9223372036854775807);  slice_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_141: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_467, 3, 0, 64);  add_467 = None
        add_470: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_140, slice_scatter_141);  slice_scatter_140 = slice_scatter_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_142: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_299, 3, 64, 9223372036854775807);  slice_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_143: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_469, 3, 0, 64);  add_469 = None
        add_471: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_142, slice_scatter_143);  slice_scatter_142 = slice_scatter_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_894: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1243, [0, 2, 1, 3]);  view_1243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_244: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_894, memory_format = torch.contiguous_format);  permute_894 = None
        view_1250: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_244, [1, 128, 4096]);  clone_244 = None
        view_1251: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_471, [1, 128, 4096]);  add_471 = None
        view_1252: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_470, [1, 128, 4096]);  add_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1253: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1250, [128, 4096]);  view_1250 = None
        permute_895: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1253, [1, 0])
        mm_324: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_895, view_280);  permute_895 = None
        mm_325: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1253, permute_897);  view_1253 = permute_897 = None
        view_1254: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_325, [1, 128, 4096]);  mm_325 = None
        convert_element_type_2106: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1254, torch.float32);  view_1254 = None
        add_472: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2080, convert_element_type_2106);  convert_element_type_2080 = convert_element_type_2106 = None
        convert_element_type_2107: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_324, torch.float32);  mm_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1255: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1251, [128, 4096]);  view_1251 = None
        permute_899: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1255, [1, 0])
        mm_326: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_899, view_280);  permute_899 = None
        mm_327: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1255, permute_901);  view_1255 = permute_901 = None
        view_1256: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_327, [1, 128, 4096]);  mm_327 = None
        convert_element_type_2112: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1256, torch.float32);  view_1256 = None
        add_473: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_472, convert_element_type_2112);  add_472 = convert_element_type_2112 = None
        convert_element_type_2113: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_326, torch.float32);  mm_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1257: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1252, [128, 4096]);  view_1252 = None
        permute_903: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1257, [1, 0])
        mm_328: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_903, view_280);  permute_903 = view_280 = None
        mm_329: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1257, permute_905);  view_1257 = permute_905 = None
        view_1258: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_329, [1, 128, 4096]);  mm_329 = None
        convert_element_type_2118: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1258, torch.float32);  view_1258 = None
        add_474: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_473, convert_element_type_2118);  add_473 = convert_element_type_2118 = None
        convert_element_type_2119: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_328, torch.float32);  mm_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_665: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_474, primals_113);  primals_113 = None
        mul_666: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_665, 4096)
        sum_164: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_665, [2], True)
        mul_667: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_665, mul_100);  mul_665 = None
        sum_165: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_667, [2], True);  mul_667 = None
        mul_668: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, sum_165);  sum_165 = None
        sub_138: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_666, sum_164);  mul_666 = sum_164 = None
        sub_139: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_138, mul_668);  sub_138 = mul_668 = None
        mul_669: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_98, sub_139);  div_98 = sub_139 = None
        mul_670: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_474, mul_100);  mul_100 = None
        sum_166: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_670, [0, 1]);  mul_670 = None
        sum_167: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_474, [0, 1]);  add_474 = None
        add_475: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_463, mul_669);  add_463 = mul_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2120: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_475, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1259: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2120, [128, 4096]);  convert_element_type_2120 = None
        mm_330: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1259, permute_907);  permute_907 = None
        permute_908: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1259, [1, 0])
        mm_331: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_908, view_278);  view_278 = None
        sum_168: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1259, [0], True, dtype = torch.float32)
        view_1260: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_168, [4096]);  sum_168 = None
        convert_element_type_2125: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1260, torch.bfloat16);  view_1260 = None
        view_1261: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_330, [1, 128, 16384]);  mm_330 = None
        convert_element_type_2126: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1261, torch.float32);  view_1261 = None
        convert_element_type_2127: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_331, torch.float32);  mm_331 = None
        convert_element_type_2128: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2125, torch.float32);  convert_element_type_2125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_277: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [1, 128, 16384]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_96: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_277, 0.5)
        mul_671: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2126, mul_96);  mul_96 = None
        convert_element_type_373: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_277, torch.float32)
        pow_10: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_373, 3.0)
        mul_97: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_89: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_277, mul_97);  view_277 = mul_97 = None
        mul_98: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_9: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_98);  mul_98 = None
        add_90: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_672: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2126, add_90);  convert_element_type_2126 = add_90 = None
        convert_element_type_2129: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_672, torch.bfloat16);  mul_672 = None
        mul_673: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_140: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_673);  mul_673 = None
        mul_674: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_671, sub_140);  mul_671 = sub_140 = None
        mul_675: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_674, 0.7978845608028654);  mul_674 = None
        convert_element_type_2130: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_675, torch.bfloat16)
        mul_676: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_675, 0.044715);  mul_675 = None
        pow_47: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_373, 2.0);  convert_element_type_373 = None
        mul_677: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_47, 3.0);  pow_47 = None
        mul_678: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_676, mul_677);  mul_676 = mul_677 = None
        convert_element_type_2131: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_678, torch.bfloat16);  mul_678 = None
        add_476: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2130, convert_element_type_2131);  convert_element_type_2130 = convert_element_type_2131 = None
        mul_679: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2129, 0.5);  convert_element_type_2129 = None
        add_477: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_476, mul_679);  add_476 = mul_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1262: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_477, [128, 16384]);  add_477 = None
        mm_332: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1262, permute_911);  permute_911 = None
        permute_912: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1262, [1, 0])
        mm_333: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_912, view_252);  permute_912 = None
        sum_169: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1262, [0], True, dtype = torch.float32);  view_1262 = None
        view_1263: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_169, [16384]);  sum_169 = None
        convert_element_type_2136: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1263, torch.bfloat16);  view_1263 = None
        view_1264: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_332, [1, 128, 4096]);  mm_332 = None
        convert_element_type_2137: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1264, torch.float32);  view_1264 = None
        convert_element_type_2138: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_333, torch.float32);  mm_333 = None
        convert_element_type_2139: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2136, torch.float32);  convert_element_type_2136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_334: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_908, view_274);  permute_908 = view_274 = None
        mm_335: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1259, permute_917);  view_1259 = permute_917 = None
        view_1266: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_335, [1, 128, 4096]);  mm_335 = None
        convert_element_type_2144: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_334, torch.float32);  mm_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1267: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1266, [1, 128, 16, 256]);  view_1266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_919: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1267, [0, 2, 1, 3]);  view_1267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1268: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_919, [16, 128, 256]);  permute_919 = None
        bmm_128: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_920, view_1268);  permute_920 = None
        bmm_129: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1268, permute_921);  view_1268 = permute_921 = None
        view_1269: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_128, [1, 16, 128, 256]);  bmm_128 = None
        view_1270: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_129, [1, 16, 128, 128]);  bmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2149: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1270, torch.float32);  view_1270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_680: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2149, div_19);  convert_element_type_2149 = None
        sum_170: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_680, [-1], True)
        neg_112: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_19);  div_19 = None
        fma_18: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_112, sum_170, mul_680);  neg_112 = sum_170 = mul_680 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2150: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_18, torch.bfloat16);  fma_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_99: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2150, 16.0);  convert_element_type_2150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1271: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_99, [16, 128, 128]);  div_99 = None
        bmm_130: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_922, view_1271);  permute_922 = None
        bmm_131: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1271, permute_923);  view_1271 = permute_923 = None
        view_1272: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_130, [1, 16, 256, 128]);  bmm_130 = None
        view_1273: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_131, [1, 16, 128, 256]);  bmm_131 = None
        convert_element_type_2156: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1272, torch.float32);  view_1272 = None
        permute_924: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2156, [0, 1, 3, 2]);  convert_element_type_2156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2157: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_924, torch.bfloat16);  permute_924 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_9: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1273, torch.bfloat16);  view_1273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_925: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_9, [0, 2, 1, 3]);  convert_element_type_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_926: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2157, [0, 2, 1, 3]);  convert_element_type_2157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_300: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_925, 3, 0, 64)
        slice_301: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_925, 3, 64, 256);  permute_925 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_302: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_926, 3, 0, 64)
        slice_303: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_926, 3, 64, 256);  permute_926 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_73: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_129, [1, 128, 1, 32, 2]);  unsqueeze_129 = None
        clone_73: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_261: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [1, 128, 1, 64]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_681: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_300, view_261)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1274: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_681, [1, 128, 16, 32, 2]);  mul_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_72: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1274, -1, 0)
        select_73: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1274, -1, 1);  view_1274 = None
        neg_113: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_72);  select_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_144: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_113, 3, 1, 9223372036854775807, 2);  neg_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_145: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_73, 3, 0, 9223372036854775807, 2);  select_73 = None
        add_478: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_144, slice_scatter_145);  slice_scatter_144 = slice_scatter_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_74: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_131, [1, 128, 1, 32, 2]);  unsqueeze_131 = None
        clone_74: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_262: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [1, 128, 1, 64]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_682: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_300, view_262);  slice_300 = None
        add_479: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_478, mul_682);  add_478 = mul_682 = None
        mul_683: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_302, view_261);  view_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1275: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_683, [1, 128, 16, 32, 2]);  mul_683 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_74: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1275, -1, 0)
        select_75: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1275, -1, 1);  view_1275 = None
        neg_114: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_74);  select_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_146: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_114, 3, 1, 9223372036854775807, 2);  neg_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_147: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_75, 3, 0, 9223372036854775807, 2);  select_75 = None
        add_480: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_146, slice_scatter_147);  slice_scatter_146 = slice_scatter_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_684: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_302, view_262);  slice_302 = view_262 = None
        add_481: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_480, mul_684);  add_480 = mul_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_148: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_301, 3, 64, 9223372036854775807);  slice_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_149: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_479, 3, 0, 64);  add_479 = None
        add_482: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_148, slice_scatter_149);  slice_scatter_148 = slice_scatter_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_150: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_303, 3, 64, 9223372036854775807);  slice_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_151: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_481, 3, 0, 64);  add_481 = None
        add_483: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_150, slice_scatter_151);  slice_scatter_150 = slice_scatter_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_927: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1269, [0, 2, 1, 3]);  view_1269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_245: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_927, memory_format = torch.contiguous_format);  permute_927 = None
        view_1276: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_245, [1, 128, 4096]);  clone_245 = None
        view_1277: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_483, [1, 128, 4096]);  add_483 = None
        view_1278: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_482, [1, 128, 4096]);  add_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1279: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1276, [128, 4096]);  view_1276 = None
        permute_928: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1279, [1, 0])
        mm_336: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_928, view_252);  permute_928 = None
        mm_337: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1279, permute_930);  view_1279 = permute_930 = None
        view_1280: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_337, [1, 128, 4096]);  mm_337 = None
        convert_element_type_2163: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1280, torch.float32);  view_1280 = None
        add_484: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2137, convert_element_type_2163);  convert_element_type_2137 = convert_element_type_2163 = None
        convert_element_type_2164: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_336, torch.float32);  mm_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1281: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1277, [128, 4096]);  view_1277 = None
        permute_932: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1281, [1, 0])
        mm_338: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_932, view_252);  permute_932 = None
        mm_339: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1281, permute_934);  view_1281 = permute_934 = None
        view_1282: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_339, [1, 128, 4096]);  mm_339 = None
        convert_element_type_2169: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1282, torch.float32);  view_1282 = None
        add_485: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_484, convert_element_type_2169);  add_484 = convert_element_type_2169 = None
        convert_element_type_2170: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_338, torch.float32);  mm_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1283: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1278, [128, 4096]);  view_1278 = None
        permute_936: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1283, [1, 0])
        mm_340: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_936, view_252);  permute_936 = view_252 = None
        mm_341: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1283, permute_938);  view_1283 = permute_938 = None
        view_1284: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_341, [1, 128, 4096]);  mm_341 = None
        convert_element_type_2175: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1284, torch.float32);  view_1284 = None
        add_486: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_485, convert_element_type_2175);  add_485 = convert_element_type_2175 = None
        convert_element_type_2176: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_340, torch.float32);  mm_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_686: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_486, primals_102);  primals_102 = None
        mul_687: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_686, 4096)
        sum_171: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_686, [2], True)
        mul_688: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_686, mul_90);  mul_686 = None
        sum_172: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_688, [2], True);  mul_688 = None
        mul_689: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, sum_172);  sum_172 = None
        sub_142: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_687, sum_171);  mul_687 = sum_171 = None
        sub_143: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_142, mul_689);  sub_142 = mul_689 = None
        mul_690: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_100, sub_143);  div_100 = sub_143 = None
        mul_691: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_486, mul_90);  mul_90 = None
        sum_173: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 1]);  mul_691 = None
        sum_174: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_486, [0, 1]);  add_486 = None
        add_487: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_475, mul_690);  add_475 = mul_690 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2177: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_487, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1285: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2177, [128, 4096]);  convert_element_type_2177 = None
        mm_342: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1285, permute_940);  permute_940 = None
        permute_941: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1285, [1, 0])
        mm_343: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_941, view_250);  view_250 = None
        sum_175: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1285, [0], True, dtype = torch.float32)
        view_1286: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_175, [4096]);  sum_175 = None
        convert_element_type_2182: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1286, torch.bfloat16);  view_1286 = None
        view_1287: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_342, [1, 128, 16384]);  mm_342 = None
        convert_element_type_2183: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1287, torch.float32);  view_1287 = None
        convert_element_type_2184: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_343, torch.float32);  mm_343 = None
        convert_element_type_2185: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2182, torch.float32);  convert_element_type_2182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_249: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [1, 128, 16384]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_86: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_249, 0.5)
        mul_692: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2183, mul_86);  mul_86 = None
        convert_element_type_335: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_249, torch.float32)
        pow_9: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_335, 3.0)
        mul_87: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_80: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_249, mul_87);  view_249 = mul_87 = None
        mul_88: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_80, 0.7978845608028654);  add_80 = None
        tanh_8: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_88);  mul_88 = None
        add_81: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_693: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2183, add_81);  convert_element_type_2183 = add_81 = None
        convert_element_type_2186: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_693, torch.bfloat16);  mul_693 = None
        mul_694: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_144: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_694);  mul_694 = None
        mul_695: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_692, sub_144);  mul_692 = sub_144 = None
        mul_696: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_695, 0.7978845608028654);  mul_695 = None
        convert_element_type_2187: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_696, torch.bfloat16)
        mul_697: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_696, 0.044715);  mul_696 = None
        pow_48: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_335, 2.0);  convert_element_type_335 = None
        mul_698: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_48, 3.0);  pow_48 = None
        mul_699: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_697, mul_698);  mul_697 = mul_698 = None
        convert_element_type_2188: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_699, torch.bfloat16);  mul_699 = None
        add_488: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2187, convert_element_type_2188);  convert_element_type_2187 = convert_element_type_2188 = None
        mul_700: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2186, 0.5);  convert_element_type_2186 = None
        add_489: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_488, mul_700);  add_488 = mul_700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1288: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_489, [128, 16384]);  add_489 = None
        mm_344: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1288, permute_944);  permute_944 = None
        permute_945: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1288, [1, 0])
        mm_345: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_945, view_224);  permute_945 = None
        sum_176: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1288, [0], True, dtype = torch.float32);  view_1288 = None
        view_1289: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_176, [16384]);  sum_176 = None
        convert_element_type_2193: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1289, torch.bfloat16);  view_1289 = None
        view_1290: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_344, [1, 128, 4096]);  mm_344 = None
        convert_element_type_2194: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1290, torch.float32);  view_1290 = None
        convert_element_type_2195: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_345, torch.float32);  mm_345 = None
        convert_element_type_2196: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2193, torch.float32);  convert_element_type_2193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_346: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_941, view_246);  permute_941 = view_246 = None
        mm_347: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1285, permute_950);  view_1285 = permute_950 = None
        view_1292: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_347, [1, 128, 4096]);  mm_347 = None
        convert_element_type_2201: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_346, torch.float32);  mm_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1293: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1292, [1, 128, 16, 256]);  view_1292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_952: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1293, [0, 2, 1, 3]);  view_1293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1294: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_952, [16, 128, 256]);  permute_952 = None
        bmm_132: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_953, view_1294);  permute_953 = None
        bmm_133: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1294, permute_954);  view_1294 = permute_954 = None
        view_1295: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_132, [1, 16, 128, 256]);  bmm_132 = None
        view_1296: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_133, [1, 16, 128, 128]);  bmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2206: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1296, torch.float32);  view_1296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_701: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2206, div_17);  convert_element_type_2206 = None
        sum_177: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_701, [-1], True)
        neg_115: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_17);  div_17 = None
        fma_19: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_115, sum_177, mul_701);  neg_115 = sum_177 = mul_701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2207: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_19, torch.bfloat16);  fma_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_101: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2207, 16.0);  convert_element_type_2207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1297: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_101, [16, 128, 128]);  div_101 = None
        bmm_134: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_955, view_1297);  permute_955 = None
        bmm_135: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1297, permute_956);  view_1297 = permute_956 = None
        view_1298: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_134, [1, 16, 256, 128]);  bmm_134 = None
        view_1299: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_135, [1, 16, 128, 256]);  bmm_135 = None
        convert_element_type_2213: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1298, torch.float32);  view_1298 = None
        permute_957: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2213, [0, 1, 3, 2]);  convert_element_type_2213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2214: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_957, torch.bfloat16);  permute_957 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_8: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1299, torch.bfloat16);  view_1299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_958: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_8, [0, 2, 1, 3]);  convert_element_type_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_959: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2214, [0, 2, 1, 3]);  convert_element_type_2214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_304: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_958, 3, 0, 64)
        slice_305: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_958, 3, 64, 256);  permute_958 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_306: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_959, 3, 0, 64)
        slice_307: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_959, 3, 64, 256);  permute_959 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_65: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_116, [1, 128, 1, 32, 2]);  unsqueeze_116 = None
        clone_65: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_233: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [1, 128, 1, 64]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_702: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_304, view_233)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1300: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_702, [1, 128, 16, 32, 2]);  mul_702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_76: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1300, -1, 0)
        select_77: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1300, -1, 1);  view_1300 = None
        neg_116: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_76);  select_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_152: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_116, 3, 1, 9223372036854775807, 2);  neg_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_153: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_77, 3, 0, 9223372036854775807, 2);  select_77 = None
        add_490: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_152, slice_scatter_153);  slice_scatter_152 = slice_scatter_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_66: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_118, [1, 128, 1, 32, 2]);  unsqueeze_118 = None
        clone_66: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_234: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_66, [1, 128, 1, 64]);  clone_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_703: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_304, view_234);  slice_304 = None
        add_491: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_490, mul_703);  add_490 = mul_703 = None
        mul_704: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_306, view_233);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1301: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_704, [1, 128, 16, 32, 2]);  mul_704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_78: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1301, -1, 0)
        select_79: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1301, -1, 1);  view_1301 = None
        neg_117: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_78);  select_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_154: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_117, 3, 1, 9223372036854775807, 2);  neg_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_155: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_79, 3, 0, 9223372036854775807, 2);  select_79 = None
        add_492: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_154, slice_scatter_155);  slice_scatter_154 = slice_scatter_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_705: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_306, view_234);  slice_306 = view_234 = None
        add_493: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_492, mul_705);  add_492 = mul_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_156: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_305, 3, 64, 9223372036854775807);  slice_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_157: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_491, 3, 0, 64);  add_491 = None
        add_494: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_156, slice_scatter_157);  slice_scatter_156 = slice_scatter_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_158: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_307, 3, 64, 9223372036854775807);  slice_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_159: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_493, 3, 0, 64);  add_493 = None
        add_495: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_158, slice_scatter_159);  slice_scatter_158 = slice_scatter_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_960: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1295, [0, 2, 1, 3]);  view_1295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_246: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_960, memory_format = torch.contiguous_format);  permute_960 = None
        view_1302: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_246, [1, 128, 4096]);  clone_246 = None
        view_1303: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_495, [1, 128, 4096]);  add_495 = None
        view_1304: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_494, [1, 128, 4096]);  add_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1305: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1302, [128, 4096]);  view_1302 = None
        permute_961: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1305, [1, 0])
        mm_348: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_961, view_224);  permute_961 = None
        mm_349: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1305, permute_963);  view_1305 = permute_963 = None
        view_1306: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_349, [1, 128, 4096]);  mm_349 = None
        convert_element_type_2220: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1306, torch.float32);  view_1306 = None
        add_496: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2194, convert_element_type_2220);  convert_element_type_2194 = convert_element_type_2220 = None
        convert_element_type_2221: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_348, torch.float32);  mm_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1307: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1303, [128, 4096]);  view_1303 = None
        permute_965: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1307, [1, 0])
        mm_350: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_965, view_224);  permute_965 = None
        mm_351: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1307, permute_967);  view_1307 = permute_967 = None
        view_1308: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_351, [1, 128, 4096]);  mm_351 = None
        convert_element_type_2226: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1308, torch.float32);  view_1308 = None
        add_497: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_496, convert_element_type_2226);  add_496 = convert_element_type_2226 = None
        convert_element_type_2227: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_350, torch.float32);  mm_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1309: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1304, [128, 4096]);  view_1304 = None
        permute_969: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1309, [1, 0])
        mm_352: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_969, view_224);  permute_969 = view_224 = None
        mm_353: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1309, permute_971);  view_1309 = permute_971 = None
        view_1310: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_353, [1, 128, 4096]);  mm_353 = None
        convert_element_type_2232: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1310, torch.float32);  view_1310 = None
        add_498: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_497, convert_element_type_2232);  add_497 = convert_element_type_2232 = None
        convert_element_type_2233: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_352, torch.float32);  mm_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_707: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_498, primals_91);  primals_91 = None
        mul_708: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_707, 4096)
        sum_178: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_707, [2], True)
        mul_709: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_707, mul_80);  mul_707 = None
        sum_179: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_709, [2], True);  mul_709 = None
        mul_710: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, sum_179);  sum_179 = None
        sub_146: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_708, sum_178);  mul_708 = sum_178 = None
        sub_147: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_146, mul_710);  sub_146 = mul_710 = None
        mul_711: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_102, sub_147);  div_102 = sub_147 = None
        mul_712: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_498, mul_80);  mul_80 = None
        sum_180: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_712, [0, 1]);  mul_712 = None
        sum_181: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_498, [0, 1]);  add_498 = None
        add_499: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_487, mul_711);  add_487 = mul_711 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2234: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_499, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1311: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2234, [128, 4096]);  convert_element_type_2234 = None
        mm_354: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1311, permute_973);  permute_973 = None
        permute_974: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1311, [1, 0])
        mm_355: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_974, view_222);  view_222 = None
        sum_182: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1311, [0], True, dtype = torch.float32)
        view_1312: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_182, [4096]);  sum_182 = None
        convert_element_type_2239: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1312, torch.bfloat16);  view_1312 = None
        view_1313: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_354, [1, 128, 16384]);  mm_354 = None
        convert_element_type_2240: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1313, torch.float32);  view_1313 = None
        convert_element_type_2241: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_355, torch.float32);  mm_355 = None
        convert_element_type_2242: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2239, torch.float32);  convert_element_type_2239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_221: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [1, 128, 16384]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_221, 0.5)
        mul_713: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2240, mul_76);  mul_76 = None
        convert_element_type_297: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_221, torch.float32)
        pow_8: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_297, 3.0)
        mul_77: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_71: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_221, mul_77);  view_221 = mul_77 = None
        mul_78: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_7: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_72: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_714: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2240, add_72);  convert_element_type_2240 = add_72 = None
        convert_element_type_2243: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_714, torch.bfloat16);  mul_714 = None
        mul_715: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_148: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_715);  mul_715 = None
        mul_716: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_713, sub_148);  mul_713 = sub_148 = None
        mul_717: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_716, 0.7978845608028654);  mul_716 = None
        convert_element_type_2244: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_717, torch.bfloat16)
        mul_718: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_717, 0.044715);  mul_717 = None
        pow_49: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_297, 2.0);  convert_element_type_297 = None
        mul_719: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_49, 3.0);  pow_49 = None
        mul_720: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
        convert_element_type_2245: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_720, torch.bfloat16);  mul_720 = None
        add_500: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2244, convert_element_type_2245);  convert_element_type_2244 = convert_element_type_2245 = None
        mul_721: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2243, 0.5);  convert_element_type_2243 = None
        add_501: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_500, mul_721);  add_500 = mul_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1314: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_501, [128, 16384]);  add_501 = None
        mm_356: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1314, permute_977);  permute_977 = None
        permute_978: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1314, [1, 0])
        mm_357: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_978, view_196);  permute_978 = None
        sum_183: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1314, [0], True, dtype = torch.float32);  view_1314 = None
        view_1315: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_183, [16384]);  sum_183 = None
        convert_element_type_2250: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1315, torch.bfloat16);  view_1315 = None
        view_1316: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_356, [1, 128, 4096]);  mm_356 = None
        convert_element_type_2251: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1316, torch.float32);  view_1316 = None
        convert_element_type_2252: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_357, torch.float32);  mm_357 = None
        convert_element_type_2253: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2250, torch.float32);  convert_element_type_2250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_358: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_974, view_218);  permute_974 = view_218 = None
        mm_359: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1311, permute_983);  view_1311 = permute_983 = None
        view_1318: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_359, [1, 128, 4096]);  mm_359 = None
        convert_element_type_2258: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_358, torch.float32);  mm_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1319: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1318, [1, 128, 16, 256]);  view_1318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_985: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1319, [0, 2, 1, 3]);  view_1319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1320: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_985, [16, 128, 256]);  permute_985 = None
        bmm_136: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_986, view_1320);  permute_986 = None
        bmm_137: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1320, permute_987);  view_1320 = permute_987 = None
        view_1321: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_136, [1, 16, 128, 256]);  bmm_136 = None
        view_1322: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_137, [1, 16, 128, 128]);  bmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2263: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1322, torch.float32);  view_1322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_722: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2263, div_15);  convert_element_type_2263 = None
        sum_184: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_722, [-1], True)
        neg_118: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_15);  div_15 = None
        fma_20: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_118, sum_184, mul_722);  neg_118 = sum_184 = mul_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2264: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_20, torch.bfloat16);  fma_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_103: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2264, 16.0);  convert_element_type_2264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1323: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_103, [16, 128, 128]);  div_103 = None
        bmm_138: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_988, view_1323);  permute_988 = None
        bmm_139: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1323, permute_989);  view_1323 = permute_989 = None
        view_1324: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_138, [1, 16, 256, 128]);  bmm_138 = None
        view_1325: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_139, [1, 16, 128, 256]);  bmm_139 = None
        convert_element_type_2270: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1324, torch.float32);  view_1324 = None
        permute_990: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2270, [0, 1, 3, 2]);  convert_element_type_2270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2271: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_990, torch.bfloat16);  permute_990 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_7: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1325, torch.bfloat16);  view_1325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_991: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_7, [0, 2, 1, 3]);  convert_element_type_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_992: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2271, [0, 2, 1, 3]);  convert_element_type_2271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_308: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_991, 3, 0, 64)
        slice_309: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_991, 3, 64, 256);  permute_991 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_310: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_992, 3, 0, 64)
        slice_311: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_992, 3, 64, 256);  permute_992 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_57: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_103, [1, 128, 1, 32, 2]);  unsqueeze_103 = None
        clone_57: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_205: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [1, 128, 1, 64]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_723: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_308, view_205)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1326: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_723, [1, 128, 16, 32, 2]);  mul_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_80: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1326, -1, 0)
        select_81: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1326, -1, 1);  view_1326 = None
        neg_119: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_80);  select_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_160: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_119, 3, 1, 9223372036854775807, 2);  neg_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_161: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_81, 3, 0, 9223372036854775807, 2);  select_81 = None
        add_502: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_160, slice_scatter_161);  slice_scatter_160 = slice_scatter_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_58: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_105, [1, 128, 1, 32, 2]);  unsqueeze_105 = None
        clone_58: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_206: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [1, 128, 1, 64]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_724: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_308, view_206);  slice_308 = None
        add_503: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_502, mul_724);  add_502 = mul_724 = None
        mul_725: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_310, view_205);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1327: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_725, [1, 128, 16, 32, 2]);  mul_725 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_82: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1327, -1, 0)
        select_83: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1327, -1, 1);  view_1327 = None
        neg_120: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_82);  select_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_162: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_120, 3, 1, 9223372036854775807, 2);  neg_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_163: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_83, 3, 0, 9223372036854775807, 2);  select_83 = None
        add_504: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_162, slice_scatter_163);  slice_scatter_162 = slice_scatter_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_726: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_310, view_206);  slice_310 = view_206 = None
        add_505: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_504, mul_726);  add_504 = mul_726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_164: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_309, 3, 64, 9223372036854775807);  slice_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_165: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_503, 3, 0, 64);  add_503 = None
        add_506: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_164, slice_scatter_165);  slice_scatter_164 = slice_scatter_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_166: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_311, 3, 64, 9223372036854775807);  slice_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_167: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_505, 3, 0, 64);  add_505 = None
        add_507: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_166, slice_scatter_167);  slice_scatter_166 = slice_scatter_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_993: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1321, [0, 2, 1, 3]);  view_1321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_247: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_993, memory_format = torch.contiguous_format);  permute_993 = None
        view_1328: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_247, [1, 128, 4096]);  clone_247 = None
        view_1329: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_507, [1, 128, 4096]);  add_507 = None
        view_1330: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_506, [1, 128, 4096]);  add_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1331: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1328, [128, 4096]);  view_1328 = None
        permute_994: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1331, [1, 0])
        mm_360: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_994, view_196);  permute_994 = None
        mm_361: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1331, permute_996);  view_1331 = permute_996 = None
        view_1332: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_361, [1, 128, 4096]);  mm_361 = None
        convert_element_type_2277: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1332, torch.float32);  view_1332 = None
        add_508: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2251, convert_element_type_2277);  convert_element_type_2251 = convert_element_type_2277 = None
        convert_element_type_2278: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_360, torch.float32);  mm_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1333: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1329, [128, 4096]);  view_1329 = None
        permute_998: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1333, [1, 0])
        mm_362: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_998, view_196);  permute_998 = None
        mm_363: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1333, permute_1000);  view_1333 = permute_1000 = None
        view_1334: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_363, [1, 128, 4096]);  mm_363 = None
        convert_element_type_2283: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1334, torch.float32);  view_1334 = None
        add_509: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_508, convert_element_type_2283);  add_508 = convert_element_type_2283 = None
        convert_element_type_2284: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_362, torch.float32);  mm_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1335: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1330, [128, 4096]);  view_1330 = None
        permute_1002: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1335, [1, 0])
        mm_364: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1002, view_196);  permute_1002 = view_196 = None
        mm_365: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1335, permute_1004);  view_1335 = permute_1004 = None
        view_1336: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_365, [1, 128, 4096]);  mm_365 = None
        convert_element_type_2289: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1336, torch.float32);  view_1336 = None
        add_510: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_509, convert_element_type_2289);  add_509 = convert_element_type_2289 = None
        convert_element_type_2290: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_364, torch.float32);  mm_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_728: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_510, primals_80);  primals_80 = None
        mul_729: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_728, 4096)
        sum_185: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_728, [2], True)
        mul_730: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_728, mul_70);  mul_728 = None
        sum_186: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_730, [2], True);  mul_730 = None
        mul_731: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, sum_186);  sum_186 = None
        sub_150: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_729, sum_185);  mul_729 = sum_185 = None
        sub_151: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_150, mul_731);  sub_150 = mul_731 = None
        mul_732: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_104, sub_151);  div_104 = sub_151 = None
        mul_733: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_510, mul_70);  mul_70 = None
        sum_187: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 1]);  mul_733 = None
        sum_188: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_510, [0, 1]);  add_510 = None
        add_511: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_499, mul_732);  add_499 = mul_732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2291: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_511, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1337: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2291, [128, 4096]);  convert_element_type_2291 = None
        mm_366: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1337, permute_1006);  permute_1006 = None
        permute_1007: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1337, [1, 0])
        mm_367: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_1007, view_194);  view_194 = None
        sum_189: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1337, [0], True, dtype = torch.float32)
        view_1338: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_189, [4096]);  sum_189 = None
        convert_element_type_2296: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1338, torch.bfloat16);  view_1338 = None
        view_1339: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_366, [1, 128, 16384]);  mm_366 = None
        convert_element_type_2297: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1339, torch.float32);  view_1339 = None
        convert_element_type_2298: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_367, torch.float32);  mm_367 = None
        convert_element_type_2299: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2296, torch.float32);  convert_element_type_2296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_193: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [1, 128, 16384]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_66: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_193, 0.5)
        mul_734: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2297, mul_66);  mul_66 = None
        convert_element_type_259: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_193, torch.float32)
        pow_7: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_259, 3.0)
        mul_67: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_62: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_193, mul_67);  view_193 = mul_67 = None
        mul_68: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
        tanh_6: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_68);  mul_68 = None
        add_63: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_735: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2297, add_63);  convert_element_type_2297 = add_63 = None
        convert_element_type_2300: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_735, torch.bfloat16);  mul_735 = None
        mul_736: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_152: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_736);  mul_736 = None
        mul_737: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_734, sub_152);  mul_734 = sub_152 = None
        mul_738: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_737, 0.7978845608028654);  mul_737 = None
        convert_element_type_2301: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_738, torch.bfloat16)
        mul_739: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_738, 0.044715);  mul_738 = None
        pow_50: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_259, 2.0);  convert_element_type_259 = None
        mul_740: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_50, 3.0);  pow_50 = None
        mul_741: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_739, mul_740);  mul_739 = mul_740 = None
        convert_element_type_2302: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_741, torch.bfloat16);  mul_741 = None
        add_512: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2301, convert_element_type_2302);  convert_element_type_2301 = convert_element_type_2302 = None
        mul_742: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2300, 0.5);  convert_element_type_2300 = None
        add_513: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_512, mul_742);  add_512 = mul_742 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1340: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_513, [128, 16384]);  add_513 = None
        mm_368: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1340, permute_1010);  permute_1010 = None
        permute_1011: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1340, [1, 0])
        mm_369: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1011, view_168);  permute_1011 = None
        sum_190: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1340, [0], True, dtype = torch.float32);  view_1340 = None
        view_1341: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_190, [16384]);  sum_190 = None
        convert_element_type_2307: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1341, torch.bfloat16);  view_1341 = None
        view_1342: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_368, [1, 128, 4096]);  mm_368 = None
        convert_element_type_2308: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1342, torch.float32);  view_1342 = None
        convert_element_type_2309: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_369, torch.float32);  mm_369 = None
        convert_element_type_2310: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2307, torch.float32);  convert_element_type_2307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_370: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1007, view_190);  permute_1007 = view_190 = None
        mm_371: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1337, permute_1016);  view_1337 = permute_1016 = None
        view_1344: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_371, [1, 128, 4096]);  mm_371 = None
        convert_element_type_2315: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_370, torch.float32);  mm_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1345: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1344, [1, 128, 16, 256]);  view_1344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1018: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1345, [0, 2, 1, 3]);  view_1345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1346: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1018, [16, 128, 256]);  permute_1018 = None
        bmm_140: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1019, view_1346);  permute_1019 = None
        bmm_141: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1346, permute_1020);  view_1346 = permute_1020 = None
        view_1347: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_140, [1, 16, 128, 256]);  bmm_140 = None
        view_1348: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_141, [1, 16, 128, 128]);  bmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2320: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1348, torch.float32);  view_1348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_743: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2320, div_13);  convert_element_type_2320 = None
        sum_191: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_743, [-1], True)
        neg_121: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_21: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_121, sum_191, mul_743);  neg_121 = sum_191 = mul_743 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2321: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_21, torch.bfloat16);  fma_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_105: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2321, 16.0);  convert_element_type_2321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1349: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_105, [16, 128, 128]);  div_105 = None
        bmm_142: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1021, view_1349);  permute_1021 = None
        bmm_143: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1349, permute_1022);  view_1349 = permute_1022 = None
        view_1350: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_142, [1, 16, 256, 128]);  bmm_142 = None
        view_1351: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_143, [1, 16, 128, 256]);  bmm_143 = None
        convert_element_type_2327: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1350, torch.float32);  view_1350 = None
        permute_1023: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2327, [0, 1, 3, 2]);  convert_element_type_2327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2328: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1023, torch.bfloat16);  permute_1023 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_6: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1351, torch.bfloat16);  view_1351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1024: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_6, [0, 2, 1, 3]);  convert_element_type_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1025: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2328, [0, 2, 1, 3]);  convert_element_type_2328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_312: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1024, 3, 0, 64)
        slice_313: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1024, 3, 64, 256);  permute_1024 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_314: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1025, 3, 0, 64)
        slice_315: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1025, 3, 64, 256);  permute_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_49: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_90, [1, 128, 1, 32, 2]);  unsqueeze_90 = None
        clone_49: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_177: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [1, 128, 1, 64]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_744: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_312, view_177)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1352: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_744, [1, 128, 16, 32, 2]);  mul_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_84: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1352, -1, 0)
        select_85: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1352, -1, 1);  view_1352 = None
        neg_122: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_84);  select_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_168: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_122, 3, 1, 9223372036854775807, 2);  neg_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_169: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_85, 3, 0, 9223372036854775807, 2);  select_85 = None
        add_514: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_168, slice_scatter_169);  slice_scatter_168 = slice_scatter_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_50: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_92, [1, 128, 1, 32, 2]);  unsqueeze_92 = None
        clone_50: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_178: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [1, 128, 1, 64]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_745: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_312, view_178);  slice_312 = None
        add_515: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_514, mul_745);  add_514 = mul_745 = None
        mul_746: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_314, view_177);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1353: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_746, [1, 128, 16, 32, 2]);  mul_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_86: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1353, -1, 0)
        select_87: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1353, -1, 1);  view_1353 = None
        neg_123: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_86);  select_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_170: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_123, 3, 1, 9223372036854775807, 2);  neg_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_171: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_87, 3, 0, 9223372036854775807, 2);  select_87 = None
        add_516: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_170, slice_scatter_171);  slice_scatter_170 = slice_scatter_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_747: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_314, view_178);  slice_314 = view_178 = None
        add_517: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_516, mul_747);  add_516 = mul_747 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_172: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_313, 3, 64, 9223372036854775807);  slice_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_173: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_515, 3, 0, 64);  add_515 = None
        add_518: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_172, slice_scatter_173);  slice_scatter_172 = slice_scatter_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_174: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_315, 3, 64, 9223372036854775807);  slice_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_175: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_517, 3, 0, 64);  add_517 = None
        add_519: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_174, slice_scatter_175);  slice_scatter_174 = slice_scatter_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1026: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1347, [0, 2, 1, 3]);  view_1347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_248: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_1026, memory_format = torch.contiguous_format);  permute_1026 = None
        view_1354: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_248, [1, 128, 4096]);  clone_248 = None
        view_1355: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_519, [1, 128, 4096]);  add_519 = None
        view_1356: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_518, [1, 128, 4096]);  add_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1357: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1354, [128, 4096]);  view_1354 = None
        permute_1027: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1357, [1, 0])
        mm_372: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1027, view_168);  permute_1027 = None
        mm_373: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1357, permute_1029);  view_1357 = permute_1029 = None
        view_1358: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_373, [1, 128, 4096]);  mm_373 = None
        convert_element_type_2334: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1358, torch.float32);  view_1358 = None
        add_520: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2308, convert_element_type_2334);  convert_element_type_2308 = convert_element_type_2334 = None
        convert_element_type_2335: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_372, torch.float32);  mm_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1359: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1355, [128, 4096]);  view_1355 = None
        permute_1031: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1359, [1, 0])
        mm_374: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1031, view_168);  permute_1031 = None
        mm_375: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1359, permute_1033);  view_1359 = permute_1033 = None
        view_1360: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_375, [1, 128, 4096]);  mm_375 = None
        convert_element_type_2340: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1360, torch.float32);  view_1360 = None
        add_521: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_520, convert_element_type_2340);  add_520 = convert_element_type_2340 = None
        convert_element_type_2341: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_374, torch.float32);  mm_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1361: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1356, [128, 4096]);  view_1356 = None
        permute_1035: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1361, [1, 0])
        mm_376: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1035, view_168);  permute_1035 = view_168 = None
        mm_377: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1361, permute_1037);  view_1361 = permute_1037 = None
        view_1362: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_377, [1, 128, 4096]);  mm_377 = None
        convert_element_type_2346: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1362, torch.float32);  view_1362 = None
        add_522: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_521, convert_element_type_2346);  add_521 = convert_element_type_2346 = None
        convert_element_type_2347: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_376, torch.float32);  mm_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_749: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_522, primals_69);  primals_69 = None
        mul_750: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_749, 4096)
        sum_192: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_749, [2], True)
        mul_751: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_749, mul_60);  mul_749 = None
        sum_193: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_751, [2], True);  mul_751 = None
        mul_752: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, sum_193);  sum_193 = None
        sub_154: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_750, sum_192);  mul_750 = sum_192 = None
        sub_155: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_154, mul_752);  sub_154 = mul_752 = None
        mul_753: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_106, sub_155);  div_106 = sub_155 = None
        mul_754: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_522, mul_60);  mul_60 = None
        sum_194: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_754, [0, 1]);  mul_754 = None
        sum_195: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_522, [0, 1]);  add_522 = None
        add_523: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_511, mul_753);  add_511 = mul_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2348: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_523, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1363: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2348, [128, 4096]);  convert_element_type_2348 = None
        mm_378: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1363, permute_1039);  permute_1039 = None
        permute_1040: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1363, [1, 0])
        mm_379: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_1040, view_166);  view_166 = None
        sum_196: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1363, [0], True, dtype = torch.float32)
        view_1364: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_196, [4096]);  sum_196 = None
        convert_element_type_2353: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1364, torch.bfloat16);  view_1364 = None
        view_1365: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_378, [1, 128, 16384]);  mm_378 = None
        convert_element_type_2354: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1365, torch.float32);  view_1365 = None
        convert_element_type_2355: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_379, torch.float32);  mm_379 = None
        convert_element_type_2356: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2353, torch.float32);  convert_element_type_2353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_165: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [1, 128, 16384]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_165, 0.5)
        mul_755: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2354, mul_56);  mul_56 = None
        convert_element_type_221: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_165, torch.float32)
        pow_6: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_221, 3.0)
        mul_57: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_53: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_165, mul_57);  view_165 = mul_57 = None
        mul_58: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_53, 0.7978845608028654);  add_53 = None
        tanh_5: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_54: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_756: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2354, add_54);  convert_element_type_2354 = add_54 = None
        convert_element_type_2357: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_756, torch.bfloat16);  mul_756 = None
        mul_757: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_156: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_757);  mul_757 = None
        mul_758: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_755, sub_156);  mul_755 = sub_156 = None
        mul_759: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_758, 0.7978845608028654);  mul_758 = None
        convert_element_type_2358: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_759, torch.bfloat16)
        mul_760: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_759, 0.044715);  mul_759 = None
        pow_51: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_221, 2.0);  convert_element_type_221 = None
        mul_761: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_51, 3.0);  pow_51 = None
        mul_762: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_760, mul_761);  mul_760 = mul_761 = None
        convert_element_type_2359: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_762, torch.bfloat16);  mul_762 = None
        add_524: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2358, convert_element_type_2359);  convert_element_type_2358 = convert_element_type_2359 = None
        mul_763: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2357, 0.5);  convert_element_type_2357 = None
        add_525: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_524, mul_763);  add_524 = mul_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1366: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_525, [128, 16384]);  add_525 = None
        mm_380: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1366, permute_1043);  permute_1043 = None
        permute_1044: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1366, [1, 0])
        mm_381: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1044, view_140);  permute_1044 = None
        sum_197: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1366, [0], True, dtype = torch.float32);  view_1366 = None
        view_1367: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_197, [16384]);  sum_197 = None
        convert_element_type_2364: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1367, torch.bfloat16);  view_1367 = None
        view_1368: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_380, [1, 128, 4096]);  mm_380 = None
        convert_element_type_2365: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1368, torch.float32);  view_1368 = None
        convert_element_type_2366: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_381, torch.float32);  mm_381 = None
        convert_element_type_2367: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2364, torch.float32);  convert_element_type_2364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_382: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1040, view_162);  permute_1040 = view_162 = None
        mm_383: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1363, permute_1049);  view_1363 = permute_1049 = None
        view_1370: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_383, [1, 128, 4096]);  mm_383 = None
        convert_element_type_2372: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_382, torch.float32);  mm_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1371: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1370, [1, 128, 16, 256]);  view_1370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1051: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1371, [0, 2, 1, 3]);  view_1371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1372: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1051, [16, 128, 256]);  permute_1051 = None
        bmm_144: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1052, view_1372);  permute_1052 = None
        bmm_145: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1372, permute_1053);  view_1372 = permute_1053 = None
        view_1373: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_144, [1, 16, 128, 256]);  bmm_144 = None
        view_1374: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_145, [1, 16, 128, 128]);  bmm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2377: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1374, torch.float32);  view_1374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_764: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2377, div_11);  convert_element_type_2377 = None
        sum_198: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_764, [-1], True)
        neg_124: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_11);  div_11 = None
        fma_22: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_124, sum_198, mul_764);  neg_124 = sum_198 = mul_764 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2378: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_22, torch.bfloat16);  fma_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_107: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2378, 16.0);  convert_element_type_2378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1375: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_107, [16, 128, 128]);  div_107 = None
        bmm_146: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1054, view_1375);  permute_1054 = None
        bmm_147: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1375, permute_1055);  view_1375 = permute_1055 = None
        view_1376: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_146, [1, 16, 256, 128]);  bmm_146 = None
        view_1377: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_147, [1, 16, 128, 256]);  bmm_147 = None
        convert_element_type_2384: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1376, torch.float32);  view_1376 = None
        permute_1056: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2384, [0, 1, 3, 2]);  convert_element_type_2384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2385: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1056, torch.bfloat16);  permute_1056 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_5: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1377, torch.bfloat16);  view_1377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1057: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_5, [0, 2, 1, 3]);  convert_element_type_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1058: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2385, [0, 2, 1, 3]);  convert_element_type_2385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_316: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1057, 3, 0, 64)
        slice_317: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1057, 3, 64, 256);  permute_1057 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_318: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1058, 3, 0, 64)
        slice_319: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1058, 3, 64, 256);  permute_1058 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_41: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_77, [1, 128, 1, 32, 2]);  unsqueeze_77 = None
        clone_41: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_149: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [1, 128, 1, 64]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_765: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_316, view_149)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1378: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_765, [1, 128, 16, 32, 2]);  mul_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_88: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1378, -1, 0)
        select_89: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1378, -1, 1);  view_1378 = None
        neg_125: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_88);  select_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_176: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_125, 3, 1, 9223372036854775807, 2);  neg_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_177: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_89, 3, 0, 9223372036854775807, 2);  select_89 = None
        add_526: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_176, slice_scatter_177);  slice_scatter_176 = slice_scatter_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_42: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_79, [1, 128, 1, 32, 2]);  unsqueeze_79 = None
        clone_42: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_150: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [1, 128, 1, 64]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_766: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_316, view_150);  slice_316 = None
        add_527: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_526, mul_766);  add_526 = mul_766 = None
        mul_767: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_318, view_149);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1379: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_767, [1, 128, 16, 32, 2]);  mul_767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_90: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1379, -1, 0)
        select_91: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1379, -1, 1);  view_1379 = None
        neg_126: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_90);  select_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_178: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_126, 3, 1, 9223372036854775807, 2);  neg_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_179: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_91, 3, 0, 9223372036854775807, 2);  select_91 = None
        add_528: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_178, slice_scatter_179);  slice_scatter_178 = slice_scatter_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_768: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_318, view_150);  slice_318 = view_150 = None
        add_529: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_528, mul_768);  add_528 = mul_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_180: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_317, 3, 64, 9223372036854775807);  slice_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_181: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_527, 3, 0, 64);  add_527 = None
        add_530: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_180, slice_scatter_181);  slice_scatter_180 = slice_scatter_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_182: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_319, 3, 64, 9223372036854775807);  slice_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_183: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_529, 3, 0, 64);  add_529 = None
        add_531: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_182, slice_scatter_183);  slice_scatter_182 = slice_scatter_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1059: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1373, [0, 2, 1, 3]);  view_1373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_249: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_1059, memory_format = torch.contiguous_format);  permute_1059 = None
        view_1380: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_249, [1, 128, 4096]);  clone_249 = None
        view_1381: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_531, [1, 128, 4096]);  add_531 = None
        view_1382: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_530, [1, 128, 4096]);  add_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1383: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1380, [128, 4096]);  view_1380 = None
        permute_1060: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1383, [1, 0])
        mm_384: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1060, view_140);  permute_1060 = None
        mm_385: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1383, permute_1062);  view_1383 = permute_1062 = None
        view_1384: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_385, [1, 128, 4096]);  mm_385 = None
        convert_element_type_2391: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1384, torch.float32);  view_1384 = None
        add_532: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2365, convert_element_type_2391);  convert_element_type_2365 = convert_element_type_2391 = None
        convert_element_type_2392: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_384, torch.float32);  mm_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1385: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1381, [128, 4096]);  view_1381 = None
        permute_1064: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1385, [1, 0])
        mm_386: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1064, view_140);  permute_1064 = None
        mm_387: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1385, permute_1066);  view_1385 = permute_1066 = None
        view_1386: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_387, [1, 128, 4096]);  mm_387 = None
        convert_element_type_2397: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1386, torch.float32);  view_1386 = None
        add_533: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_532, convert_element_type_2397);  add_532 = convert_element_type_2397 = None
        convert_element_type_2398: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_386, torch.float32);  mm_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1387: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1382, [128, 4096]);  view_1382 = None
        permute_1068: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1387, [1, 0])
        mm_388: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1068, view_140);  permute_1068 = view_140 = None
        mm_389: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1387, permute_1070);  view_1387 = permute_1070 = None
        view_1388: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_389, [1, 128, 4096]);  mm_389 = None
        convert_element_type_2403: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1388, torch.float32);  view_1388 = None
        add_534: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_533, convert_element_type_2403);  add_533 = convert_element_type_2403 = None
        convert_element_type_2404: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_388, torch.float32);  mm_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_770: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_534, primals_58);  primals_58 = None
        mul_771: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_770, 4096)
        sum_199: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_770, [2], True)
        mul_772: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_770, mul_50);  mul_770 = None
        sum_200: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_772, [2], True);  mul_772 = None
        mul_773: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, sum_200);  sum_200 = None
        sub_158: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_771, sum_199);  mul_771 = sum_199 = None
        sub_159: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_158, mul_773);  sub_158 = mul_773 = None
        mul_774: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_108, sub_159);  div_108 = sub_159 = None
        mul_775: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_534, mul_50);  mul_50 = None
        sum_201: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_775, [0, 1]);  mul_775 = None
        sum_202: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_534, [0, 1]);  add_534 = None
        add_535: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_523, mul_774);  add_523 = mul_774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2405: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_535, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1389: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2405, [128, 4096]);  convert_element_type_2405 = None
        mm_390: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1389, permute_1072);  permute_1072 = None
        permute_1073: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1389, [1, 0])
        mm_391: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_1073, view_138);  view_138 = None
        sum_203: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1389, [0], True, dtype = torch.float32)
        view_1390: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_203, [4096]);  sum_203 = None
        convert_element_type_2410: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1390, torch.bfloat16);  view_1390 = None
        view_1391: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_390, [1, 128, 16384]);  mm_390 = None
        convert_element_type_2411: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1391, torch.float32);  view_1391 = None
        convert_element_type_2412: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_391, torch.float32);  mm_391 = None
        convert_element_type_2413: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2410, torch.float32);  convert_element_type_2410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_137: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 16384]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_46: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_137, 0.5)
        mul_776: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2411, mul_46);  mul_46 = None
        convert_element_type_183: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_137, torch.float32)
        pow_5: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_183, 3.0)
        mul_47: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_44: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_137, mul_47);  view_137 = mul_47 = None
        mul_48: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_44, 0.7978845608028654);  add_44 = None
        tanh_4: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
        add_45: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_777: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2411, add_45);  convert_element_type_2411 = add_45 = None
        convert_element_type_2414: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_777, torch.bfloat16);  mul_777 = None
        mul_778: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_160: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_778);  mul_778 = None
        mul_779: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_776, sub_160);  mul_776 = sub_160 = None
        mul_780: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_779, 0.7978845608028654);  mul_779 = None
        convert_element_type_2415: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_780, torch.bfloat16)
        mul_781: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_780, 0.044715);  mul_780 = None
        pow_52: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_183, 2.0);  convert_element_type_183 = None
        mul_782: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_52, 3.0);  pow_52 = None
        mul_783: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_781, mul_782);  mul_781 = mul_782 = None
        convert_element_type_2416: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_783, torch.bfloat16);  mul_783 = None
        add_536: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2415, convert_element_type_2416);  convert_element_type_2415 = convert_element_type_2416 = None
        mul_784: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2414, 0.5);  convert_element_type_2414 = None
        add_537: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_536, mul_784);  add_536 = mul_784 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1392: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_537, [128, 16384]);  add_537 = None
        mm_392: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1392, permute_1076);  permute_1076 = None
        permute_1077: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1392, [1, 0])
        mm_393: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1077, view_112);  permute_1077 = None
        sum_204: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1392, [0], True, dtype = torch.float32);  view_1392 = None
        view_1393: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_204, [16384]);  sum_204 = None
        convert_element_type_2421: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1393, torch.bfloat16);  view_1393 = None
        view_1394: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_392, [1, 128, 4096]);  mm_392 = None
        convert_element_type_2422: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1394, torch.float32);  view_1394 = None
        convert_element_type_2423: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_393, torch.float32);  mm_393 = None
        convert_element_type_2424: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2421, torch.float32);  convert_element_type_2421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_394: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1073, view_134);  permute_1073 = view_134 = None
        mm_395: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1389, permute_1082);  view_1389 = permute_1082 = None
        view_1396: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_395, [1, 128, 4096]);  mm_395 = None
        convert_element_type_2429: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_394, torch.float32);  mm_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1397: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1396, [1, 128, 16, 256]);  view_1396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1084: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1397, [0, 2, 1, 3]);  view_1397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1398: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1084, [16, 128, 256]);  permute_1084 = None
        bmm_148: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1085, view_1398);  permute_1085 = None
        bmm_149: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1398, permute_1086);  view_1398 = permute_1086 = None
        view_1399: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_148, [1, 16, 128, 256]);  bmm_148 = None
        view_1400: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_149, [1, 16, 128, 128]);  bmm_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2434: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1400, torch.float32);  view_1400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_785: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2434, div_9);  convert_element_type_2434 = None
        sum_205: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_785, [-1], True)
        neg_127: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_9);  div_9 = None
        fma_23: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_127, sum_205, mul_785);  neg_127 = sum_205 = mul_785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2435: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_23, torch.bfloat16);  fma_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_109: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2435, 16.0);  convert_element_type_2435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1401: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_109, [16, 128, 128]);  div_109 = None
        bmm_150: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1087, view_1401);  permute_1087 = None
        bmm_151: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1401, permute_1088);  view_1401 = permute_1088 = None
        view_1402: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_150, [1, 16, 256, 128]);  bmm_150 = None
        view_1403: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_151, [1, 16, 128, 256]);  bmm_151 = None
        convert_element_type_2441: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1402, torch.float32);  view_1402 = None
        permute_1089: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2441, [0, 1, 3, 2]);  convert_element_type_2441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2442: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1089, torch.bfloat16);  permute_1089 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_4: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1403, torch.bfloat16);  view_1403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1090: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_4, [0, 2, 1, 3]);  convert_element_type_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1091: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2442, [0, 2, 1, 3]);  convert_element_type_2442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_320: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1090, 3, 0, 64)
        slice_321: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1090, 3, 64, 256);  permute_1090 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_322: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1091, 3, 0, 64)
        slice_323: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1091, 3, 64, 256);  permute_1091 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_33: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_64, [1, 128, 1, 32, 2]);  unsqueeze_64 = None
        clone_33: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_121: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [1, 128, 1, 64]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_786: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_320, view_121)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1404: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_786, [1, 128, 16, 32, 2]);  mul_786 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_92: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1404, -1, 0)
        select_93: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1404, -1, 1);  view_1404 = None
        neg_128: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_92);  select_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_184: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_128, 3, 1, 9223372036854775807, 2);  neg_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_185: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_93, 3, 0, 9223372036854775807, 2);  select_93 = None
        add_538: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_184, slice_scatter_185);  slice_scatter_184 = slice_scatter_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_34: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_66, [1, 128, 1, 32, 2]);  unsqueeze_66 = None
        clone_34: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_122: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [1, 128, 1, 64]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_787: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_320, view_122);  slice_320 = None
        add_539: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_538, mul_787);  add_538 = mul_787 = None
        mul_788: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_322, view_121);  view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1405: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_788, [1, 128, 16, 32, 2]);  mul_788 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_94: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1405, -1, 0)
        select_95: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1405, -1, 1);  view_1405 = None
        neg_129: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_94);  select_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_186: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_129, 3, 1, 9223372036854775807, 2);  neg_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_187: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_95, 3, 0, 9223372036854775807, 2);  select_95 = None
        add_540: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_186, slice_scatter_187);  slice_scatter_186 = slice_scatter_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_789: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_322, view_122);  slice_322 = view_122 = None
        add_541: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_540, mul_789);  add_540 = mul_789 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_188: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_321, 3, 64, 9223372036854775807);  slice_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_189: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_539, 3, 0, 64);  add_539 = None
        add_542: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_188, slice_scatter_189);  slice_scatter_188 = slice_scatter_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_190: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_323, 3, 64, 9223372036854775807);  slice_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_191: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_541, 3, 0, 64);  add_541 = None
        add_543: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_190, slice_scatter_191);  slice_scatter_190 = slice_scatter_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1092: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1399, [0, 2, 1, 3]);  view_1399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_250: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_1092, memory_format = torch.contiguous_format);  permute_1092 = None
        view_1406: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_250, [1, 128, 4096]);  clone_250 = None
        view_1407: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_543, [1, 128, 4096]);  add_543 = None
        view_1408: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_542, [1, 128, 4096]);  add_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1409: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1406, [128, 4096]);  view_1406 = None
        permute_1093: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1409, [1, 0])
        mm_396: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1093, view_112);  permute_1093 = None
        mm_397: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1409, permute_1095);  view_1409 = permute_1095 = None
        view_1410: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_397, [1, 128, 4096]);  mm_397 = None
        convert_element_type_2448: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1410, torch.float32);  view_1410 = None
        add_544: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2422, convert_element_type_2448);  convert_element_type_2422 = convert_element_type_2448 = None
        convert_element_type_2449: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_396, torch.float32);  mm_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1411: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1407, [128, 4096]);  view_1407 = None
        permute_1097: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1411, [1, 0])
        mm_398: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1097, view_112);  permute_1097 = None
        mm_399: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1411, permute_1099);  view_1411 = permute_1099 = None
        view_1412: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_399, [1, 128, 4096]);  mm_399 = None
        convert_element_type_2454: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1412, torch.float32);  view_1412 = None
        add_545: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_544, convert_element_type_2454);  add_544 = convert_element_type_2454 = None
        convert_element_type_2455: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_398, torch.float32);  mm_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1413: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1408, [128, 4096]);  view_1408 = None
        permute_1101: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1413, [1, 0])
        mm_400: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1101, view_112);  permute_1101 = view_112 = None
        mm_401: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1413, permute_1103);  view_1413 = permute_1103 = None
        view_1414: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_401, [1, 128, 4096]);  mm_401 = None
        convert_element_type_2460: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1414, torch.float32);  view_1414 = None
        add_546: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_545, convert_element_type_2460);  add_545 = convert_element_type_2460 = None
        convert_element_type_2461: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_400, torch.float32);  mm_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_791: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_546, primals_47);  primals_47 = None
        mul_792: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_791, 4096)
        sum_206: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_791, [2], True)
        mul_793: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_791, mul_40);  mul_791 = None
        sum_207: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_793, [2], True);  mul_793 = None
        mul_794: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, sum_207);  sum_207 = None
        sub_162: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_792, sum_206);  mul_792 = sum_206 = None
        sub_163: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_162, mul_794);  sub_162 = mul_794 = None
        mul_795: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_110, sub_163);  div_110 = sub_163 = None
        mul_796: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_546, mul_40);  mul_40 = None
        sum_208: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_796, [0, 1]);  mul_796 = None
        sum_209: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_546, [0, 1]);  add_546 = None
        add_547: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_535, mul_795);  add_535 = mul_795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2462: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_547, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1415: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2462, [128, 4096]);  convert_element_type_2462 = None
        mm_402: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1415, permute_1105);  permute_1105 = None
        permute_1106: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1415, [1, 0])
        mm_403: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_1106, view_110);  view_110 = None
        sum_210: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1415, [0], True, dtype = torch.float32)
        view_1416: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_210, [4096]);  sum_210 = None
        convert_element_type_2467: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1416, torch.bfloat16);  view_1416 = None
        view_1417: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_402, [1, 128, 16384]);  mm_402 = None
        convert_element_type_2468: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1417, torch.float32);  view_1417 = None
        convert_element_type_2469: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_403, torch.float32);  mm_403 = None
        convert_element_type_2470: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2467, torch.float32);  convert_element_type_2467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_109: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [1, 128, 16384]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_797: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2468, mul_36);  mul_36 = None
        convert_element_type_145: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_109, torch.float32)
        pow_4: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_145, 3.0)
        mul_37: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_35: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_109, mul_37);  view_109 = mul_37 = None
        mul_38: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_35, 0.7978845608028654);  add_35 = None
        tanh_3: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_36: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_798: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2468, add_36);  convert_element_type_2468 = add_36 = None
        convert_element_type_2471: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_798, torch.bfloat16);  mul_798 = None
        mul_799: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_164: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_799);  mul_799 = None
        mul_800: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_797, sub_164);  mul_797 = sub_164 = None
        mul_801: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_800, 0.7978845608028654);  mul_800 = None
        convert_element_type_2472: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_801, torch.bfloat16)
        mul_802: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_801, 0.044715);  mul_801 = None
        pow_53: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_145, 2.0);  convert_element_type_145 = None
        mul_803: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_53, 3.0);  pow_53 = None
        mul_804: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_802, mul_803);  mul_802 = mul_803 = None
        convert_element_type_2473: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_804, torch.bfloat16);  mul_804 = None
        add_548: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2472, convert_element_type_2473);  convert_element_type_2472 = convert_element_type_2473 = None
        mul_805: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2471, 0.5);  convert_element_type_2471 = None
        add_549: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_548, mul_805);  add_548 = mul_805 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1418: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_549, [128, 16384]);  add_549 = None
        mm_404: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1418, permute_1109);  permute_1109 = None
        permute_1110: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1418, [1, 0])
        mm_405: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1110, view_84);  permute_1110 = None
        sum_211: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1418, [0], True, dtype = torch.float32);  view_1418 = None
        view_1419: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_211, [16384]);  sum_211 = None
        convert_element_type_2478: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1419, torch.bfloat16);  view_1419 = None
        view_1420: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_404, [1, 128, 4096]);  mm_404 = None
        convert_element_type_2479: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1420, torch.float32);  view_1420 = None
        convert_element_type_2480: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_405, torch.float32);  mm_405 = None
        convert_element_type_2481: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2478, torch.float32);  convert_element_type_2478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_406: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1106, view_106);  permute_1106 = view_106 = None
        mm_407: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1415, permute_1115);  view_1415 = permute_1115 = None
        view_1422: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_407, [1, 128, 4096]);  mm_407 = None
        convert_element_type_2486: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_406, torch.float32);  mm_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1423: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1422, [1, 128, 16, 256]);  view_1422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1117: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1423, [0, 2, 1, 3]);  view_1423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1424: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1117, [16, 128, 256]);  permute_1117 = None
        bmm_152: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1118, view_1424);  permute_1118 = None
        bmm_153: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1424, permute_1119);  view_1424 = permute_1119 = None
        view_1425: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_152, [1, 16, 128, 256]);  bmm_152 = None
        view_1426: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_153, [1, 16, 128, 128]);  bmm_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2491: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1426, torch.float32);  view_1426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_806: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2491, div_7);  convert_element_type_2491 = None
        sum_212: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_806, [-1], True)
        neg_130: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_24: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_130, sum_212, mul_806);  neg_130 = sum_212 = mul_806 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2492: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_24, torch.bfloat16);  fma_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_111: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2492, 16.0);  convert_element_type_2492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1427: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_111, [16, 128, 128]);  div_111 = None
        bmm_154: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1120, view_1427);  permute_1120 = None
        bmm_155: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1427, permute_1121);  view_1427 = permute_1121 = None
        view_1428: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_154, [1, 16, 256, 128]);  bmm_154 = None
        view_1429: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_155, [1, 16, 128, 256]);  bmm_155 = None
        convert_element_type_2498: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1428, torch.float32);  view_1428 = None
        permute_1122: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2498, [0, 1, 3, 2]);  convert_element_type_2498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2499: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1122, torch.bfloat16);  permute_1122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_3: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1429, torch.bfloat16);  view_1429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1123: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_3, [0, 2, 1, 3]);  convert_element_type_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1124: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2499, [0, 2, 1, 3]);  convert_element_type_2499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_324: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1123, 3, 0, 64)
        slice_325: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1123, 3, 64, 256);  permute_1123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_326: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1124, 3, 0, 64)
        slice_327: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1124, 3, 64, 256);  permute_1124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_25: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_51, [1, 128, 1, 32, 2]);  unsqueeze_51 = None
        clone_25: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_93: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [1, 128, 1, 64]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_807: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_324, view_93)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1430: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_807, [1, 128, 16, 32, 2]);  mul_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_96: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1430, -1, 0)
        select_97: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1430, -1, 1);  view_1430 = None
        neg_131: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_96);  select_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_192: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_131, 3, 1, 9223372036854775807, 2);  neg_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_193: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_97, 3, 0, 9223372036854775807, 2);  select_97 = None
        add_550: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_192, slice_scatter_193);  slice_scatter_192 = slice_scatter_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_26: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_53, [1, 128, 1, 32, 2]);  unsqueeze_53 = None
        clone_26: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_94: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [1, 128, 1, 64]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_808: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_324, view_94);  slice_324 = None
        add_551: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_550, mul_808);  add_550 = mul_808 = None
        mul_809: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_326, view_93);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1431: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_809, [1, 128, 16, 32, 2]);  mul_809 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_98: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1431, -1, 0)
        select_99: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1431, -1, 1);  view_1431 = None
        neg_132: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_98);  select_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_194: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_132, 3, 1, 9223372036854775807, 2);  neg_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_195: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_99, 3, 0, 9223372036854775807, 2);  select_99 = None
        add_552: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_194, slice_scatter_195);  slice_scatter_194 = slice_scatter_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_810: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_326, view_94);  slice_326 = view_94 = None
        add_553: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_552, mul_810);  add_552 = mul_810 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_196: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_325, 3, 64, 9223372036854775807);  slice_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_197: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_551, 3, 0, 64);  add_551 = None
        add_554: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_196, slice_scatter_197);  slice_scatter_196 = slice_scatter_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_198: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_327, 3, 64, 9223372036854775807);  slice_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_199: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_553, 3, 0, 64);  add_553 = None
        add_555: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_198, slice_scatter_199);  slice_scatter_198 = slice_scatter_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1125: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1425, [0, 2, 1, 3]);  view_1425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_251: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_1125, memory_format = torch.contiguous_format);  permute_1125 = None
        view_1432: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_251, [1, 128, 4096]);  clone_251 = None
        view_1433: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_555, [1, 128, 4096]);  add_555 = None
        view_1434: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_554, [1, 128, 4096]);  add_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1435: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1432, [128, 4096]);  view_1432 = None
        permute_1126: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1435, [1, 0])
        mm_408: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1126, view_84);  permute_1126 = None
        mm_409: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1435, permute_1128);  view_1435 = permute_1128 = None
        view_1436: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_409, [1, 128, 4096]);  mm_409 = None
        convert_element_type_2505: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1436, torch.float32);  view_1436 = None
        add_556: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2479, convert_element_type_2505);  convert_element_type_2479 = convert_element_type_2505 = None
        convert_element_type_2506: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_408, torch.float32);  mm_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1437: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1433, [128, 4096]);  view_1433 = None
        permute_1130: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1437, [1, 0])
        mm_410: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1130, view_84);  permute_1130 = None
        mm_411: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1437, permute_1132);  view_1437 = permute_1132 = None
        view_1438: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_411, [1, 128, 4096]);  mm_411 = None
        convert_element_type_2511: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1438, torch.float32);  view_1438 = None
        add_557: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_556, convert_element_type_2511);  add_556 = convert_element_type_2511 = None
        convert_element_type_2512: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_410, torch.float32);  mm_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1439: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1434, [128, 4096]);  view_1434 = None
        permute_1134: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1439, [1, 0])
        mm_412: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1134, view_84);  permute_1134 = view_84 = None
        mm_413: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1439, permute_1136);  view_1439 = permute_1136 = None
        view_1440: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_413, [1, 128, 4096]);  mm_413 = None
        convert_element_type_2517: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1440, torch.float32);  view_1440 = None
        add_558: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_557, convert_element_type_2517);  add_557 = convert_element_type_2517 = None
        convert_element_type_2518: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_412, torch.float32);  mm_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_812: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_558, primals_36);  primals_36 = None
        mul_813: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_812, 4096)
        sum_213: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_812, [2], True)
        mul_814: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_812, mul_30);  mul_812 = None
        sum_214: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_814, [2], True);  mul_814 = None
        mul_815: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, sum_214);  sum_214 = None
        sub_166: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_813, sum_213);  mul_813 = sum_213 = None
        sub_167: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_166, mul_815);  sub_166 = mul_815 = None
        mul_816: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_112, sub_167);  div_112 = sub_167 = None
        mul_817: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_558, mul_30);  mul_30 = None
        sum_215: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_817, [0, 1]);  mul_817 = None
        sum_216: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_558, [0, 1]);  add_558 = None
        add_559: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_547, mul_816);  add_547 = mul_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2519: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_559, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1441: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2519, [128, 4096]);  convert_element_type_2519 = None
        mm_414: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1441, permute_1138);  permute_1138 = None
        permute_1139: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1441, [1, 0])
        mm_415: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_1139, view_82);  view_82 = None
        sum_217: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1441, [0], True, dtype = torch.float32)
        view_1442: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_217, [4096]);  sum_217 = None
        convert_element_type_2524: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1442, torch.bfloat16);  view_1442 = None
        view_1443: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_414, [1, 128, 16384]);  mm_414 = None
        convert_element_type_2525: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1443, torch.float32);  view_1443 = None
        convert_element_type_2526: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_415, torch.float32);  mm_415 = None
        convert_element_type_2527: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2524, torch.float32);  convert_element_type_2524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_81: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 16384]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_26: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_81, 0.5)
        mul_818: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2525, mul_26);  mul_26 = None
        convert_element_type_107: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_81, torch.float32)
        pow_3: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_107, 3.0)
        mul_27: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_26: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_81, mul_27);  view_81 = mul_27 = None
        mul_28: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_26, 0.7978845608028654);  add_26 = None
        tanh_2: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_28);  mul_28 = None
        add_27: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_819: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2525, add_27);  convert_element_type_2525 = add_27 = None
        convert_element_type_2528: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_819, torch.bfloat16);  mul_819 = None
        mul_820: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_168: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_820);  mul_820 = None
        mul_821: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_818, sub_168);  mul_818 = sub_168 = None
        mul_822: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_821, 0.7978845608028654);  mul_821 = None
        convert_element_type_2529: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_822, torch.bfloat16)
        mul_823: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_822, 0.044715);  mul_822 = None
        pow_54: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_107, 2.0);  convert_element_type_107 = None
        mul_824: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_54, 3.0);  pow_54 = None
        mul_825: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_823, mul_824);  mul_823 = mul_824 = None
        convert_element_type_2530: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_825, torch.bfloat16);  mul_825 = None
        add_560: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2529, convert_element_type_2530);  convert_element_type_2529 = convert_element_type_2530 = None
        mul_826: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2528, 0.5);  convert_element_type_2528 = None
        add_561: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_560, mul_826);  add_560 = mul_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1444: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_561, [128, 16384]);  add_561 = None
        mm_416: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1444, permute_1142);  permute_1142 = None
        permute_1143: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1444, [1, 0])
        mm_417: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1143, view_56);  permute_1143 = None
        sum_218: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1444, [0], True, dtype = torch.float32);  view_1444 = None
        view_1445: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_218, [16384]);  sum_218 = None
        convert_element_type_2535: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1445, torch.bfloat16);  view_1445 = None
        view_1446: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_416, [1, 128, 4096]);  mm_416 = None
        convert_element_type_2536: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1446, torch.float32);  view_1446 = None
        convert_element_type_2537: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_417, torch.float32);  mm_417 = None
        convert_element_type_2538: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2535, torch.float32);  convert_element_type_2535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_418: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1139, view_78);  permute_1139 = view_78 = None
        mm_419: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1441, permute_1148);  view_1441 = permute_1148 = None
        view_1448: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_419, [1, 128, 4096]);  mm_419 = None
        convert_element_type_2543: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_418, torch.float32);  mm_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1449: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1448, [1, 128, 16, 256]);  view_1448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1150: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1449, [0, 2, 1, 3]);  view_1449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1450: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1150, [16, 128, 256]);  permute_1150 = None
        bmm_156: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1151, view_1450);  permute_1151 = None
        bmm_157: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1450, permute_1152);  view_1450 = permute_1152 = None
        view_1451: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_156, [1, 16, 128, 256]);  bmm_156 = None
        view_1452: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_157, [1, 16, 128, 128]);  bmm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2548: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1452, torch.float32);  view_1452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_827: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2548, div_5);  convert_element_type_2548 = None
        sum_219: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_827, [-1], True)
        neg_133: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_25: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_133, sum_219, mul_827);  neg_133 = sum_219 = mul_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2549: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_25, torch.bfloat16);  fma_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_113: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2549, 16.0);  convert_element_type_2549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1453: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_113, [16, 128, 128]);  div_113 = None
        bmm_158: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1153, view_1453);  permute_1153 = None
        bmm_159: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1453, permute_1154);  view_1453 = permute_1154 = None
        view_1454: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_158, [1, 16, 256, 128]);  bmm_158 = None
        view_1455: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_159, [1, 16, 128, 256]);  bmm_159 = None
        convert_element_type_2555: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1454, torch.float32);  view_1454 = None
        permute_1155: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2555, [0, 1, 3, 2]);  convert_element_type_2555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2556: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1155, torch.bfloat16);  permute_1155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_2: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1455, torch.bfloat16);  view_1455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1156: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_2, [0, 2, 1, 3]);  convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1157: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2556, [0, 2, 1, 3]);  convert_element_type_2556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_328: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1156, 3, 0, 64)
        slice_329: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1156, 3, 64, 256);  permute_1156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_330: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1157, 3, 0, 64)
        slice_331: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1157, 3, 64, 256);  permute_1157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_17: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_38, [1, 128, 1, 32, 2]);  unsqueeze_38 = None
        clone_17: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_65: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [1, 128, 1, 64]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_828: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_328, view_65)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1456: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_828, [1, 128, 16, 32, 2]);  mul_828 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_100: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1456, -1, 0)
        select_101: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1456, -1, 1);  view_1456 = None
        neg_134: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_100);  select_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_200: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_134, 3, 1, 9223372036854775807, 2);  neg_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_201: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_101, 3, 0, 9223372036854775807, 2);  select_101 = None
        add_562: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_200, slice_scatter_201);  slice_scatter_200 = slice_scatter_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_18: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_40, [1, 128, 1, 32, 2]);  unsqueeze_40 = None
        clone_18: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_66: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [1, 128, 1, 64]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_829: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_328, view_66);  slice_328 = None
        add_563: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_562, mul_829);  add_562 = mul_829 = None
        mul_830: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_330, view_65);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1457: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_830, [1, 128, 16, 32, 2]);  mul_830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_102: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1457, -1, 0)
        select_103: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1457, -1, 1);  view_1457 = None
        neg_135: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_102);  select_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_202: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_135, 3, 1, 9223372036854775807, 2);  neg_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_203: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_103, 3, 0, 9223372036854775807, 2);  select_103 = None
        add_564: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_202, slice_scatter_203);  slice_scatter_202 = slice_scatter_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_831: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_330, view_66);  slice_330 = view_66 = None
        add_565: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_564, mul_831);  add_564 = mul_831 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_204: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_329, 3, 64, 9223372036854775807);  slice_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_205: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_563, 3, 0, 64);  add_563 = None
        add_566: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_204, slice_scatter_205);  slice_scatter_204 = slice_scatter_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_206: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_331, 3, 64, 9223372036854775807);  slice_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_207: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_565, 3, 0, 64);  add_565 = None
        add_567: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_206, slice_scatter_207);  slice_scatter_206 = slice_scatter_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1158: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1451, [0, 2, 1, 3]);  view_1451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_252: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_1158, memory_format = torch.contiguous_format);  permute_1158 = None
        view_1458: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_252, [1, 128, 4096]);  clone_252 = None
        view_1459: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_567, [1, 128, 4096]);  add_567 = None
        view_1460: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_566, [1, 128, 4096]);  add_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1461: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1458, [128, 4096]);  view_1458 = None
        permute_1159: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1461, [1, 0])
        mm_420: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1159, view_56);  permute_1159 = None
        mm_421: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1461, permute_1161);  view_1461 = permute_1161 = None
        view_1462: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_421, [1, 128, 4096]);  mm_421 = None
        convert_element_type_2562: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1462, torch.float32);  view_1462 = None
        add_568: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2536, convert_element_type_2562);  convert_element_type_2536 = convert_element_type_2562 = None
        convert_element_type_2563: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_420, torch.float32);  mm_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1463: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1459, [128, 4096]);  view_1459 = None
        permute_1163: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1463, [1, 0])
        mm_422: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1163, view_56);  permute_1163 = None
        mm_423: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1463, permute_1165);  view_1463 = permute_1165 = None
        view_1464: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_423, [1, 128, 4096]);  mm_423 = None
        convert_element_type_2568: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1464, torch.float32);  view_1464 = None
        add_569: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_568, convert_element_type_2568);  add_568 = convert_element_type_2568 = None
        convert_element_type_2569: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_422, torch.float32);  mm_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1465: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1460, [128, 4096]);  view_1460 = None
        permute_1167: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1465, [1, 0])
        mm_424: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1167, view_56);  permute_1167 = view_56 = None
        mm_425: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1465, permute_1169);  view_1465 = permute_1169 = None
        view_1466: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_425, [1, 128, 4096]);  mm_425 = None
        convert_element_type_2574: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1466, torch.float32);  view_1466 = None
        add_570: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_569, convert_element_type_2574);  add_569 = convert_element_type_2574 = None
        convert_element_type_2575: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_424, torch.float32);  mm_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_833: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_570, primals_25);  primals_25 = None
        mul_834: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_833, 4096)
        sum_220: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_833, [2], True)
        mul_835: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_833, mul_20);  mul_833 = None
        sum_221: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_835, [2], True);  mul_835 = None
        mul_836: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, sum_221);  sum_221 = None
        sub_170: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_834, sum_220);  mul_834 = sum_220 = None
        sub_171: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_170, mul_836);  sub_170 = mul_836 = None
        mul_837: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_114, sub_171);  div_114 = sub_171 = None
        mul_838: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_570, mul_20);  mul_20 = None
        sum_222: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_838, [0, 1]);  mul_838 = None
        sum_223: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_570, [0, 1]);  add_570 = None
        add_571: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_559, mul_837);  add_559 = mul_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2576: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_571, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1467: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2576, [128, 4096]);  convert_element_type_2576 = None
        mm_426: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1467, permute_1171);  permute_1171 = None
        permute_1172: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1467, [1, 0])
        mm_427: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_1172, view_54);  view_54 = None
        sum_224: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1467, [0], True, dtype = torch.float32)
        view_1468: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_224, [4096]);  sum_224 = None
        convert_element_type_2581: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1468, torch.bfloat16);  view_1468 = None
        view_1469: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_426, [1, 128, 16384]);  mm_426 = None
        convert_element_type_2582: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1469, torch.float32);  view_1469 = None
        convert_element_type_2583: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_427, torch.float32);  mm_427 = None
        convert_element_type_2584: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2581, torch.float32);  convert_element_type_2581 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_53: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [1, 128, 16384]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_16: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_53, 0.5)
        mul_839: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2582, mul_16);  mul_16 = None
        convert_element_type_69: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_53, torch.float32)
        pow_2: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_69, 3.0)
        mul_17: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_53, mul_17);  view_53 = mul_17 = None
        mul_18: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
        add_18: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_840: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2582, add_18);  convert_element_type_2582 = add_18 = None
        convert_element_type_2585: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_840, torch.bfloat16);  mul_840 = None
        mul_841: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_172: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_841);  mul_841 = None
        mul_842: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_839, sub_172);  mul_839 = sub_172 = None
        mul_843: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_842, 0.7978845608028654);  mul_842 = None
        convert_element_type_2586: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_843, torch.bfloat16)
        mul_844: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_843, 0.044715);  mul_843 = None
        pow_55: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_69, 2.0);  convert_element_type_69 = None
        mul_845: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_55, 3.0);  pow_55 = None
        mul_846: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_844, mul_845);  mul_844 = mul_845 = None
        convert_element_type_2587: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_846, torch.bfloat16);  mul_846 = None
        add_572: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2586, convert_element_type_2587);  convert_element_type_2586 = convert_element_type_2587 = None
        mul_847: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2585, 0.5);  convert_element_type_2585 = None
        add_573: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_572, mul_847);  add_572 = mul_847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1470: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_573, [128, 16384]);  add_573 = None
        mm_428: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1470, permute_1175);  permute_1175 = None
        permute_1176: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1470, [1, 0])
        mm_429: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1176, view_28);  permute_1176 = None
        sum_225: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1470, [0], True, dtype = torch.float32);  view_1470 = None
        view_1471: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_225, [16384]);  sum_225 = None
        convert_element_type_2592: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1471, torch.bfloat16);  view_1471 = None
        view_1472: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_428, [1, 128, 4096]);  mm_428 = None
        convert_element_type_2593: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1472, torch.float32);  view_1472 = None
        convert_element_type_2594: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_429, torch.float32);  mm_429 = None
        convert_element_type_2595: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2592, torch.float32);  convert_element_type_2592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_430: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1172, view_50);  permute_1172 = view_50 = None
        mm_431: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1467, permute_1181);  view_1467 = permute_1181 = None
        view_1474: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_431, [1, 128, 4096]);  mm_431 = None
        convert_element_type_2600: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_430, torch.float32);  mm_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1475: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1474, [1, 128, 16, 256]);  view_1474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1183: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1475, [0, 2, 1, 3]);  view_1475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1476: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1183, [16, 128, 256]);  permute_1183 = None
        bmm_160: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1184, view_1476);  permute_1184 = None
        bmm_161: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1476, permute_1185);  view_1476 = permute_1185 = None
        view_1477: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_160, [1, 16, 128, 256]);  bmm_160 = None
        view_1478: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_161, [1, 16, 128, 128]);  bmm_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2605: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1478, torch.float32);  view_1478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_848: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2605, div_3);  convert_element_type_2605 = None
        sum_226: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_848, [-1], True)
        neg_136: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_26: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_136, sum_226, mul_848);  neg_136 = sum_226 = mul_848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2606: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_26, torch.bfloat16);  fma_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_115: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2606, 16.0);  convert_element_type_2606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1479: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_115, [16, 128, 128]);  div_115 = None
        bmm_162: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1186, view_1479);  permute_1186 = None
        bmm_163: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1479, permute_1187);  view_1479 = permute_1187 = None
        view_1480: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_162, [1, 16, 256, 128]);  bmm_162 = None
        view_1481: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_163, [1, 16, 128, 256]);  bmm_163 = None
        convert_element_type_2612: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1480, torch.float32);  view_1480 = None
        permute_1188: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2612, [0, 1, 3, 2]);  convert_element_type_2612 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2613: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1188, torch.bfloat16);  permute_1188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_1: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1481, torch.bfloat16);  view_1481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1189: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_1, [0, 2, 1, 3]);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1190: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2613, [0, 2, 1, 3]);  convert_element_type_2613 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_332: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1189, 3, 0, 64)
        slice_333: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1189, 3, 64, 256);  permute_1189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_334: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1190, 3, 0, 64)
        slice_335: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1190, 3, 64, 256);  permute_1190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_9: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_25, [1, 128, 1, 32, 2]);  unsqueeze_25 = None
        clone_9: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_37: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [1, 128, 1, 64]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_849: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_332, view_37)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1482: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_849, [1, 128, 16, 32, 2]);  mul_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_104: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1482, -1, 0)
        select_105: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1482, -1, 1);  view_1482 = None
        neg_137: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_104);  select_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_208: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_137, 3, 1, 9223372036854775807, 2);  neg_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_209: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_105, 3, 0, 9223372036854775807, 2);  select_105 = None
        add_574: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_208, slice_scatter_209);  slice_scatter_208 = slice_scatter_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_10: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_27, [1, 128, 1, 32, 2]);  unsqueeze_27 = None
        clone_10: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_38: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [1, 128, 1, 64]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_850: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_332, view_38);  slice_332 = None
        add_575: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_574, mul_850);  add_574 = mul_850 = None
        mul_851: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_334, view_37);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1483: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_851, [1, 128, 16, 32, 2]);  mul_851 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_106: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1483, -1, 0)
        select_107: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1483, -1, 1);  view_1483 = None
        neg_138: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_106);  select_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_210: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_138, 3, 1, 9223372036854775807, 2);  neg_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_211: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_107, 3, 0, 9223372036854775807, 2);  select_107 = None
        add_576: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_210, slice_scatter_211);  slice_scatter_210 = slice_scatter_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_852: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_334, view_38);  slice_334 = view_38 = None
        add_577: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_576, mul_852);  add_576 = mul_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_212: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_333, 3, 64, 9223372036854775807);  slice_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_213: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_575, 3, 0, 64);  add_575 = None
        add_578: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_212, slice_scatter_213);  slice_scatter_212 = slice_scatter_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_214: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_335, 3, 64, 9223372036854775807);  slice_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_215: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_577, 3, 0, 64);  add_577 = None
        add_579: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_214, slice_scatter_215);  slice_scatter_214 = slice_scatter_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1191: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1477, [0, 2, 1, 3]);  view_1477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_253: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_1191, memory_format = torch.contiguous_format);  permute_1191 = None
        view_1484: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_253, [1, 128, 4096]);  clone_253 = None
        view_1485: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_579, [1, 128, 4096]);  add_579 = None
        view_1486: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_578, [1, 128, 4096]);  add_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1487: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1484, [128, 4096]);  view_1484 = None
        permute_1192: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1487, [1, 0])
        mm_432: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1192, view_28);  permute_1192 = None
        mm_433: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1487, permute_1194);  view_1487 = permute_1194 = None
        view_1488: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_433, [1, 128, 4096]);  mm_433 = None
        convert_element_type_2619: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1488, torch.float32);  view_1488 = None
        add_580: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2593, convert_element_type_2619);  convert_element_type_2593 = convert_element_type_2619 = None
        convert_element_type_2620: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_432, torch.float32);  mm_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1489: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1485, [128, 4096]);  view_1485 = None
        permute_1196: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1489, [1, 0])
        mm_434: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1196, view_28);  permute_1196 = None
        mm_435: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1489, permute_1198);  view_1489 = permute_1198 = None
        view_1490: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_435, [1, 128, 4096]);  mm_435 = None
        convert_element_type_2625: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1490, torch.float32);  view_1490 = None
        add_581: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_580, convert_element_type_2625);  add_580 = convert_element_type_2625 = None
        convert_element_type_2626: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_434, torch.float32);  mm_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1491: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1486, [128, 4096]);  view_1486 = None
        permute_1200: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1491, [1, 0])
        mm_436: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1200, view_28);  permute_1200 = view_28 = None
        mm_437: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1491, permute_1202);  view_1491 = permute_1202 = None
        view_1492: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_437, [1, 128, 4096]);  mm_437 = None
        convert_element_type_2631: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1492, torch.float32);  view_1492 = None
        add_582: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_581, convert_element_type_2631);  add_581 = convert_element_type_2631 = None
        convert_element_type_2632: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_436, torch.float32);  mm_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_854: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_582, primals_14);  primals_14 = None
        mul_855: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_854, 4096)
        sum_227: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_854, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        add_11: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_10, embedding);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_4: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_11, getitem_5);  add_11 = getitem_5 = None
        mul_10: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = None
        mul_856: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_854, mul_10);  mul_854 = None
        sum_228: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_856, [2], True);  mul_856 = None
        mul_857: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, sum_228);  sum_228 = None
        sub_174: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_855, sum_227);  mul_855 = sum_227 = None
        sub_175: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_174, mul_857);  sub_174 = mul_857 = None
        div_116: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 4096);  rsqrt_1 = None
        mul_858: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_116, sub_175);  div_116 = sub_175 = None
        mul_859: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_582, mul_10);  mul_10 = None
        sum_229: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_859, [0, 1]);  mul_859 = None
        sum_230: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_582, [0, 1]);  add_582 = None
        add_583: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_571, mul_858);  add_571 = mul_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:411 in forward, code: hidden_states = attn_outputs + feed_forward_hidden_states + residual
        convert_element_type_2633: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_583, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1493: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2633, [128, 4096]);  convert_element_type_2633 = None
        mm_438: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(view_1493, permute_1204);  permute_1204 = None
        permute_1205: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1493, [1, 0])
        mm_439: "bf16[4096, 16384][16384, 1]cuda:0" = torch.ops.aten.mm.default(permute_1205, view_26);  view_26 = None
        sum_231: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1493, [0], True, dtype = torch.float32)
        view_1494: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_231, [4096]);  sum_231 = None
        convert_element_type_2638: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1494, torch.bfloat16);  view_1494 = None
        view_1495: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_438, [1, 128, 16384]);  mm_438 = None
        convert_element_type_2639: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1495, torch.float32);  view_1495 = None
        convert_element_type_2640: "f32[4096, 16384][16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_439, torch.float32);  mm_439 = None
        convert_element_type_2641: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2638, torch.float32);  convert_element_type_2638 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_25: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [1, 128, 16384]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_6: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_25, 0.5)
        mul_860: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2639, mul_6);  mul_6 = None
        convert_element_type_31: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_25, torch.float32)
        pow_1: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_31, 3.0)
        mul_7: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_8: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(view_25, mul_7);  view_25 = mul_7 = None
        mul_8: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_8, 0.7978845608028654);  add_8 = None
        tanh: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
        add_9: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_861: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2639, add_9);  convert_element_type_2639 = add_9 = None
        convert_element_type_2642: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_861, torch.bfloat16);  mul_861 = None
        mul_862: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_176: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_862);  mul_862 = None
        mul_863: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_860, sub_176);  mul_860 = sub_176 = None
        mul_864: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_863, 0.7978845608028654);  mul_863 = None
        convert_element_type_2643: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_864, torch.bfloat16)
        mul_865: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_864, 0.044715);  mul_864 = None
        pow_56: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_31, 2.0);  convert_element_type_31 = None
        mul_866: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_56, 3.0);  pow_56 = None
        mul_867: "f32[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_865, mul_866);  mul_865 = mul_866 = None
        convert_element_type_2644: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_867, torch.bfloat16);  mul_867 = None
        add_584: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2643, convert_element_type_2644);  convert_element_type_2643 = convert_element_type_2644 = None
        mul_868: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2642, 0.5);  convert_element_type_2642 = None
        add_585: "bf16[1, 128, 16384][2097152, 16384, 1]cuda:0" = torch.ops.aten.add.Tensor(add_584, mul_868);  add_584 = mul_868 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1496: "bf16[128, 16384][16384, 1]cuda:0" = torch.ops.aten.reshape.default(add_585, [128, 16384]);  add_585 = None
        mm_440: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1496, permute_1208);  permute_1208 = None
        permute_1209: "bf16[16384, 128][1, 16384]cuda:0" = torch.ops.aten.permute.default(view_1496, [1, 0])
        mm_441: "bf16[16384, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1209, view);  permute_1209 = None
        sum_232: "f32[1, 16384][16384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1496, [0], True, dtype = torch.float32);  view_1496 = None
        view_1497: "f32[16384][1]cuda:0" = torch.ops.aten.reshape.default(sum_232, [16384]);  sum_232 = None
        convert_element_type_2649: "bf16[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1497, torch.bfloat16);  view_1497 = None
        view_1498: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_440, [1, 128, 4096]);  mm_440 = None
        convert_element_type_2650: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1498, torch.float32);  view_1498 = None
        convert_element_type_2651: "f32[16384, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_441, torch.float32);  mm_441 = None
        convert_element_type_2652: "f32[16384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2649, torch.float32);  convert_element_type_2649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_442: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1205, view_22);  permute_1205 = view_22 = None
        mm_443: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1493, permute_1214);  view_1493 = permute_1214 = None
        view_1500: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_443, [1, 128, 4096]);  mm_443 = None
        convert_element_type_2657: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_442, torch.float32);  mm_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1501: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1500, [1, 128, 16, 256]);  view_1500 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1216: "bf16[1, 16, 128, 256][524288, 256, 4096, 1]cuda:0" = torch.ops.aten.permute.default(view_1501, [0, 2, 1, 3]);  view_1501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        view_1502: "bf16[16, 128, 256][256, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1216, [16, 128, 256]);  permute_1216 = None
        bmm_164: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1217, view_1502);  permute_1217 = None
        bmm_165: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1502, permute_1218);  view_1502 = permute_1218 = None
        view_1503: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_164, [1, 16, 128, 256]);  bmm_164 = None
        view_1504: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_165, [1, 16, 128, 128]);  bmm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:154 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2662: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1504, torch.float32);  view_1504 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:153 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_869: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2662, div_1);  convert_element_type_2662 = None
        sum_233: "f32[1, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_869, [-1], True)
        neg_139: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_27: "f32[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_139, sum_233, mul_869);  neg_139 = sum_233 = mul_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:151 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2663: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_27, torch.bfloat16);  fma_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:148 in _attn, code: attn_weights = attn_weights / self.scale_attn
        div_117: "bf16[1, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_2663, 16.0);  convert_element_type_2663 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1505: "bf16[16, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_117, [16, 128, 128]);  div_117 = None
        bmm_166: "bf16[16, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1219, view_1505);  permute_1219 = None
        bmm_167: "bf16[16, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.bmm.default(view_1505, permute_1220);  view_1505 = permute_1220 = None
        view_1506: "bf16[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_166, [1, 16, 256, 128]);  bmm_166 = None
        view_1507: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_167, [1, 16, 128, 256]);  bmm_167 = None
        convert_element_type_2669: "f32[1, 16, 256, 128][524288, 32768, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1506, torch.float32);  view_1506 = None
        permute_1221: "f32[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2669, [0, 1, 3, 2]);  convert_element_type_2669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:145 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2670: "bf16[1, 16, 128, 256][524288, 32768, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1221, torch.bfloat16);  permute_1221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:144 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default: "bf16[1, 16, 128, 256][524288, 32768, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1507, torch.bfloat16);  view_1507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1222: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default, [0, 2, 1, 3]);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1223: "bf16[1, 128, 16, 256][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2670, [0, 2, 1, 3]);  convert_element_type_2670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_336: "bf16[1, 128, 16, 64][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1222, 3, 0, 64)
        slice_337: "bf16[1, 128, 16, 192][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1222, 3, 64, 256);  permute_1222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_338: "bf16[1, 128, 16, 64][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1223, 3, 0, 64)
        slice_339: "bf16[1, 128, 16, 192][524288, 1, 32768, 128]cuda:0" = torch.ops.aten.slice.Tensor(permute_1223, 3, 64, 256);  permute_1223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_1: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_12, [1, 128, 1, 32, 2]);  unsqueeze_12 = None
        clone_1: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_9: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [1, 128, 1, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_870: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_336, view_9)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1508: "bf16[1, 128, 16, 32, 2][131072, 64, 8192, 2, 1]cuda:0" = torch.ops.aten.reshape.default(mul_870, [1, 128, 16, 32, 2]);  mul_870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_108: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1508, -1, 0)
        select_109: "bf16[1, 128, 16, 32][131072, 64, 8192, 2]cuda:0" = torch.ops.aten.select.int(view_1508, -1, 1);  view_1508 = None
        neg_140: "bf16[1, 128, 16, 32][65536, 32, 4096, 1]cuda:0" = torch.ops.aten.neg.default(select_108);  select_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_216: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_140, 3, 1, 9223372036854775807, 2);  neg_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_217: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_109, 3, 0, 9223372036854775807, 2);  select_109 = None
        add_586: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_216, slice_scatter_217);  slice_scatter_216 = slice_scatter_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_2: "bf16[1, 128, 1, 32, 2][8192, 64, 32, 1, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_14, [1, 128, 1, 32, 2]);  unsqueeze_14 = None
        clone_2: "bf16[1, 128, 1, 32, 2][8192, 64, 64, 2, 1]cuda:0" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_10: "bf16[1, 128, 1, 64][8192, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [1, 128, 1, 64]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_871: "bf16[1, 128, 16, 64][131072, 64, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(slice_336, view_10);  slice_336 = None
        add_587: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_586, mul_871);  add_586 = mul_871 = None
        mul_872: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_338, view_9);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1509: "bf16[1, 128, 16, 32, 2][131072, 1, 8192, 256, 128]cuda:0" = torch.ops.aten.reshape.default(mul_872, [1, 128, 16, 32, 2]);  mul_872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_110: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1509, -1, 0)
        select_111: "bf16[1, 128, 16, 32][131072, 1, 8192, 256]cuda:0" = torch.ops.aten.select.int(view_1509, -1, 1);  view_1509 = None
        neg_141: "bf16[1, 128, 16, 32][65536, 1, 4096, 128]cuda:0" = torch.ops.aten.neg.default(select_110);  select_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_218: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, neg_141, 3, 1, 9223372036854775807, 2);  neg_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_219: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_12, select_111, 3, 0, 9223372036854775807, 2);  full_default_12 = select_111 = None
        add_588: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_218, slice_scatter_219);  slice_scatter_218 = slice_scatter_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_873: "bf16[1, 128, 16, 64][131072, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_338, view_10);  slice_338 = view_10 = None
        add_589: "bf16[1, 128, 16, 64][131072, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(add_588, mul_873);  add_588 = mul_873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_220: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_337, 3, 64, 9223372036854775807);  slice_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_221: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_587, 3, 0, 64);  add_587 = None
        add_590: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_220, slice_scatter_221);  slice_scatter_220 = slice_scatter_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_222: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, slice_339, 3, 64, 9223372036854775807);  slice_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_223: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_16, add_589, 3, 0, 64);  full_default_16 = add_589 = None
        add_591: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_scatter_222, slice_scatter_223);  slice_scatter_222 = slice_scatter_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1224: "bf16[1, 128, 16, 256][524288, 256, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1503, [0, 2, 1, 3]);  view_1503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_254: "bf16[1, 128, 16, 256][524288, 4096, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_1224, memory_format = torch.contiguous_format);  permute_1224 = None
        view_1510: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(clone_254, [1, 128, 4096]);  clone_254 = None
        view_1511: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_591, [1, 128, 4096]);  add_591 = None
        view_1512: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(add_590, [1, 128, 4096]);  add_590 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1513: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1510, [128, 4096]);  view_1510 = None
        permute_1225: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1513, [1, 0])
        mm_444: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1225, view);  permute_1225 = None
        mm_445: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1513, permute_1227);  view_1513 = permute_1227 = None
        view_1514: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_445, [1, 128, 4096]);  mm_445 = None
        convert_element_type_2676: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1514, torch.float32);  view_1514 = None
        add_592: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2650, convert_element_type_2676);  convert_element_type_2650 = convert_element_type_2676 = None
        convert_element_type_2677: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_444, torch.float32);  mm_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1515: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1511, [128, 4096]);  view_1511 = None
        permute_1229: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1515, [1, 0])
        mm_446: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1229, view);  permute_1229 = None
        mm_447: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1515, permute_1231);  view_1515 = permute_1231 = None
        view_1516: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_447, [1, 128, 4096]);  mm_447 = None
        convert_element_type_2682: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1516, torch.float32);  view_1516 = None
        add_593: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_592, convert_element_type_2682);  add_592 = convert_element_type_2682 = None
        convert_element_type_2683: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_446, torch.float32);  mm_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1517: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(view_1512, [128, 4096]);  view_1512 = None
        permute_1233: "bf16[4096, 128][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1517, [1, 0])
        mm_448: "bf16[4096, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1233, view);  permute_1233 = view = None
        mm_449: "bf16[128, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1517, permute_1235);  view_1517 = permute_1235 = None
        view_1518: "bf16[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_449, [1, 128, 4096]);  mm_449 = None
        convert_element_type_2688: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1518, torch.float32);  view_1518 = None
        add_594: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_593, convert_element_type_2688);  add_593 = convert_element_type_2688 = None
        convert_element_type_2689: "f32[4096, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_448, torch.float32);  mm_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_875: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_594, primals_3);  primals_3 = None
        mul_876: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_875, 4096)
        sum_234: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_875, [2], True)
        sub_2: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(embedding, getitem_1);  embedding = getitem_1 = None
        mul: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_877: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_875, mul);  mul_875 = None
        sum_235: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_877, [2], True);  mul_877 = None
        mul_878: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_235);  sum_235 = None
        sub_178: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_876, sum_234);  mul_876 = sum_234 = None
        sub_179: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_178, mul_878);  sub_178 = mul_878 = None
        div_118: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 4096);  rsqrt = None
        mul_879: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_118, sub_179);  div_118 = sub_179 = None
        mul_880: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_594, mul);  mul = None
        sum_236: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_880, [0, 1]);  mul_880 = None
        sum_237: "f32[4096][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_594, [0, 1]);  add_594 = None
        add_595: "f32[1, 128, 4096][524288, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(add_583, mul_879);  add_583 = mul_879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:494 in forward, code: inputs_embeds = self.wte(input_ids)
        ge: "b8[1, 128][128, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_1, 0)
        lt: "b8[1, 128][128, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_1, 50400)
        bitwise_and_2: "b8[1, 128][128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne_11: "b8[1, 128][128, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_1, -1)
        bitwise_and_3: "b8[1, 128][128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_11);  bitwise_and_2 = ne_11 = None
        unsqueeze_380: "b8[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_default_236: "f32[50400, 4096][4096, 1]cuda:0" = torch.ops.aten.full.default([50400, 4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate: "f32[50400, 4096][4096, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_236, unsqueeze_380, [primals_1], add_595);  full_default_236 = unsqueeze_380 = primals_1 = add_595 = None
        return (None, _unsafe_masked_index_put_accumulate, sum_236, sum_237, convert_element_type_2689, convert_element_type_2683, convert_element_type_2677, None, convert_element_type_2657, convert_element_type_2651, convert_element_type_2652, convert_element_type_2640, convert_element_type_2641, sum_229, sum_230, convert_element_type_2632, convert_element_type_2626, convert_element_type_2620, None, convert_element_type_2600, convert_element_type_2594, convert_element_type_2595, convert_element_type_2583, convert_element_type_2584, sum_222, sum_223, convert_element_type_2575, convert_element_type_2569, convert_element_type_2563, None, convert_element_type_2543, convert_element_type_2537, convert_element_type_2538, convert_element_type_2526, convert_element_type_2527, sum_215, sum_216, convert_element_type_2518, convert_element_type_2512, convert_element_type_2506, None, convert_element_type_2486, convert_element_type_2480, convert_element_type_2481, convert_element_type_2469, convert_element_type_2470, sum_208, sum_209, convert_element_type_2461, convert_element_type_2455, convert_element_type_2449, None, convert_element_type_2429, convert_element_type_2423, convert_element_type_2424, convert_element_type_2412, convert_element_type_2413, sum_201, sum_202, convert_element_type_2404, convert_element_type_2398, convert_element_type_2392, None, convert_element_type_2372, convert_element_type_2366, convert_element_type_2367, convert_element_type_2355, convert_element_type_2356, sum_194, sum_195, convert_element_type_2347, convert_element_type_2341, convert_element_type_2335, None, convert_element_type_2315, convert_element_type_2309, convert_element_type_2310, convert_element_type_2298, convert_element_type_2299, sum_187, sum_188, convert_element_type_2290, convert_element_type_2284, convert_element_type_2278, None, convert_element_type_2258, convert_element_type_2252, convert_element_type_2253, convert_element_type_2241, convert_element_type_2242, sum_180, sum_181, convert_element_type_2233, convert_element_type_2227, convert_element_type_2221, None, convert_element_type_2201, convert_element_type_2195, convert_element_type_2196, convert_element_type_2184, convert_element_type_2185, sum_173, sum_174, convert_element_type_2176, convert_element_type_2170, convert_element_type_2164, None, convert_element_type_2144, convert_element_type_2138, convert_element_type_2139, convert_element_type_2127, convert_element_type_2128, sum_166, sum_167, convert_element_type_2119, convert_element_type_2113, convert_element_type_2107, None, convert_element_type_2087, convert_element_type_2081, convert_element_type_2082, convert_element_type_2070, convert_element_type_2071, sum_159, sum_160, convert_element_type_2062, convert_element_type_2056, convert_element_type_2050, None, convert_element_type_2030, convert_element_type_2024, convert_element_type_2025, convert_element_type_2013, convert_element_type_2014, sum_152, sum_153, convert_element_type_2005, convert_element_type_1999, convert_element_type_1993, None, convert_element_type_1973, convert_element_type_1967, convert_element_type_1968, convert_element_type_1956, convert_element_type_1957, sum_145, sum_146, convert_element_type_1948, convert_element_type_1942, convert_element_type_1936, None, convert_element_type_1916, convert_element_type_1910, convert_element_type_1911, convert_element_type_1899, convert_element_type_1900, sum_138, sum_139, convert_element_type_1891, convert_element_type_1885, convert_element_type_1879, None, convert_element_type_1859, convert_element_type_1853, convert_element_type_1854, convert_element_type_1842, convert_element_type_1843, sum_131, sum_132, convert_element_type_1834, convert_element_type_1828, convert_element_type_1822, None, convert_element_type_1802, convert_element_type_1796, convert_element_type_1797, convert_element_type_1785, convert_element_type_1786, sum_124, sum_125, convert_element_type_1777, convert_element_type_1771, convert_element_type_1765, None, convert_element_type_1745, convert_element_type_1739, convert_element_type_1740, convert_element_type_1728, convert_element_type_1729, sum_117, sum_118, convert_element_type_1720, convert_element_type_1714, convert_element_type_1708, None, convert_element_type_1688, convert_element_type_1682, convert_element_type_1683, convert_element_type_1671, convert_element_type_1672, sum_110, sum_111, convert_element_type_1663, convert_element_type_1657, convert_element_type_1651, None, convert_element_type_1631, convert_element_type_1625, convert_element_type_1626, convert_element_type_1614, convert_element_type_1615, sum_103, sum_104, convert_element_type_1606, convert_element_type_1600, convert_element_type_1594, None, convert_element_type_1574, convert_element_type_1568, convert_element_type_1569, convert_element_type_1557, convert_element_type_1558, sum_96, sum_97, convert_element_type_1549, convert_element_type_1543, convert_element_type_1537, None, convert_element_type_1517, convert_element_type_1511, convert_element_type_1512, convert_element_type_1500, convert_element_type_1501, sum_89, sum_90, convert_element_type_1492, convert_element_type_1486, convert_element_type_1480, None, convert_element_type_1460, convert_element_type_1454, convert_element_type_1455, convert_element_type_1443, convert_element_type_1444, sum_82, sum_83, convert_element_type_1435, convert_element_type_1429, convert_element_type_1423, None, convert_element_type_1403, convert_element_type_1397, convert_element_type_1398, convert_element_type_1386, convert_element_type_1387, sum_75, sum_76, convert_element_type_1378, convert_element_type_1372, convert_element_type_1366, None, convert_element_type_1346, convert_element_type_1340, convert_element_type_1341, convert_element_type_1329, convert_element_type_1330, sum_68, sum_69, convert_element_type_1321, convert_element_type_1315, convert_element_type_1309, None, convert_element_type_1289, convert_element_type_1283, convert_element_type_1284, convert_element_type_1272, convert_element_type_1273, sum_61, sum_62, convert_element_type_1264, convert_element_type_1258, convert_element_type_1252, None, convert_element_type_1232, convert_element_type_1226, convert_element_type_1227, convert_element_type_1215, convert_element_type_1216, sum_54, sum_55, convert_element_type_1207, convert_element_type_1201, convert_element_type_1195, None, convert_element_type_1175, convert_element_type_1169, convert_element_type_1170, convert_element_type_1158, convert_element_type_1159, sum_47, sum_48, convert_element_type_1150, convert_element_type_1144, convert_element_type_1138, None, convert_element_type_1118, convert_element_type_1112, convert_element_type_1113, convert_element_type_1101, convert_element_type_1102, sum_40, sum_41, convert_element_type_1092, convert_element_type_1093, None, None)
