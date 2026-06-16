class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 128][128, 1]cuda:0", primals_4: "f32[2048][1]cuda:0", primals_9: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_12: "f32[2048][1]cuda:0", primals_18: "f32[2048][1]cuda:0", primals_23: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_26: "f32[2048][1]cuda:0", primals_32: "f32[2048][1]cuda:0", primals_37: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_40: "f32[2048][1]cuda:0", primals_46: "f32[2048][1]cuda:0", primals_51: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_54: "f32[2048][1]cuda:0", primals_60: "f32[2048][1]cuda:0", primals_65: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_68: "f32[2048][1]cuda:0", primals_74: "f32[2048][1]cuda:0", primals_79: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_82: "f32[2048][1]cuda:0", primals_88: "f32[2048][1]cuda:0", primals_93: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_96: "f32[2048][1]cuda:0", primals_102: "f32[2048][1]cuda:0", primals_107: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_110: "f32[2048][1]cuda:0", primals_116: "f32[2048][1]cuda:0", primals_121: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_124: "f32[2048][1]cuda:0", primals_130: "f32[2048][1]cuda:0", primals_135: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_138: "f32[2048][1]cuda:0", primals_144: "f32[2048][1]cuda:0", primals_149: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_152: "f32[2048][1]cuda:0", primals_158: "f32[2048][1]cuda:0", primals_163: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_166: "f32[2048][1]cuda:0", primals_172: "f32[2048][1]cuda:0", primals_177: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_180: "f32[2048][1]cuda:0", primals_186: "f32[2048][1]cuda:0", primals_191: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_194: "f32[2048][1]cuda:0", primals_200: "f32[2048][1]cuda:0", primals_205: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_208: "f32[2048][1]cuda:0", primals_214: "f32[2048][1]cuda:0", primals_219: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_222: "f32[2048][1]cuda:0", primals_228: "f32[2048][1]cuda:0", primals_233: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_236: "f32[2048][1]cuda:0", primals_242: "f32[2048][1]cuda:0", primals_247: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_250: "f32[2048][1]cuda:0", primals_256: "f32[2048][1]cuda:0", primals_261: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_264: "f32[2048][1]cuda:0", primals_270: "f32[2048][1]cuda:0", primals_275: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_278: "f32[2048][1]cuda:0", primals_284: "f32[2048][1]cuda:0", primals_289: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_292: "f32[2048][1]cuda:0", primals_298: "f32[2048][1]cuda:0", primals_303: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_306: "f32[2048][1]cuda:0", primals_312: "f32[2048][1]cuda:0", primals_317: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_320: "f32[2048][1]cuda:0", primals_326: "f32[2048][1]cuda:0", primals_331: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_334: "f32[2048][1]cuda:0", primals_340: "f32[2048][1]cuda:0", embedding: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", unsqueeze: "i64[1, 128][128, 1]cuda:0", cumsum: "i64[32, 128][128, 1]cuda:0", embedding_1: "f32[1, 128, 2048][262144, 2048, 1]cuda:0", getitem_1: "f32[32, 128, 1][128, 1, 1]cuda:0", rsqrt: "f32[32, 128, 1][128, 1, 1]cuda:0", view: "bf16[4096, 2048][2048, 1]cuda:0", bmm: "bf16[512, 128, 128][16384, 128, 1]cuda:0", amax: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0", sum_1: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0", view_16: "bf16[4096, 2048][2048, 1]cuda:0", mul_2: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_18: "bf16[4096, 2048][2048, 1]cuda:0", addmm_1: "bf16[4096, 8192][8192, 1]cuda:0", view_20: "bf16[4096, 8192][8192, 1]cuda:0", mul_8: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_22: "bf16[4096, 2048][2048, 1]cuda:0", div_1: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_38: "bf16[4096, 2048][2048, 1]cuda:0", mul_10: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_40: "bf16[4096, 2048][2048, 1]cuda:0", addmm_4: "bf16[4096, 8192][8192, 1]cuda:0", view_42: "bf16[4096, 8192][8192, 1]cuda:0", mul_16: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_44: "bf16[4096, 2048][2048, 1]cuda:0", div_2: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_60: "bf16[4096, 2048][2048, 1]cuda:0", mul_18: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_62: "bf16[4096, 2048][2048, 1]cuda:0", addmm_7: "bf16[4096, 8192][8192, 1]cuda:0", view_64: "bf16[4096, 8192][8192, 1]cuda:0", mul_24: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_66: "bf16[4096, 2048][2048, 1]cuda:0", div_3: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_82: "bf16[4096, 2048][2048, 1]cuda:0", mul_26: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_84: "bf16[4096, 2048][2048, 1]cuda:0", addmm_10: "bf16[4096, 8192][8192, 1]cuda:0", view_86: "bf16[4096, 8192][8192, 1]cuda:0", mul_32: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_88: "bf16[4096, 2048][2048, 1]cuda:0", div_4: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_104: "bf16[4096, 2048][2048, 1]cuda:0", mul_34: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_106: "bf16[4096, 2048][2048, 1]cuda:0", addmm_13: "bf16[4096, 8192][8192, 1]cuda:0", view_108: "bf16[4096, 8192][8192, 1]cuda:0", mul_40: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_110: "bf16[4096, 2048][2048, 1]cuda:0", div_5: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_126: "bf16[4096, 2048][2048, 1]cuda:0", mul_42: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_128: "bf16[4096, 2048][2048, 1]cuda:0", addmm_16: "bf16[4096, 8192][8192, 1]cuda:0", view_130: "bf16[4096, 8192][8192, 1]cuda:0", mul_48: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_132: "bf16[4096, 2048][2048, 1]cuda:0", div_6: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_148: "bf16[4096, 2048][2048, 1]cuda:0", mul_50: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_150: "bf16[4096, 2048][2048, 1]cuda:0", addmm_19: "bf16[4096, 8192][8192, 1]cuda:0", view_152: "bf16[4096, 8192][8192, 1]cuda:0", mul_56: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_154: "bf16[4096, 2048][2048, 1]cuda:0", div_7: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_170: "bf16[4096, 2048][2048, 1]cuda:0", mul_58: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_172: "bf16[4096, 2048][2048, 1]cuda:0", addmm_22: "bf16[4096, 8192][8192, 1]cuda:0", view_174: "bf16[4096, 8192][8192, 1]cuda:0", mul_64: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_176: "bf16[4096, 2048][2048, 1]cuda:0", div_8: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_192: "bf16[4096, 2048][2048, 1]cuda:0", mul_66: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_194: "bf16[4096, 2048][2048, 1]cuda:0", addmm_25: "bf16[4096, 8192][8192, 1]cuda:0", view_196: "bf16[4096, 8192][8192, 1]cuda:0", mul_72: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_198: "bf16[4096, 2048][2048, 1]cuda:0", div_9: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_214: "bf16[4096, 2048][2048, 1]cuda:0", mul_74: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_216: "bf16[4096, 2048][2048, 1]cuda:0", addmm_28: "bf16[4096, 8192][8192, 1]cuda:0", view_218: "bf16[4096, 8192][8192, 1]cuda:0", mul_80: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_220: "bf16[4096, 2048][2048, 1]cuda:0", div_10: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_236: "bf16[4096, 2048][2048, 1]cuda:0", mul_82: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_238: "bf16[4096, 2048][2048, 1]cuda:0", addmm_31: "bf16[4096, 8192][8192, 1]cuda:0", view_240: "bf16[4096, 8192][8192, 1]cuda:0", mul_88: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_242: "bf16[4096, 2048][2048, 1]cuda:0", div_11: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_258: "bf16[4096, 2048][2048, 1]cuda:0", mul_90: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_260: "bf16[4096, 2048][2048, 1]cuda:0", addmm_34: "bf16[4096, 8192][8192, 1]cuda:0", view_262: "bf16[4096, 8192][8192, 1]cuda:0", mul_96: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_264: "bf16[4096, 2048][2048, 1]cuda:0", div_12: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_280: "bf16[4096, 2048][2048, 1]cuda:0", mul_98: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_282: "bf16[4096, 2048][2048, 1]cuda:0", addmm_37: "bf16[4096, 8192][8192, 1]cuda:0", view_284: "bf16[4096, 8192][8192, 1]cuda:0", mul_104: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_286: "bf16[4096, 2048][2048, 1]cuda:0", div_13: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_302: "bf16[4096, 2048][2048, 1]cuda:0", mul_106: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_304: "bf16[4096, 2048][2048, 1]cuda:0", addmm_40: "bf16[4096, 8192][8192, 1]cuda:0", view_306: "bf16[4096, 8192][8192, 1]cuda:0", mul_112: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_308: "bf16[4096, 2048][2048, 1]cuda:0", div_14: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_324: "bf16[4096, 2048][2048, 1]cuda:0", mul_114: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_326: "bf16[4096, 2048][2048, 1]cuda:0", addmm_43: "bf16[4096, 8192][8192, 1]cuda:0", view_328: "bf16[4096, 8192][8192, 1]cuda:0", mul_120: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_330: "bf16[4096, 2048][2048, 1]cuda:0", div_15: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_346: "bf16[4096, 2048][2048, 1]cuda:0", mul_122: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_348: "bf16[4096, 2048][2048, 1]cuda:0", addmm_46: "bf16[4096, 8192][8192, 1]cuda:0", view_350: "bf16[4096, 8192][8192, 1]cuda:0", mul_128: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_352: "bf16[4096, 2048][2048, 1]cuda:0", div_16: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_368: "bf16[4096, 2048][2048, 1]cuda:0", mul_130: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_370: "bf16[4096, 2048][2048, 1]cuda:0", addmm_49: "bf16[4096, 8192][8192, 1]cuda:0", view_372: "bf16[4096, 8192][8192, 1]cuda:0", mul_136: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_374: "bf16[4096, 2048][2048, 1]cuda:0", div_17: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_390: "bf16[4096, 2048][2048, 1]cuda:0", mul_138: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_392: "bf16[4096, 2048][2048, 1]cuda:0", addmm_52: "bf16[4096, 8192][8192, 1]cuda:0", view_394: "bf16[4096, 8192][8192, 1]cuda:0", mul_144: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_396: "bf16[4096, 2048][2048, 1]cuda:0", div_18: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_412: "bf16[4096, 2048][2048, 1]cuda:0", mul_146: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_414: "bf16[4096, 2048][2048, 1]cuda:0", addmm_55: "bf16[4096, 8192][8192, 1]cuda:0", view_416: "bf16[4096, 8192][8192, 1]cuda:0", mul_152: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_418: "bf16[4096, 2048][2048, 1]cuda:0", div_19: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_434: "bf16[4096, 2048][2048, 1]cuda:0", mul_154: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_436: "bf16[4096, 2048][2048, 1]cuda:0", addmm_58: "bf16[4096, 8192][8192, 1]cuda:0", view_438: "bf16[4096, 8192][8192, 1]cuda:0", mul_160: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_440: "bf16[4096, 2048][2048, 1]cuda:0", div_20: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_456: "bf16[4096, 2048][2048, 1]cuda:0", mul_162: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_458: "bf16[4096, 2048][2048, 1]cuda:0", addmm_61: "bf16[4096, 8192][8192, 1]cuda:0", view_460: "bf16[4096, 8192][8192, 1]cuda:0", mul_168: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_462: "bf16[4096, 2048][2048, 1]cuda:0", div_21: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_478: "bf16[4096, 2048][2048, 1]cuda:0", mul_170: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_480: "bf16[4096, 2048][2048, 1]cuda:0", addmm_64: "bf16[4096, 8192][8192, 1]cuda:0", view_482: "bf16[4096, 8192][8192, 1]cuda:0", mul_176: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_484: "bf16[4096, 2048][2048, 1]cuda:0", div_22: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_500: "bf16[4096, 2048][2048, 1]cuda:0", mul_178: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_502: "bf16[4096, 2048][2048, 1]cuda:0", addmm_67: "bf16[4096, 8192][8192, 1]cuda:0", view_504: "bf16[4096, 8192][8192, 1]cuda:0", mul_184: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_506: "bf16[4096, 2048][2048, 1]cuda:0", div_23: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_522: "bf16[4096, 2048][2048, 1]cuda:0", mul_186: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_524: "bf16[4096, 2048][2048, 1]cuda:0", addmm_70: "bf16[4096, 8192][8192, 1]cuda:0", view_526: "bf16[4096, 8192][8192, 1]cuda:0", mul_192: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_529: "bf16[4096, 2048][2048, 1]cuda:0", view_530: "bf16[32, 128, 50257][6433792, 50264, 1]cuda:0", constant_pad_nd: "i64[32, 129][129, 1]cuda:0", amax_24: "f32[4096, 1][1, 1]cuda:0", log: "f32[4096, 1][1, 1]cuda:0", convert_element_type_941: "f32[][]cuda:0", permute_267: "bf16[50257, 2048][2048, 1]cuda:0", div_26: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_269: "bf16[2048, 8192][8192, 1]cuda:0", permute_273: "bf16[8192, 2048][2048, 1]cuda:0", div_27: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_277: "bf16[2048, 2048][2048, 1]cuda:0", permute_282: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_283: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_284: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_285: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_292: "bf16[2048, 2048][2048, 1]cuda:0", permute_296: "bf16[2048, 2048][2048, 1]cuda:0", permute_300: "bf16[2048, 2048][2048, 1]cuda:0", div_28: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_302: "bf16[2048, 8192][8192, 1]cuda:0", permute_306: "bf16[8192, 2048][2048, 1]cuda:0", div_29: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_310: "bf16[2048, 2048][2048, 1]cuda:0", permute_315: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_316: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_317: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_318: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_325: "bf16[2048, 2048][2048, 1]cuda:0", permute_329: "bf16[2048, 2048][2048, 1]cuda:0", permute_333: "bf16[2048, 2048][2048, 1]cuda:0", div_30: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_335: "bf16[2048, 8192][8192, 1]cuda:0", permute_339: "bf16[8192, 2048][2048, 1]cuda:0", div_31: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_343: "bf16[2048, 2048][2048, 1]cuda:0", permute_348: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_349: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_350: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_351: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_358: "bf16[2048, 2048][2048, 1]cuda:0", permute_362: "bf16[2048, 2048][2048, 1]cuda:0", permute_366: "bf16[2048, 2048][2048, 1]cuda:0", div_32: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_368: "bf16[2048, 8192][8192, 1]cuda:0", permute_372: "bf16[8192, 2048][2048, 1]cuda:0", div_33: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_376: "bf16[2048, 2048][2048, 1]cuda:0", permute_381: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_382: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_383: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_384: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_391: "bf16[2048, 2048][2048, 1]cuda:0", permute_395: "bf16[2048, 2048][2048, 1]cuda:0", permute_399: "bf16[2048, 2048][2048, 1]cuda:0", div_34: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_401: "bf16[2048, 8192][8192, 1]cuda:0", permute_405: "bf16[8192, 2048][2048, 1]cuda:0", div_35: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_409: "bf16[2048, 2048][2048, 1]cuda:0", permute_414: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_415: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_416: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_417: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_424: "bf16[2048, 2048][2048, 1]cuda:0", permute_428: "bf16[2048, 2048][2048, 1]cuda:0", permute_432: "bf16[2048, 2048][2048, 1]cuda:0", div_36: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_434: "bf16[2048, 8192][8192, 1]cuda:0", permute_438: "bf16[8192, 2048][2048, 1]cuda:0", div_37: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_442: "bf16[2048, 2048][2048, 1]cuda:0", permute_447: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_448: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_449: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_450: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_457: "bf16[2048, 2048][2048, 1]cuda:0", permute_461: "bf16[2048, 2048][2048, 1]cuda:0", permute_465: "bf16[2048, 2048][2048, 1]cuda:0", div_38: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_467: "bf16[2048, 8192][8192, 1]cuda:0", permute_471: "bf16[8192, 2048][2048, 1]cuda:0", div_39: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_475: "bf16[2048, 2048][2048, 1]cuda:0", permute_480: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_481: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_482: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_483: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_490: "bf16[2048, 2048][2048, 1]cuda:0", permute_494: "bf16[2048, 2048][2048, 1]cuda:0", permute_498: "bf16[2048, 2048][2048, 1]cuda:0", div_40: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_500: "bf16[2048, 8192][8192, 1]cuda:0", permute_504: "bf16[8192, 2048][2048, 1]cuda:0", div_41: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_508: "bf16[2048, 2048][2048, 1]cuda:0", permute_513: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_514: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_515: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_516: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_523: "bf16[2048, 2048][2048, 1]cuda:0", permute_527: "bf16[2048, 2048][2048, 1]cuda:0", permute_531: "bf16[2048, 2048][2048, 1]cuda:0", div_42: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_533: "bf16[2048, 8192][8192, 1]cuda:0", permute_537: "bf16[8192, 2048][2048, 1]cuda:0", div_43: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_541: "bf16[2048, 2048][2048, 1]cuda:0", permute_546: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_547: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_548: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_549: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_556: "bf16[2048, 2048][2048, 1]cuda:0", permute_560: "bf16[2048, 2048][2048, 1]cuda:0", permute_564: "bf16[2048, 2048][2048, 1]cuda:0", div_44: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_566: "bf16[2048, 8192][8192, 1]cuda:0", permute_570: "bf16[8192, 2048][2048, 1]cuda:0", div_45: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_574: "bf16[2048, 2048][2048, 1]cuda:0", permute_579: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_580: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_581: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_582: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_589: "bf16[2048, 2048][2048, 1]cuda:0", permute_593: "bf16[2048, 2048][2048, 1]cuda:0", permute_597: "bf16[2048, 2048][2048, 1]cuda:0", div_46: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_599: "bf16[2048, 8192][8192, 1]cuda:0", permute_603: "bf16[8192, 2048][2048, 1]cuda:0", div_47: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_607: "bf16[2048, 2048][2048, 1]cuda:0", permute_612: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_613: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_614: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_615: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_622: "bf16[2048, 2048][2048, 1]cuda:0", permute_626: "bf16[2048, 2048][2048, 1]cuda:0", permute_630: "bf16[2048, 2048][2048, 1]cuda:0", div_48: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_632: "bf16[2048, 8192][8192, 1]cuda:0", permute_636: "bf16[8192, 2048][2048, 1]cuda:0", div_49: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_640: "bf16[2048, 2048][2048, 1]cuda:0", permute_645: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_646: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_647: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_648: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_655: "bf16[2048, 2048][2048, 1]cuda:0", permute_659: "bf16[2048, 2048][2048, 1]cuda:0", permute_663: "bf16[2048, 2048][2048, 1]cuda:0", div_50: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_665: "bf16[2048, 8192][8192, 1]cuda:0", permute_669: "bf16[8192, 2048][2048, 1]cuda:0", div_51: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_673: "bf16[2048, 2048][2048, 1]cuda:0", permute_678: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_679: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_680: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_681: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_688: "bf16[2048, 2048][2048, 1]cuda:0", permute_692: "bf16[2048, 2048][2048, 1]cuda:0", permute_696: "bf16[2048, 2048][2048, 1]cuda:0", div_52: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_698: "bf16[2048, 8192][8192, 1]cuda:0", permute_702: "bf16[8192, 2048][2048, 1]cuda:0", div_53: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_706: "bf16[2048, 2048][2048, 1]cuda:0", permute_711: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_712: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_713: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_714: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_721: "bf16[2048, 2048][2048, 1]cuda:0", permute_725: "bf16[2048, 2048][2048, 1]cuda:0", permute_729: "bf16[2048, 2048][2048, 1]cuda:0", div_54: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_731: "bf16[2048, 8192][8192, 1]cuda:0", permute_735: "bf16[8192, 2048][2048, 1]cuda:0", div_55: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_739: "bf16[2048, 2048][2048, 1]cuda:0", permute_744: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_745: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_746: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_747: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_754: "bf16[2048, 2048][2048, 1]cuda:0", permute_758: "bf16[2048, 2048][2048, 1]cuda:0", permute_762: "bf16[2048, 2048][2048, 1]cuda:0", div_56: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_764: "bf16[2048, 8192][8192, 1]cuda:0", permute_768: "bf16[8192, 2048][2048, 1]cuda:0", div_57: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_772: "bf16[2048, 2048][2048, 1]cuda:0", permute_777: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_778: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_779: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_780: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_787: "bf16[2048, 2048][2048, 1]cuda:0", permute_791: "bf16[2048, 2048][2048, 1]cuda:0", permute_795: "bf16[2048, 2048][2048, 1]cuda:0", div_58: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_797: "bf16[2048, 8192][8192, 1]cuda:0", permute_801: "bf16[8192, 2048][2048, 1]cuda:0", div_59: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_805: "bf16[2048, 2048][2048, 1]cuda:0", permute_810: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_811: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_812: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_813: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_820: "bf16[2048, 2048][2048, 1]cuda:0", permute_824: "bf16[2048, 2048][2048, 1]cuda:0", permute_828: "bf16[2048, 2048][2048, 1]cuda:0", div_60: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_830: "bf16[2048, 8192][8192, 1]cuda:0", permute_834: "bf16[8192, 2048][2048, 1]cuda:0", div_61: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_838: "bf16[2048, 2048][2048, 1]cuda:0", permute_843: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_844: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_845: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_846: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_853: "bf16[2048, 2048][2048, 1]cuda:0", permute_857: "bf16[2048, 2048][2048, 1]cuda:0", permute_861: "bf16[2048, 2048][2048, 1]cuda:0", div_62: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_863: "bf16[2048, 8192][8192, 1]cuda:0", permute_867: "bf16[8192, 2048][2048, 1]cuda:0", div_63: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_871: "bf16[2048, 2048][2048, 1]cuda:0", permute_876: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_877: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_878: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_879: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_886: "bf16[2048, 2048][2048, 1]cuda:0", permute_890: "bf16[2048, 2048][2048, 1]cuda:0", permute_894: "bf16[2048, 2048][2048, 1]cuda:0", div_64: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_896: "bf16[2048, 8192][8192, 1]cuda:0", permute_900: "bf16[8192, 2048][2048, 1]cuda:0", div_65: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_904: "bf16[2048, 2048][2048, 1]cuda:0", permute_909: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_910: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_911: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_912: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_919: "bf16[2048, 2048][2048, 1]cuda:0", permute_923: "bf16[2048, 2048][2048, 1]cuda:0", permute_927: "bf16[2048, 2048][2048, 1]cuda:0", div_66: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_929: "bf16[2048, 8192][8192, 1]cuda:0", permute_933: "bf16[8192, 2048][2048, 1]cuda:0", div_67: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_937: "bf16[2048, 2048][2048, 1]cuda:0", permute_942: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_943: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_944: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_945: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_952: "bf16[2048, 2048][2048, 1]cuda:0", permute_956: "bf16[2048, 2048][2048, 1]cuda:0", permute_960: "bf16[2048, 2048][2048, 1]cuda:0", div_68: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_962: "bf16[2048, 8192][8192, 1]cuda:0", permute_966: "bf16[8192, 2048][2048, 1]cuda:0", div_69: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_970: "bf16[2048, 2048][2048, 1]cuda:0", permute_975: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_976: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_977: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_978: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_985: "bf16[2048, 2048][2048, 1]cuda:0", permute_989: "bf16[2048, 2048][2048, 1]cuda:0", permute_993: "bf16[2048, 2048][2048, 1]cuda:0", div_70: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_995: "bf16[2048, 8192][8192, 1]cuda:0", permute_999: "bf16[8192, 2048][2048, 1]cuda:0", div_71: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_1003: "bf16[2048, 2048][2048, 1]cuda:0", permute_1008: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_1009: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_1010: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_1011: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_1018: "bf16[2048, 2048][2048, 1]cuda:0", permute_1022: "bf16[2048, 2048][2048, 1]cuda:0", permute_1026: "bf16[2048, 2048][2048, 1]cuda:0", div_72: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_1028: "bf16[2048, 8192][8192, 1]cuda:0", permute_1032: "bf16[8192, 2048][2048, 1]cuda:0", div_73: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_1036: "bf16[2048, 2048][2048, 1]cuda:0", permute_1041: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_1042: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_1043: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_1044: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_1051: "bf16[2048, 2048][2048, 1]cuda:0", permute_1055: "bf16[2048, 2048][2048, 1]cuda:0", permute_1059: "bf16[2048, 2048][2048, 1]cuda:0", tangents_1: "f32[][]cuda:0", tangents_2: "bf16[32, 128, 50257][6432896, 50257, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        div_25: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_941);  tangents_1 = convert_element_type_941 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_52: "i64[32, 128][129, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_169: "i64[32, 128][128, 1]cuda:0" = torch.ops.aten.clone.default(slice_52, memory_format = torch.contiguous_format);  slice_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_532: "i64[4096][1]cuda:0" = torch.ops.aten.reshape.default(clone_169, [-1]);  clone_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        unsqueeze_11: "i64[4096, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_532, 1);  view_532 = None
        ne_4: "b8[4096, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_11, -100)
        full_default_26: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27: "i64[4096, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_4, unsqueeze_11, full_default_26);  unsqueeze_11 = full_default_26 = None

        # No stacktrace found for following nodes
        iota_default: "i64[50257][1]cuda:0" = torch.ops.prims.iota.default(50257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 50257][50257, 1]cuda:0" = torch.ops.aten.reshape.default(iota_default, [1, 50257]);  iota_default = None
        expand_default: "i64[4096, 50257][1, 0]cuda:0" = torch.ops.aten.expand.default(where_27, [4096, 50257]);  where_27 = None
        eq_tensor: "b8[4096, 50257][50257, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_self: "f32[4096, 50257][50257, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_28: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_4, div_25, full_default);  ne_4 = div_25 = None
        mul_194: "f32[4096, 50257][50257, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self, where_28);  where_self = where_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_940: "f32[32, 128, 50257][6432896, 50257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_530, torch.float32);  view_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_531: "f32[4096, 50257][50257, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_940, [-1, 50257]);  convert_element_type_940 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        sub_75: "f32[4096, 50257][50257, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_531, amax_24);  view_531 = amax_24 = None
        sub_76: "f32[4096, 50257][50257, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_75, log);  sub_75 = log = None
        exp_25: "f32[4096, 50257][50257, 1]cuda:0" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
        sum_28: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_194, [1], True)
        mul_195: "f32[4096, 50257][50257, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
        sub_77: "f32[4096, 50257][50257, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_533: "f32[32, 128, 50257][6432896, 50257, 1]cuda:0" = torch.ops.aten.reshape.default(sub_77, [32, 128, 50257]);  sub_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_942: "bf16[32, 128, 50257][6432896, 50257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_533, torch.bfloat16);  view_533 = None
        add_222: "bf16[32, 128, 50257][6432896, 50257, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, convert_element_type_942);  tangents_2 = convert_element_type_942 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:583 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_534: "bf16[4096, 50257][50257, 1]cuda:0" = torch.ops.aten.reshape.default(add_222, [4096, 50257]);  add_222 = None
        permute_265: "bf16[50257, 4096][1, 50257]cuda:0" = torch.ops.aten.permute.default(view_534, [1, 0])
        constant_pad_nd_default_2: "bf16[50264, 4096][4096, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_265, [0, 0, 0, 7]);  permute_265 = None
        mm_default_1: "bf16[50264, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default_2, view_529);  constant_pad_nd_default_2 = view_529 = None
        slice_tensor: "bf16[50257, 2048][2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(mm_default_1, 0, 0, -7);  mm_default_1 = None
        constant_pad_nd_default: "bf16[4096, 50264][50264, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_534, [0, 7, 0, 0]);  view_534 = None
        constant_pad_nd_default_1: "bf16[50264, 2048][2048, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_267, [0, 0, 0, 7]);  permute_267 = None
        mm_default: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        view_535: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_default, [32, 128, 2048]);  mm_default = None
        convert_element_type_947: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_535, torch.float32);  view_535 = None
        convert_element_type_948: "f32[50257, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:492 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_197: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_947, primals_340);  primals_340 = None
        mul_198: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_197, 2048)
        sum_29: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True)
        mul_199: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_197, mul_192);  mul_197 = None
        sum_30: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True);  mul_199 = None
        mul_200: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, sum_30);  sum_30 = None
        sub_79: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_198, sum_29);  mul_198 = sum_29 = None
        sub_80: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_79, mul_200);  sub_79 = mul_200 = None
        mul_201: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_26, sub_80);  div_26 = sub_80 = None
        mul_202: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_947, mul_192);  mul_192 = None
        sum_31: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_202, [0, 1]);  mul_202 = None
        sum_32: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_947, [0, 1]);  convert_element_type_947 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_949: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_201, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_537: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_949, [4096, 2048]);  convert_element_type_949 = None
        mm_75: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_537, permute_269);  permute_269 = None
        permute_270: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_537, [1, 0])
        mm_76: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_270, view_526);  permute_270 = view_526 = None
        sum_33: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_537, [0], True, dtype = torch.float32);  view_537 = None
        view_538: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_33, [2048]);  sum_33 = None
        convert_element_type_954: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_538, torch.bfloat16);  view_538 = None
        view_539: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_75, [32, 128, 8192]);  mm_75 = None
        convert_element_type_955: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_539, torch.float32);  view_539 = None
        convert_element_type_956: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_76, torch.float32);  mm_76 = None
        convert_element_type_957: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_954, torch.float32);  convert_element_type_954 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_525: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [32, 128, 8192]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_188: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_525, 0.5)
        mul_203: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_955, mul_188);  mul_188 = None
        convert_element_type_929: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_525, torch.float32)
        pow_24: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_929, 3.0)
        mul_189: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_217: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_525, mul_189);  view_525 = mul_189 = None
        mul_190: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_217, 0.7978845608028654);  add_217 = None
        tanh_23: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_190);  mul_190 = None
        add_218: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_23, 1.0)
        mul_204: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_955, add_218);  convert_element_type_955 = add_218 = None
        convert_element_type_958: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_204, torch.bfloat16);  mul_204 = None
        mul_205: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_23, tanh_23);  tanh_23 = None
        sub_81: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_205);  mul_205 = None
        mul_206: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_203, sub_81);  mul_203 = sub_81 = None
        mul_207: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, 0.7978845608028654);  mul_206 = None
        convert_element_type_959: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_207, torch.bfloat16)
        mul_208: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_207, 0.044715);  mul_207 = None
        pow_25: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_929, 2.0);  convert_element_type_929 = None
        mul_209: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_25, 3.0);  pow_25 = None
        mul_210: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_208, mul_209);  mul_208 = mul_209 = None
        convert_element_type_960: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_210, torch.bfloat16);  mul_210 = None
        add_223: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_959, convert_element_type_960);  convert_element_type_959 = convert_element_type_960 = None
        mul_211: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_958, 0.5);  convert_element_type_958 = None
        add_224: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_223, mul_211);  add_223 = mul_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_540: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_224, [4096, 8192]);  add_224 = None
        mm_77: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_540, permute_273);  permute_273 = None
        permute_274: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_540, [1, 0])
        mm_78: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_274, view_524);  permute_274 = view_524 = None
        sum_34: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_540, [0], True, dtype = torch.float32);  view_540 = None
        view_541: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_34, [8192]);  sum_34 = None
        convert_element_type_965: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_541, torch.bfloat16);  view_541 = None
        view_542: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_77, [32, 128, 2048]);  mm_77 = None
        convert_element_type_966: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_542, torch.float32);  view_542 = None
        convert_element_type_967: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_78, torch.float32);  mm_78 = None
        convert_element_type_968: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_965, torch.float32);  convert_element_type_965 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_213: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_966, primals_334);  primals_334 = None
        mul_214: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, 2048)
        sum_35: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
        mul_215: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, mul_186);  mul_213 = None
        sum_36: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
        mul_216: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_186, sum_36);  sum_36 = None
        sub_83: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_214, sum_35);  mul_214 = sum_35 = None
        sub_84: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_83, mul_216);  sub_83 = mul_216 = None
        mul_217: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_27, sub_84);  div_27 = sub_84 = None
        mul_218: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_966, mul_186);  mul_186 = None
        sum_37: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
        sum_38: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_966, [0, 1]);  convert_element_type_966 = None
        add_225: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_201, mul_217);  mul_201 = mul_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_969: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_225, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_543: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_969, [4096, 2048]);  convert_element_type_969 = None
        mm_79: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_543, permute_277);  permute_277 = None
        permute_278: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_543, [1, 0])
        mm_80: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_278, view_522);  permute_278 = view_522 = None
        sum_39: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_543, [0], True, dtype = torch.float32);  view_543 = None
        view_544: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_39, [2048]);  sum_39 = None
        convert_element_type_974: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_544, torch.bfloat16);  view_544 = None
        view_545: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_79, [32, 128, 2048]);  mm_79 = None
        convert_element_type_975: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_80, torch.float32);  mm_80 = None
        convert_element_type_976: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_974, torch.float32);  convert_element_type_974 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_546: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_545, [32, 128, 16, 128]);  view_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_281: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_170: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_281, memory_format = torch.contiguous_format);  permute_281 = None
        view_547: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_170, [512, 128, 128]);  clone_170 = None
        bmm_48: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_282, view_547);  permute_282 = None
        bmm_49: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_547, permute_283);  view_547 = permute_283 = None
        view_548: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_48, [32, 16, 128, 128]);  bmm_48 = None
        view_549: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_49, [32, 16, 128, 128]);  bmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_981: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_549, torch.float32);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_219: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_981, div_23);  convert_element_type_981 = None
        sum_40: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_219, [-1], True)
        neg_1: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_23);  div_23 = None
        fma: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_1, sum_40, mul_219);  neg_1 = sum_40 = mul_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_982: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        full_default_31: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_50: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_331, 2, 0, 128);  primals_331 = None
        slice_51: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_50, 3, 0, 128);  slice_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_29: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_51, convert_element_type_982, full_default_31);  slice_51 = convert_element_type_982 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_550: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_29, [512, 128, 128]);  where_29 = None
        bmm_50: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_284, view_550);  permute_284 = None
        bmm_51: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_550, permute_285);  view_550 = permute_285 = None
        view_551: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_50, [32, 16, 128, 128]);  bmm_50 = None
        view_552: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_51, [32, 16, 128, 128]);  bmm_51 = None
        convert_element_type_988: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_551, torch.float32);  view_551 = None
        permute_286: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_988, [0, 1, 3, 2]);  convert_element_type_988 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_989: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_286, torch.bfloat16);  permute_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_23: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_552, torch.bfloat16);  view_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_287: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_548, [0, 2, 1, 3]);  view_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_171: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
        view_553: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_171, [32, 128, 2048]);  clone_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_288: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_989, [0, 2, 1, 3]);  convert_element_type_989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_554: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_288, [32, 128, 2048]);  permute_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_289: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_23, [0, 2, 1, 3]);  convert_element_type_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_172: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
        view_555: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_172, [32, 128, 2048]);  clone_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_556: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_553, [4096, 2048]);  view_553 = None
        permute_290: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_556, [1, 0])
        mm_81: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_290, view_506);  permute_290 = None
        mm_82: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_556, permute_292);  view_556 = permute_292 = None
        view_557: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [32, 128, 2048]);  mm_82 = None
        convert_element_type_995: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_557, torch.float32);  view_557 = None
        convert_element_type_996: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_173: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_554, memory_format = torch.contiguous_format);  view_554 = None
        view_558: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_173, [4096, 2048]);  clone_173 = None
        permute_294: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_558, [1, 0])
        mm_83: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_294, view_506);  permute_294 = None
        mm_84: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_558, permute_296);  view_558 = permute_296 = None
        view_559: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [32, 128, 2048]);  mm_84 = None
        convert_element_type_1001: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_559, torch.float32);  view_559 = None
        add_226: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_995, convert_element_type_1001);  convert_element_type_995 = convert_element_type_1001 = None
        convert_element_type_1002: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_560: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_555, [4096, 2048]);  view_555 = None
        permute_298: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_560, [1, 0])
        mm_85: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_298, view_506);  permute_298 = view_506 = None
        mm_86: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_560, permute_300);  view_560 = permute_300 = None
        view_561: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [32, 128, 2048]);  mm_86 = None
        convert_element_type_1007: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_561, torch.float32);  view_561 = None
        add_227: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_226, convert_element_type_1007);  add_226 = convert_element_type_1007 = None
        convert_element_type_1008: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_221: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_227, primals_326);  primals_326 = None
        mul_222: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_221, 2048)
        sum_41: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_221, [2], True)
        mul_223: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_221, mul_184);  mul_221 = None
        sum_42: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_223, [2], True);  mul_223 = None
        mul_224: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, sum_42);  sum_42 = None
        sub_86: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_222, sum_41);  mul_222 = sum_41 = None
        sub_87: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_86, mul_224);  sub_86 = mul_224 = None
        mul_225: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_28, sub_87);  div_28 = sub_87 = None
        mul_226: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_227, mul_184);  mul_184 = None
        sum_43: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_226, [0, 1]);  mul_226 = None
        sum_44: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_227, [0, 1]);  add_227 = None
        add_228: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_225, mul_225);  add_225 = mul_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1009: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_228, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_562: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1009, [4096, 2048]);  convert_element_type_1009 = None
        mm_87: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_562, permute_302);  permute_302 = None
        permute_303: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_562, [1, 0])
        mm_88: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_303, view_504);  permute_303 = view_504 = None
        sum_45: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_562, [0], True, dtype = torch.float32);  view_562 = None
        view_563: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_45, [2048]);  sum_45 = None
        convert_element_type_1014: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_563, torch.bfloat16);  view_563 = None
        view_564: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_87, [32, 128, 8192]);  mm_87 = None
        convert_element_type_1015: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_564, torch.float32);  view_564 = None
        convert_element_type_1016: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_88, torch.float32);  mm_88 = None
        convert_element_type_1017: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1014, torch.float32);  convert_element_type_1014 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_503: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [32, 128, 8192]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_180: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_503, 0.5)
        mul_227: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1015, mul_180);  mul_180 = None
        convert_element_type_890: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_503, torch.float32)
        pow_23: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_890, 3.0)
        mul_181: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_208: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_503, mul_181);  view_503 = mul_181 = None
        mul_182: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_208, 0.7978845608028654);  add_208 = None
        tanh_22: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_182);  mul_182 = None
        add_209: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_22, 1.0)
        mul_228: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1015, add_209);  convert_element_type_1015 = add_209 = None
        convert_element_type_1018: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_228, torch.bfloat16);  mul_228 = None
        mul_229: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_22, tanh_22);  tanh_22 = None
        sub_88: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_229);  mul_229 = None
        mul_230: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, sub_88);  mul_227 = sub_88 = None
        mul_231: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, 0.7978845608028654);  mul_230 = None
        convert_element_type_1019: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_231, torch.bfloat16)
        mul_232: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_231, 0.044715);  mul_231 = None
        pow_26: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_890, 2.0);  convert_element_type_890 = None
        mul_233: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_26, 3.0);  pow_26 = None
        mul_234: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        convert_element_type_1020: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_234, torch.bfloat16);  mul_234 = None
        add_229: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1019, convert_element_type_1020);  convert_element_type_1019 = convert_element_type_1020 = None
        mul_235: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1018, 0.5);  convert_element_type_1018 = None
        add_230: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_229, mul_235);  add_229 = mul_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_565: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_230, [4096, 8192]);  add_230 = None
        mm_89: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_565, permute_306);  permute_306 = None
        permute_307: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_565, [1, 0])
        mm_90: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_307, view_502);  permute_307 = view_502 = None
        sum_46: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_565, [0], True, dtype = torch.float32);  view_565 = None
        view_566: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_46, [8192]);  sum_46 = None
        convert_element_type_1025: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_566, torch.bfloat16);  view_566 = None
        view_567: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_89, [32, 128, 2048]);  mm_89 = None
        convert_element_type_1026: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_567, torch.float32);  view_567 = None
        convert_element_type_1027: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_90, torch.float32);  mm_90 = None
        convert_element_type_1028: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1025, torch.float32);  convert_element_type_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_237: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1026, primals_320);  primals_320 = None
        mul_238: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_237, 2048)
        sum_47: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_237, [2], True)
        mul_239: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_237, mul_178);  mul_237 = None
        sum_48: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_239, [2], True);  mul_239 = None
        mul_240: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, sum_48);  sum_48 = None
        sub_90: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_238, sum_47);  mul_238 = sum_47 = None
        sub_91: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, mul_240);  sub_90 = mul_240 = None
        mul_241: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_29, sub_91);  div_29 = sub_91 = None
        mul_242: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1026, mul_178);  mul_178 = None
        sum_49: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_242, [0, 1]);  mul_242 = None
        sum_50: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1026, [0, 1]);  convert_element_type_1026 = None
        add_231: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_228, mul_241);  add_228 = mul_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1029: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_231, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_568: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1029, [4096, 2048]);  convert_element_type_1029 = None
        mm_91: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_568, permute_310);  permute_310 = None
        permute_311: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_568, [1, 0])
        mm_92: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_311, view_500);  permute_311 = view_500 = None
        sum_51: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_568, [0], True, dtype = torch.float32);  view_568 = None
        view_569: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_51, [2048]);  sum_51 = None
        convert_element_type_1034: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_569, torch.bfloat16);  view_569 = None
        view_570: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_91, [32, 128, 2048]);  mm_91 = None
        convert_element_type_1035: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        convert_element_type_1036: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1034, torch.float32);  convert_element_type_1034 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_571: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_570, [32, 128, 16, 128]);  view_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_314: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_571, [0, 2, 1, 3]);  view_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_174: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
        view_572: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_174, [512, 128, 128]);  clone_174 = None
        bmm_52: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_315, view_572);  permute_315 = None
        bmm_53: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_572, permute_316);  view_572 = permute_316 = None
        view_573: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_52, [32, 16, 128, 128]);  bmm_52 = None
        view_574: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_53, [32, 16, 128, 128]);  bmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1041: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_574, torch.float32);  view_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_243: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1041, div_22);  convert_element_type_1041 = None
        sum_52: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_243, [-1], True)
        neg_2: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_22);  div_22 = None
        fma_1: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_2, sum_52, mul_243);  neg_2 = sum_52 = mul_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1042: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_48: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_317, 2, 0, 128);  primals_317 = None
        slice_49: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_48, 3, 0, 128);  slice_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_30: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_49, convert_element_type_1042, full_default_31);  slice_49 = convert_element_type_1042 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_575: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_30, [512, 128, 128]);  where_30 = None
        bmm_54: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_317, view_575);  permute_317 = None
        bmm_55: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_575, permute_318);  view_575 = permute_318 = None
        view_576: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_54, [32, 16, 128, 128]);  bmm_54 = None
        view_577: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_55, [32, 16, 128, 128]);  bmm_55 = None
        convert_element_type_1048: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_576, torch.float32);  view_576 = None
        permute_319: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1048, [0, 1, 3, 2]);  convert_element_type_1048 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1049: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_319, torch.bfloat16);  permute_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_22: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_577, torch.bfloat16);  view_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_320: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_175: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
        view_578: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_175, [32, 128, 2048]);  clone_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_321: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1049, [0, 2, 1, 3]);  convert_element_type_1049 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_579: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_321, [32, 128, 2048]);  permute_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_322: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_22, [0, 2, 1, 3]);  convert_element_type_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_176: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
        view_580: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_176, [32, 128, 2048]);  clone_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_581: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_578, [4096, 2048]);  view_578 = None
        permute_323: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_581, [1, 0])
        mm_93: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_323, view_484);  permute_323 = None
        mm_94: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_581, permute_325);  view_581 = permute_325 = None
        view_582: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [32, 128, 2048]);  mm_94 = None
        convert_element_type_1055: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_582, torch.float32);  view_582 = None
        convert_element_type_1056: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_177: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_579, memory_format = torch.contiguous_format);  view_579 = None
        view_583: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_177, [4096, 2048]);  clone_177 = None
        permute_327: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_583, [1, 0])
        mm_95: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_327, view_484);  permute_327 = None
        mm_96: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_583, permute_329);  view_583 = permute_329 = None
        view_584: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [32, 128, 2048]);  mm_96 = None
        convert_element_type_1061: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_584, torch.float32);  view_584 = None
        add_232: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1055, convert_element_type_1061);  convert_element_type_1055 = convert_element_type_1061 = None
        convert_element_type_1062: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_585: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_580, [4096, 2048]);  view_580 = None
        permute_331: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_585, [1, 0])
        mm_97: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_331, view_484);  permute_331 = view_484 = None
        mm_98: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_585, permute_333);  view_585 = permute_333 = None
        view_586: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [32, 128, 2048]);  mm_98 = None
        convert_element_type_1067: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_586, torch.float32);  view_586 = None
        add_233: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_232, convert_element_type_1067);  add_232 = convert_element_type_1067 = None
        convert_element_type_1068: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_245: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_233, primals_312);  primals_312 = None
        mul_246: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_245, 2048)
        sum_53: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True)
        mul_247: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_245, mul_176);  mul_245 = None
        sum_54: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_247, [2], True);  mul_247 = None
        mul_248: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, sum_54);  sum_54 = None
        sub_93: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_246, sum_53);  mul_246 = sum_53 = None
        sub_94: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_93, mul_248);  sub_93 = mul_248 = None
        mul_249: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_30, sub_94);  div_30 = sub_94 = None
        mul_250: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_233, mul_176);  mul_176 = None
        sum_55: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_250, [0, 1]);  mul_250 = None
        sum_56: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None
        add_234: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_231, mul_249);  add_231 = mul_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1069: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_234, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_587: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1069, [4096, 2048]);  convert_element_type_1069 = None
        mm_99: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_587, permute_335);  permute_335 = None
        permute_336: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_587, [1, 0])
        mm_100: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_336, view_482);  permute_336 = view_482 = None
        sum_57: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_587, [0], True, dtype = torch.float32);  view_587 = None
        view_588: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_57, [2048]);  sum_57 = None
        convert_element_type_1074: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_588, torch.bfloat16);  view_588 = None
        view_589: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_99, [32, 128, 8192]);  mm_99 = None
        convert_element_type_1075: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_589, torch.float32);  view_589 = None
        convert_element_type_1076: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_100, torch.float32);  mm_100 = None
        convert_element_type_1077: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1074, torch.float32);  convert_element_type_1074 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_481: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [32, 128, 8192]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_172: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_481, 0.5)
        mul_251: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1075, mul_172);  mul_172 = None
        convert_element_type_851: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_481, torch.float32)
        pow_22: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_851, 3.0)
        mul_173: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_199: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_481, mul_173);  view_481 = mul_173 = None
        mul_174: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_199, 0.7978845608028654);  add_199 = None
        tanh_21: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_174);  mul_174 = None
        add_200: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_21, 1.0)
        mul_252: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1075, add_200);  convert_element_type_1075 = add_200 = None
        convert_element_type_1078: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_252, torch.bfloat16);  mul_252 = None
        mul_253: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_21, tanh_21);  tanh_21 = None
        sub_95: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_253);  mul_253 = None
        mul_254: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_251, sub_95);  mul_251 = sub_95 = None
        mul_255: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_254, 0.7978845608028654);  mul_254 = None
        convert_element_type_1079: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_255, torch.bfloat16)
        mul_256: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_255, 0.044715);  mul_255 = None
        pow_27: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_851, 2.0);  convert_element_type_851 = None
        mul_257: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_27, 3.0);  pow_27 = None
        mul_258: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        convert_element_type_1080: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_258, torch.bfloat16);  mul_258 = None
        add_235: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1079, convert_element_type_1080);  convert_element_type_1079 = convert_element_type_1080 = None
        mul_259: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1078, 0.5);  convert_element_type_1078 = None
        add_236: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_235, mul_259);  add_235 = mul_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_590: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_236, [4096, 8192]);  add_236 = None
        mm_101: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_590, permute_339);  permute_339 = None
        permute_340: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_590, [1, 0])
        mm_102: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_340, view_480);  permute_340 = view_480 = None
        sum_58: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_590, [0], True, dtype = torch.float32);  view_590 = None
        view_591: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_58, [8192]);  sum_58 = None
        convert_element_type_1085: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_591, torch.bfloat16);  view_591 = None
        view_592: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_101, [32, 128, 2048]);  mm_101 = None
        convert_element_type_1086: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_592, torch.float32);  view_592 = None
        convert_element_type_1087: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_102, torch.float32);  mm_102 = None
        convert_element_type_1088: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1085, torch.float32);  convert_element_type_1085 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_261: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1086, primals_306);  primals_306 = None
        mul_262: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_261, 2048)
        sum_59: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_261, [2], True)
        mul_263: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_261, mul_170);  mul_261 = None
        sum_60: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_263, [2], True);  mul_263 = None
        mul_264: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_170, sum_60);  sum_60 = None
        sub_97: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_262, sum_59);  mul_262 = sum_59 = None
        sub_98: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_97, mul_264);  sub_97 = mul_264 = None
        mul_265: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_31, sub_98);  div_31 = sub_98 = None
        mul_266: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1086, mul_170);  mul_170 = None
        sum_61: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_266, [0, 1]);  mul_266 = None
        sum_62: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1086, [0, 1]);  convert_element_type_1086 = None
        add_237: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_234, mul_265);  add_234 = mul_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1089: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_237, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_593: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1089, [4096, 2048]);  convert_element_type_1089 = None
        mm_103: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_593, permute_343);  permute_343 = None
        permute_344: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_593, [1, 0])
        mm_104: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_344, view_478);  permute_344 = view_478 = None
        sum_63: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_593, [0], True, dtype = torch.float32);  view_593 = None
        view_594: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [2048]);  sum_63 = None
        convert_element_type_1094: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_594, torch.bfloat16);  view_594 = None
        view_595: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_103, [32, 128, 2048]);  mm_103 = None
        convert_element_type_1095: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_104, torch.float32);  mm_104 = None
        convert_element_type_1096: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1094, torch.float32);  convert_element_type_1094 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_596: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_595, [32, 128, 16, 128]);  view_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_347: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_178: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
        view_597: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_178, [512, 128, 128]);  clone_178 = None
        bmm_56: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_348, view_597);  permute_348 = None
        bmm_57: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_597, permute_349);  view_597 = permute_349 = None
        view_598: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_56, [32, 16, 128, 128]);  bmm_56 = None
        view_599: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_57, [32, 16, 128, 128]);  bmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1101: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_599, torch.float32);  view_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_267: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1101, div_21);  convert_element_type_1101 = None
        sum_64: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_267, [-1], True)
        neg_3: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_21);  div_21 = None
        fma_2: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_3, sum_64, mul_267);  neg_3 = sum_64 = mul_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1102: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_46: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_303, 2, 0, 128);  primals_303 = None
        slice_47: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_46, 3, 0, 128);  slice_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_31: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_47, convert_element_type_1102, full_default_31);  slice_47 = convert_element_type_1102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_600: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_31, [512, 128, 128]);  where_31 = None
        bmm_58: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_350, view_600);  permute_350 = None
        bmm_59: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_600, permute_351);  view_600 = permute_351 = None
        view_601: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_58, [32, 16, 128, 128]);  bmm_58 = None
        view_602: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_59, [32, 16, 128, 128]);  bmm_59 = None
        convert_element_type_1108: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_601, torch.float32);  view_601 = None
        permute_352: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1108, [0, 1, 3, 2]);  convert_element_type_1108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1109: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_352, torch.bfloat16);  permute_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_21: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_602, torch.bfloat16);  view_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_353: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_179: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
        view_603: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_179, [32, 128, 2048]);  clone_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_354: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1109, [0, 2, 1, 3]);  convert_element_type_1109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_604: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_354, [32, 128, 2048]);  permute_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_355: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_21, [0, 2, 1, 3]);  convert_element_type_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_180: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
        view_605: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_180, [32, 128, 2048]);  clone_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_606: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_603, [4096, 2048]);  view_603 = None
        permute_356: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_606, [1, 0])
        mm_105: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_356, view_462);  permute_356 = None
        mm_106: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_606, permute_358);  view_606 = permute_358 = None
        view_607: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [32, 128, 2048]);  mm_106 = None
        convert_element_type_1115: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_607, torch.float32);  view_607 = None
        convert_element_type_1116: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_181: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_604, memory_format = torch.contiguous_format);  view_604 = None
        view_608: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_181, [4096, 2048]);  clone_181 = None
        permute_360: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_608, [1, 0])
        mm_107: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_360, view_462);  permute_360 = None
        mm_108: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_608, permute_362);  view_608 = permute_362 = None
        view_609: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [32, 128, 2048]);  mm_108 = None
        convert_element_type_1121: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_609, torch.float32);  view_609 = None
        add_238: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1115, convert_element_type_1121);  convert_element_type_1115 = convert_element_type_1121 = None
        convert_element_type_1122: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_610: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_605, [4096, 2048]);  view_605 = None
        permute_364: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_610, [1, 0])
        mm_109: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_364, view_462);  permute_364 = view_462 = None
        mm_110: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_610, permute_366);  view_610 = permute_366 = None
        view_611: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [32, 128, 2048]);  mm_110 = None
        convert_element_type_1127: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_611, torch.float32);  view_611 = None
        add_239: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_238, convert_element_type_1127);  add_238 = convert_element_type_1127 = None
        convert_element_type_1128: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_269: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_239, primals_298);  primals_298 = None
        mul_270: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, 2048)
        sum_65: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True)
        mul_271: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, mul_168);  mul_269 = None
        sum_66: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True);  mul_271 = None
        mul_272: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, sum_66);  sum_66 = None
        sub_100: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_270, sum_65);  mul_270 = sum_65 = None
        sub_101: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_100, mul_272);  sub_100 = mul_272 = None
        mul_273: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_32, sub_101);  div_32 = sub_101 = None
        mul_274: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_239, mul_168);  mul_168 = None
        sum_67: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1]);  mul_274 = None
        sum_68: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_239, [0, 1]);  add_239 = None
        add_240: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_237, mul_273);  add_237 = mul_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1129: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_240, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_612: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1129, [4096, 2048]);  convert_element_type_1129 = None
        mm_111: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_612, permute_368);  permute_368 = None
        permute_369: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_612, [1, 0])
        mm_112: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_369, view_460);  permute_369 = view_460 = None
        sum_69: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_612, [0], True, dtype = torch.float32);  view_612 = None
        view_613: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_69, [2048]);  sum_69 = None
        convert_element_type_1134: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_613, torch.bfloat16);  view_613 = None
        view_614: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_111, [32, 128, 8192]);  mm_111 = None
        convert_element_type_1135: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_614, torch.float32);  view_614 = None
        convert_element_type_1136: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_112, torch.float32);  mm_112 = None
        convert_element_type_1137: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1134, torch.float32);  convert_element_type_1134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_459: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [32, 128, 8192]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_164: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_459, 0.5)
        mul_275: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1135, mul_164);  mul_164 = None
        convert_element_type_812: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_459, torch.float32)
        pow_21: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_812, 3.0)
        mul_165: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_190: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_459, mul_165);  view_459 = mul_165 = None
        mul_166: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_190, 0.7978845608028654);  add_190 = None
        tanh_20: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
        add_191: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_20, 1.0)
        mul_276: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1135, add_191);  convert_element_type_1135 = add_191 = None
        convert_element_type_1138: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_276, torch.bfloat16);  mul_276 = None
        mul_277: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_20, tanh_20);  tanh_20 = None
        sub_102: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_277);  mul_277 = None
        mul_278: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_275, sub_102);  mul_275 = sub_102 = None
        mul_279: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_278, 0.7978845608028654);  mul_278 = None
        convert_element_type_1139: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_279, torch.bfloat16)
        mul_280: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_279, 0.044715);  mul_279 = None
        pow_28: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_812, 2.0);  convert_element_type_812 = None
        mul_281: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_28, 3.0);  pow_28 = None
        mul_282: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_280, mul_281);  mul_280 = mul_281 = None
        convert_element_type_1140: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_282, torch.bfloat16);  mul_282 = None
        add_241: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1139, convert_element_type_1140);  convert_element_type_1139 = convert_element_type_1140 = None
        mul_283: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1138, 0.5);  convert_element_type_1138 = None
        add_242: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_241, mul_283);  add_241 = mul_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_615: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_242, [4096, 8192]);  add_242 = None
        mm_113: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_615, permute_372);  permute_372 = None
        permute_373: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_615, [1, 0])
        mm_114: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_373, view_458);  permute_373 = view_458 = None
        sum_70: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_615, [0], True, dtype = torch.float32);  view_615 = None
        view_616: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [8192]);  sum_70 = None
        convert_element_type_1145: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_616, torch.bfloat16);  view_616 = None
        view_617: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_113, [32, 128, 2048]);  mm_113 = None
        convert_element_type_1146: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_617, torch.float32);  view_617 = None
        convert_element_type_1147: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_114, torch.float32);  mm_114 = None
        convert_element_type_1148: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1145, torch.float32);  convert_element_type_1145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_285: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1146, primals_292);  primals_292 = None
        mul_286: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_285, 2048)
        sum_71: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_285, [2], True)
        mul_287: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_285, mul_162);  mul_285 = None
        sum_72: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True);  mul_287 = None
        mul_288: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, sum_72);  sum_72 = None
        sub_104: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_286, sum_71);  mul_286 = sum_71 = None
        sub_105: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_104, mul_288);  sub_104 = mul_288 = None
        mul_289: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_33, sub_105);  div_33 = sub_105 = None
        mul_290: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1146, mul_162);  mul_162 = None
        sum_73: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_290, [0, 1]);  mul_290 = None
        sum_74: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1146, [0, 1]);  convert_element_type_1146 = None
        add_243: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_240, mul_289);  add_240 = mul_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1149: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_243, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_618: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1149, [4096, 2048]);  convert_element_type_1149 = None
        mm_115: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_618, permute_376);  permute_376 = None
        permute_377: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_618, [1, 0])
        mm_116: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_377, view_456);  permute_377 = view_456 = None
        sum_75: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_618, [0], True, dtype = torch.float32);  view_618 = None
        view_619: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_75, [2048]);  sum_75 = None
        convert_element_type_1154: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_619, torch.bfloat16);  view_619 = None
        view_620: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_115, [32, 128, 2048]);  mm_115 = None
        convert_element_type_1155: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_116, torch.float32);  mm_116 = None
        convert_element_type_1156: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1154, torch.float32);  convert_element_type_1154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_621: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_620, [32, 128, 16, 128]);  view_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_380: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_621, [0, 2, 1, 3]);  view_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_182: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_380, memory_format = torch.contiguous_format);  permute_380 = None
        view_622: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_182, [512, 128, 128]);  clone_182 = None
        bmm_60: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_381, view_622);  permute_381 = None
        bmm_61: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_622, permute_382);  view_622 = permute_382 = None
        view_623: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_60, [32, 16, 128, 128]);  bmm_60 = None
        view_624: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_61, [32, 16, 128, 128]);  bmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1161: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_624, torch.float32);  view_624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_291: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1161, div_20);  convert_element_type_1161 = None
        sum_76: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_291, [-1], True)
        neg_4: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_20);  div_20 = None
        fma_3: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_4, sum_76, mul_291);  neg_4 = sum_76 = mul_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1162: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_44: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_289, 2, 0, 128);  primals_289 = None
        slice_45: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_44, 3, 0, 128);  slice_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_32: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_45, convert_element_type_1162, full_default_31);  slice_45 = convert_element_type_1162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_625: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_32, [512, 128, 128]);  where_32 = None
        bmm_62: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_383, view_625);  permute_383 = None
        bmm_63: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_625, permute_384);  view_625 = permute_384 = None
        view_626: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_62, [32, 16, 128, 128]);  bmm_62 = None
        view_627: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_63, [32, 16, 128, 128]);  bmm_63 = None
        convert_element_type_1168: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_626, torch.float32);  view_626 = None
        permute_385: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1168, [0, 1, 3, 2]);  convert_element_type_1168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1169: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_385, torch.bfloat16);  permute_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_20: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_627, torch.bfloat16);  view_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_386: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_623, [0, 2, 1, 3]);  view_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_183: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
        view_628: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_183, [32, 128, 2048]);  clone_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_387: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1169, [0, 2, 1, 3]);  convert_element_type_1169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_629: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_387, [32, 128, 2048]);  permute_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_388: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_20, [0, 2, 1, 3]);  convert_element_type_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_184: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_388, memory_format = torch.contiguous_format);  permute_388 = None
        view_630: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_184, [32, 128, 2048]);  clone_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_631: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_628, [4096, 2048]);  view_628 = None
        permute_389: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_631, [1, 0])
        mm_117: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_389, view_440);  permute_389 = None
        mm_118: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_631, permute_391);  view_631 = permute_391 = None
        view_632: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_118, [32, 128, 2048]);  mm_118 = None
        convert_element_type_1175: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_632, torch.float32);  view_632 = None
        convert_element_type_1176: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_185: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_629, memory_format = torch.contiguous_format);  view_629 = None
        view_633: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_185, [4096, 2048]);  clone_185 = None
        permute_393: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_633, [1, 0])
        mm_119: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_393, view_440);  permute_393 = None
        mm_120: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_633, permute_395);  view_633 = permute_395 = None
        view_634: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_120, [32, 128, 2048]);  mm_120 = None
        convert_element_type_1181: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_634, torch.float32);  view_634 = None
        add_244: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1175, convert_element_type_1181);  convert_element_type_1175 = convert_element_type_1181 = None
        convert_element_type_1182: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_635: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_630, [4096, 2048]);  view_630 = None
        permute_397: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_635, [1, 0])
        mm_121: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_397, view_440);  permute_397 = view_440 = None
        mm_122: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_635, permute_399);  view_635 = permute_399 = None
        view_636: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [32, 128, 2048]);  mm_122 = None
        convert_element_type_1187: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_636, torch.float32);  view_636 = None
        add_245: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_244, convert_element_type_1187);  add_244 = convert_element_type_1187 = None
        convert_element_type_1188: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_293: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_245, primals_284);  primals_284 = None
        mul_294: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_293, 2048)
        sum_77: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_293, [2], True)
        mul_295: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_293, mul_160);  mul_293 = None
        sum_78: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_295, [2], True);  mul_295 = None
        mul_296: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, sum_78);  sum_78 = None
        sub_107: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_294, sum_77);  mul_294 = sum_77 = None
        sub_108: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_107, mul_296);  sub_107 = mul_296 = None
        mul_297: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_34, sub_108);  div_34 = sub_108 = None
        mul_298: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_245, mul_160);  mul_160 = None
        sum_79: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_298, [0, 1]);  mul_298 = None
        sum_80: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_245, [0, 1]);  add_245 = None
        add_246: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_243, mul_297);  add_243 = mul_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1189: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_246, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_637: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1189, [4096, 2048]);  convert_element_type_1189 = None
        mm_123: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_637, permute_401);  permute_401 = None
        permute_402: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_637, [1, 0])
        mm_124: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_402, view_438);  permute_402 = view_438 = None
        sum_81: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_637, [0], True, dtype = torch.float32);  view_637 = None
        view_638: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_81, [2048]);  sum_81 = None
        convert_element_type_1194: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_638, torch.bfloat16);  view_638 = None
        view_639: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_123, [32, 128, 8192]);  mm_123 = None
        convert_element_type_1195: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_639, torch.float32);  view_639 = None
        convert_element_type_1196: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_124, torch.float32);  mm_124 = None
        convert_element_type_1197: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1194, torch.float32);  convert_element_type_1194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_437: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [32, 128, 8192]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_156: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_437, 0.5)
        mul_299: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1195, mul_156);  mul_156 = None
        convert_element_type_773: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_437, torch.float32)
        pow_20: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_773, 3.0)
        mul_157: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_181: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_437, mul_157);  view_437 = mul_157 = None
        mul_158: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_181, 0.7978845608028654);  add_181 = None
        tanh_19: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
        add_182: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_19, 1.0)
        mul_300: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1195, add_182);  convert_element_type_1195 = add_182 = None
        convert_element_type_1198: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_300, torch.bfloat16);  mul_300 = None
        mul_301: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_19, tanh_19);  tanh_19 = None
        sub_109: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_301);  mul_301 = None
        mul_302: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_299, sub_109);  mul_299 = sub_109 = None
        mul_303: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_302, 0.7978845608028654);  mul_302 = None
        convert_element_type_1199: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_303, torch.bfloat16)
        mul_304: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_303, 0.044715);  mul_303 = None
        pow_29: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_773, 2.0);  convert_element_type_773 = None
        mul_305: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_29, 3.0);  pow_29 = None
        mul_306: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_304, mul_305);  mul_304 = mul_305 = None
        convert_element_type_1200: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_306, torch.bfloat16);  mul_306 = None
        add_247: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1199, convert_element_type_1200);  convert_element_type_1199 = convert_element_type_1200 = None
        mul_307: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1198, 0.5);  convert_element_type_1198 = None
        add_248: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_247, mul_307);  add_247 = mul_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_640: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_248, [4096, 8192]);  add_248 = None
        mm_125: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_640, permute_405);  permute_405 = None
        permute_406: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_640, [1, 0])
        mm_126: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_406, view_436);  permute_406 = view_436 = None
        sum_82: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_640, [0], True, dtype = torch.float32);  view_640 = None
        view_641: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_82, [8192]);  sum_82 = None
        convert_element_type_1205: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_641, torch.bfloat16);  view_641 = None
        view_642: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_125, [32, 128, 2048]);  mm_125 = None
        convert_element_type_1206: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_642, torch.float32);  view_642 = None
        convert_element_type_1207: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_126, torch.float32);  mm_126 = None
        convert_element_type_1208: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1205, torch.float32);  convert_element_type_1205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_309: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1206, primals_278);  primals_278 = None
        mul_310: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_309, 2048)
        sum_83: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True)
        mul_311: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_309, mul_154);  mul_309 = None
        sum_84: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True);  mul_311 = None
        mul_312: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, sum_84);  sum_84 = None
        sub_111: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_310, sum_83);  mul_310 = sum_83 = None
        sub_112: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_111, mul_312);  sub_111 = mul_312 = None
        mul_313: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_35, sub_112);  div_35 = sub_112 = None
        mul_314: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1206, mul_154);  mul_154 = None
        sum_85: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 1]);  mul_314 = None
        sum_86: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1206, [0, 1]);  convert_element_type_1206 = None
        add_249: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_246, mul_313);  add_246 = mul_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1209: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_249, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_643: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1209, [4096, 2048]);  convert_element_type_1209 = None
        mm_127: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_643, permute_409);  permute_409 = None
        permute_410: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_643, [1, 0])
        mm_128: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_410, view_434);  permute_410 = view_434 = None
        sum_87: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_643, [0], True, dtype = torch.float32);  view_643 = None
        view_644: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_87, [2048]);  sum_87 = None
        convert_element_type_1214: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_644, torch.bfloat16);  view_644 = None
        view_645: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_127, [32, 128, 2048]);  mm_127 = None
        convert_element_type_1215: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_128, torch.float32);  mm_128 = None
        convert_element_type_1216: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1214, torch.float32);  convert_element_type_1214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_646: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_645, [32, 128, 16, 128]);  view_645 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_413: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_646, [0, 2, 1, 3]);  view_646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_186: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_413, memory_format = torch.contiguous_format);  permute_413 = None
        view_647: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_186, [512, 128, 128]);  clone_186 = None
        bmm_64: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_414, view_647);  permute_414 = None
        bmm_65: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_647, permute_415);  view_647 = permute_415 = None
        view_648: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_64, [32, 16, 128, 128]);  bmm_64 = None
        view_649: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_65, [32, 16, 128, 128]);  bmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1221: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_649, torch.float32);  view_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_315: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1221, div_19);  convert_element_type_1221 = None
        sum_88: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_315, [-1], True)
        neg_5: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_19);  div_19 = None
        fma_4: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_5, sum_88, mul_315);  neg_5 = sum_88 = mul_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1222: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_42: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_275, 2, 0, 128);  primals_275 = None
        slice_43: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_42, 3, 0, 128);  slice_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_33: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_43, convert_element_type_1222, full_default_31);  slice_43 = convert_element_type_1222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_650: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_33, [512, 128, 128]);  where_33 = None
        bmm_66: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_416, view_650);  permute_416 = None
        bmm_67: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_650, permute_417);  view_650 = permute_417 = None
        view_651: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_66, [32, 16, 128, 128]);  bmm_66 = None
        view_652: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_67, [32, 16, 128, 128]);  bmm_67 = None
        convert_element_type_1228: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_651, torch.float32);  view_651 = None
        permute_418: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1228, [0, 1, 3, 2]);  convert_element_type_1228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1229: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_418, torch.bfloat16);  permute_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_19: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_652, torch.bfloat16);  view_652 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_419: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_187: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
        view_653: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_187, [32, 128, 2048]);  clone_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_420: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1229, [0, 2, 1, 3]);  convert_element_type_1229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_654: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_420, [32, 128, 2048]);  permute_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_421: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_19, [0, 2, 1, 3]);  convert_element_type_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_188: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_421, memory_format = torch.contiguous_format);  permute_421 = None
        view_655: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_188, [32, 128, 2048]);  clone_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_656: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_653, [4096, 2048]);  view_653 = None
        permute_422: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_656, [1, 0])
        mm_129: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_422, view_418);  permute_422 = None
        mm_130: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_656, permute_424);  view_656 = permute_424 = None
        view_657: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_130, [32, 128, 2048]);  mm_130 = None
        convert_element_type_1235: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_657, torch.float32);  view_657 = None
        convert_element_type_1236: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_189: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_654, memory_format = torch.contiguous_format);  view_654 = None
        view_658: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_189, [4096, 2048]);  clone_189 = None
        permute_426: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_658, [1, 0])
        mm_131: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_426, view_418);  permute_426 = None
        mm_132: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_658, permute_428);  view_658 = permute_428 = None
        view_659: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [32, 128, 2048]);  mm_132 = None
        convert_element_type_1241: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_659, torch.float32);  view_659 = None
        add_250: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1235, convert_element_type_1241);  convert_element_type_1235 = convert_element_type_1241 = None
        convert_element_type_1242: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_660: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_655, [4096, 2048]);  view_655 = None
        permute_430: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_660, [1, 0])
        mm_133: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_430, view_418);  permute_430 = view_418 = None
        mm_134: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_660, permute_432);  view_660 = permute_432 = None
        view_661: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [32, 128, 2048]);  mm_134 = None
        convert_element_type_1247: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_661, torch.float32);  view_661 = None
        add_251: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_250, convert_element_type_1247);  add_250 = convert_element_type_1247 = None
        convert_element_type_1248: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_317: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_251, primals_270);  primals_270 = None
        mul_318: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_317, 2048)
        sum_89: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_317, [2], True)
        mul_319: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_317, mul_152);  mul_317 = None
        sum_90: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True);  mul_319 = None
        mul_320: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, sum_90);  sum_90 = None
        sub_114: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_318, sum_89);  mul_318 = sum_89 = None
        sub_115: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_114, mul_320);  sub_114 = mul_320 = None
        mul_321: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_36, sub_115);  div_36 = sub_115 = None
        mul_322: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_251, mul_152);  mul_152 = None
        sum_91: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_322, [0, 1]);  mul_322 = None
        sum_92: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_251, [0, 1]);  add_251 = None
        add_252: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_249, mul_321);  add_249 = mul_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1249: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_252, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_662: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1249, [4096, 2048]);  convert_element_type_1249 = None
        mm_135: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_662, permute_434);  permute_434 = None
        permute_435: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_662, [1, 0])
        mm_136: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_435, view_416);  permute_435 = view_416 = None
        sum_93: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_662, [0], True, dtype = torch.float32);  view_662 = None
        view_663: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_93, [2048]);  sum_93 = None
        convert_element_type_1254: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_663, torch.bfloat16);  view_663 = None
        view_664: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_135, [32, 128, 8192]);  mm_135 = None
        convert_element_type_1255: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_664, torch.float32);  view_664 = None
        convert_element_type_1256: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_136, torch.float32);  mm_136 = None
        convert_element_type_1257: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1254, torch.float32);  convert_element_type_1254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_415: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [32, 128, 8192]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_148: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_415, 0.5)
        mul_323: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1255, mul_148);  mul_148 = None
        convert_element_type_734: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_415, torch.float32)
        pow_19: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_734, 3.0)
        mul_149: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_172: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_415, mul_149);  view_415 = mul_149 = None
        mul_150: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_172, 0.7978845608028654);  add_172 = None
        tanh_18: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_150);  mul_150 = None
        add_173: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_18, 1.0)
        mul_324: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1255, add_173);  convert_element_type_1255 = add_173 = None
        convert_element_type_1258: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_324, torch.bfloat16);  mul_324 = None
        mul_325: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_18, tanh_18);  tanh_18 = None
        sub_116: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_325);  mul_325 = None
        mul_326: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_323, sub_116);  mul_323 = sub_116 = None
        mul_327: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_326, 0.7978845608028654);  mul_326 = None
        convert_element_type_1259: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_327, torch.bfloat16)
        mul_328: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, 0.044715);  mul_327 = None
        pow_30: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_734, 2.0);  convert_element_type_734 = None
        mul_329: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_30, 3.0);  pow_30 = None
        mul_330: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_328, mul_329);  mul_328 = mul_329 = None
        convert_element_type_1260: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_330, torch.bfloat16);  mul_330 = None
        add_253: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1259, convert_element_type_1260);  convert_element_type_1259 = convert_element_type_1260 = None
        mul_331: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1258, 0.5);  convert_element_type_1258 = None
        add_254: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_253, mul_331);  add_253 = mul_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_665: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_254, [4096, 8192]);  add_254 = None
        mm_137: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_665, permute_438);  permute_438 = None
        permute_439: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_665, [1, 0])
        mm_138: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_439, view_414);  permute_439 = view_414 = None
        sum_94: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_665, [0], True, dtype = torch.float32);  view_665 = None
        view_666: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_94, [8192]);  sum_94 = None
        convert_element_type_1265: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_666, torch.bfloat16);  view_666 = None
        view_667: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_137, [32, 128, 2048]);  mm_137 = None
        convert_element_type_1266: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_667, torch.float32);  view_667 = None
        convert_element_type_1267: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_138, torch.float32);  mm_138 = None
        convert_element_type_1268: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1265, torch.float32);  convert_element_type_1265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_333: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1266, primals_264);  primals_264 = None
        mul_334: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_333, 2048)
        sum_95: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_333, [2], True)
        mul_335: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_333, mul_146);  mul_333 = None
        sum_96: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_335, [2], True);  mul_335 = None
        mul_336: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, sum_96);  sum_96 = None
        sub_118: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_334, sum_95);  mul_334 = sum_95 = None
        sub_119: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_118, mul_336);  sub_118 = mul_336 = None
        mul_337: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_37, sub_119);  div_37 = sub_119 = None
        mul_338: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1266, mul_146);  mul_146 = None
        sum_97: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_338, [0, 1]);  mul_338 = None
        sum_98: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1266, [0, 1]);  convert_element_type_1266 = None
        add_255: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_252, mul_337);  add_252 = mul_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1269: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_255, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_668: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1269, [4096, 2048]);  convert_element_type_1269 = None
        mm_139: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_668, permute_442);  permute_442 = None
        permute_443: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_668, [1, 0])
        mm_140: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_443, view_412);  permute_443 = view_412 = None
        sum_99: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_668, [0], True, dtype = torch.float32);  view_668 = None
        view_669: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_99, [2048]);  sum_99 = None
        convert_element_type_1274: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_669, torch.bfloat16);  view_669 = None
        view_670: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_139, [32, 128, 2048]);  mm_139 = None
        convert_element_type_1275: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_140, torch.float32);  mm_140 = None
        convert_element_type_1276: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1274, torch.float32);  convert_element_type_1274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_671: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_670, [32, 128, 16, 128]);  view_670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_446: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_671, [0, 2, 1, 3]);  view_671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_190: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_446, memory_format = torch.contiguous_format);  permute_446 = None
        view_672: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_190, [512, 128, 128]);  clone_190 = None
        bmm_68: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_447, view_672);  permute_447 = None
        bmm_69: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_672, permute_448);  view_672 = permute_448 = None
        view_673: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_68, [32, 16, 128, 128]);  bmm_68 = None
        view_674: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_69, [32, 16, 128, 128]);  bmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1281: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_674, torch.float32);  view_674 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_339: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1281, div_18);  convert_element_type_1281 = None
        sum_100: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_339, [-1], True)
        neg_6: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_18);  div_18 = None
        fma_5: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_6, sum_100, mul_339);  neg_6 = sum_100 = mul_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1282: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_40: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_261, 2, 0, 128);  primals_261 = None
        slice_41: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_40, 3, 0, 128);  slice_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_34: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_41, convert_element_type_1282, full_default_31);  slice_41 = convert_element_type_1282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_675: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_34, [512, 128, 128]);  where_34 = None
        bmm_70: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_449, view_675);  permute_449 = None
        bmm_71: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_675, permute_450);  view_675 = permute_450 = None
        view_676: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_70, [32, 16, 128, 128]);  bmm_70 = None
        view_677: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_71, [32, 16, 128, 128]);  bmm_71 = None
        convert_element_type_1288: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_676, torch.float32);  view_676 = None
        permute_451: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1288, [0, 1, 3, 2]);  convert_element_type_1288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1289: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_451, torch.bfloat16);  permute_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_18: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_677, torch.bfloat16);  view_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_452: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_191: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
        view_678: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_191, [32, 128, 2048]);  clone_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_453: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1289, [0, 2, 1, 3]);  convert_element_type_1289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_679: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_453, [32, 128, 2048]);  permute_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_454: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_18, [0, 2, 1, 3]);  convert_element_type_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_192: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_454, memory_format = torch.contiguous_format);  permute_454 = None
        view_680: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_192, [32, 128, 2048]);  clone_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_681: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_678, [4096, 2048]);  view_678 = None
        permute_455: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_681, [1, 0])
        mm_141: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_455, view_396);  permute_455 = None
        mm_142: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_681, permute_457);  view_681 = permute_457 = None
        view_682: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_142, [32, 128, 2048]);  mm_142 = None
        convert_element_type_1295: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_682, torch.float32);  view_682 = None
        convert_element_type_1296: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_193: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_679, memory_format = torch.contiguous_format);  view_679 = None
        view_683: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_193, [4096, 2048]);  clone_193 = None
        permute_459: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_683, [1, 0])
        mm_143: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_459, view_396);  permute_459 = None
        mm_144: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_683, permute_461);  view_683 = permute_461 = None
        view_684: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_144, [32, 128, 2048]);  mm_144 = None
        convert_element_type_1301: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_684, torch.float32);  view_684 = None
        add_256: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1295, convert_element_type_1301);  convert_element_type_1295 = convert_element_type_1301 = None
        convert_element_type_1302: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_685: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_680, [4096, 2048]);  view_680 = None
        permute_463: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_685, [1, 0])
        mm_145: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_463, view_396);  permute_463 = view_396 = None
        mm_146: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_685, permute_465);  view_685 = permute_465 = None
        view_686: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_146, [32, 128, 2048]);  mm_146 = None
        convert_element_type_1307: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_686, torch.float32);  view_686 = None
        add_257: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_256, convert_element_type_1307);  add_256 = convert_element_type_1307 = None
        convert_element_type_1308: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_341: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_257, primals_256);  primals_256 = None
        mul_342: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_341, 2048)
        sum_101: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
        mul_343: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_341, mul_144);  mul_341 = None
        sum_102: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
        mul_344: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, sum_102);  sum_102 = None
        sub_121: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_342, sum_101);  mul_342 = sum_101 = None
        sub_122: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_121, mul_344);  sub_121 = mul_344 = None
        mul_345: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_38, sub_122);  div_38 = sub_122 = None
        mul_346: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_257, mul_144);  mul_144 = None
        sum_103: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 1]);  mul_346 = None
        sum_104: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_257, [0, 1]);  add_257 = None
        add_258: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_255, mul_345);  add_255 = mul_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1309: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_258, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_687: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1309, [4096, 2048]);  convert_element_type_1309 = None
        mm_147: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_687, permute_467);  permute_467 = None
        permute_468: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_687, [1, 0])
        mm_148: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_468, view_394);  permute_468 = view_394 = None
        sum_105: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_687, [0], True, dtype = torch.float32);  view_687 = None
        view_688: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_105, [2048]);  sum_105 = None
        convert_element_type_1314: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_688, torch.bfloat16);  view_688 = None
        view_689: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_147, [32, 128, 8192]);  mm_147 = None
        convert_element_type_1315: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_689, torch.float32);  view_689 = None
        convert_element_type_1316: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_148, torch.float32);  mm_148 = None
        convert_element_type_1317: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1314, torch.float32);  convert_element_type_1314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_393: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [32, 128, 8192]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_140: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_393, 0.5)
        mul_347: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1315, mul_140);  mul_140 = None
        convert_element_type_695: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_393, torch.float32)
        pow_18: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_695, 3.0)
        mul_141: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_163: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_393, mul_141);  view_393 = mul_141 = None
        mul_142: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_163, 0.7978845608028654);  add_163 = None
        tanh_17: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
        add_164: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_17, 1.0)
        mul_348: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1315, add_164);  convert_element_type_1315 = add_164 = None
        convert_element_type_1318: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_348, torch.bfloat16);  mul_348 = None
        mul_349: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_17, tanh_17);  tanh_17 = None
        sub_123: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_349);  mul_349 = None
        mul_350: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_347, sub_123);  mul_347 = sub_123 = None
        mul_351: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_350, 0.7978845608028654);  mul_350 = None
        convert_element_type_1319: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_351, torch.bfloat16)
        mul_352: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_351, 0.044715);  mul_351 = None
        pow_31: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_695, 2.0);  convert_element_type_695 = None
        mul_353: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_31, 3.0);  pow_31 = None
        mul_354: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_352, mul_353);  mul_352 = mul_353 = None
        convert_element_type_1320: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_354, torch.bfloat16);  mul_354 = None
        add_259: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1319, convert_element_type_1320);  convert_element_type_1319 = convert_element_type_1320 = None
        mul_355: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1318, 0.5);  convert_element_type_1318 = None
        add_260: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_259, mul_355);  add_259 = mul_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_690: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_260, [4096, 8192]);  add_260 = None
        mm_149: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_690, permute_471);  permute_471 = None
        permute_472: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_690, [1, 0])
        mm_150: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_472, view_392);  permute_472 = view_392 = None
        sum_106: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_690, [0], True, dtype = torch.float32);  view_690 = None
        view_691: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_106, [8192]);  sum_106 = None
        convert_element_type_1325: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_691, torch.bfloat16);  view_691 = None
        view_692: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_149, [32, 128, 2048]);  mm_149 = None
        convert_element_type_1326: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_692, torch.float32);  view_692 = None
        convert_element_type_1327: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_150, torch.float32);  mm_150 = None
        convert_element_type_1328: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1325, torch.float32);  convert_element_type_1325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_357: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1326, primals_250);  primals_250 = None
        mul_358: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_357, 2048)
        sum_107: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True)
        mul_359: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_357, mul_138);  mul_357 = None
        sum_108: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_359, [2], True);  mul_359 = None
        mul_360: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, sum_108);  sum_108 = None
        sub_125: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_358, sum_107);  mul_358 = sum_107 = None
        sub_126: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_125, mul_360);  sub_125 = mul_360 = None
        mul_361: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_39, sub_126);  div_39 = sub_126 = None
        mul_362: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1326, mul_138);  mul_138 = None
        sum_109: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 1]);  mul_362 = None
        sum_110: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1326, [0, 1]);  convert_element_type_1326 = None
        add_261: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_258, mul_361);  add_258 = mul_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1329: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_261, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_693: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1329, [4096, 2048]);  convert_element_type_1329 = None
        mm_151: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_693, permute_475);  permute_475 = None
        permute_476: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_693, [1, 0])
        mm_152: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_476, view_390);  permute_476 = view_390 = None
        sum_111: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_693, [0], True, dtype = torch.float32);  view_693 = None
        view_694: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_111, [2048]);  sum_111 = None
        convert_element_type_1334: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_694, torch.bfloat16);  view_694 = None
        view_695: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_151, [32, 128, 2048]);  mm_151 = None
        convert_element_type_1335: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_152, torch.float32);  mm_152 = None
        convert_element_type_1336: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1334, torch.float32);  convert_element_type_1334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_696: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_695, [32, 128, 16, 128]);  view_695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_479: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_696, [0, 2, 1, 3]);  view_696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_194: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
        view_697: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_194, [512, 128, 128]);  clone_194 = None
        bmm_72: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_480, view_697);  permute_480 = None
        bmm_73: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_697, permute_481);  view_697 = permute_481 = None
        view_698: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_72, [32, 16, 128, 128]);  bmm_72 = None
        view_699: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_73, [32, 16, 128, 128]);  bmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1341: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_699, torch.float32);  view_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_363: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1341, div_17);  convert_element_type_1341 = None
        sum_112: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_363, [-1], True)
        neg_7: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_17);  div_17 = None
        fma_6: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_7, sum_112, mul_363);  neg_7 = sum_112 = mul_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1342: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_38: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_247, 2, 0, 128);  primals_247 = None
        slice_39: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_38, 3, 0, 128);  slice_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_35: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_39, convert_element_type_1342, full_default_31);  slice_39 = convert_element_type_1342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_700: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_35, [512, 128, 128]);  where_35 = None
        bmm_74: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_482, view_700);  permute_482 = None
        bmm_75: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_700, permute_483);  view_700 = permute_483 = None
        view_701: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_74, [32, 16, 128, 128]);  bmm_74 = None
        view_702: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_75, [32, 16, 128, 128]);  bmm_75 = None
        convert_element_type_1348: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_701, torch.float32);  view_701 = None
        permute_484: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1348, [0, 1, 3, 2]);  convert_element_type_1348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1349: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_484, torch.bfloat16);  permute_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_17: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_702, torch.bfloat16);  view_702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_485: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_698, [0, 2, 1, 3]);  view_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_195: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
        view_703: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_195, [32, 128, 2048]);  clone_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_486: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1349, [0, 2, 1, 3]);  convert_element_type_1349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_704: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_486, [32, 128, 2048]);  permute_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_487: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_17, [0, 2, 1, 3]);  convert_element_type_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_196: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_487, memory_format = torch.contiguous_format);  permute_487 = None
        view_705: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_196, [32, 128, 2048]);  clone_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_706: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_703, [4096, 2048]);  view_703 = None
        permute_488: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_706, [1, 0])
        mm_153: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_488, view_374);  permute_488 = None
        mm_154: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_706, permute_490);  view_706 = permute_490 = None
        view_707: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_154, [32, 128, 2048]);  mm_154 = None
        convert_element_type_1355: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_707, torch.float32);  view_707 = None
        convert_element_type_1356: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_153, torch.float32);  mm_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_197: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_704, memory_format = torch.contiguous_format);  view_704 = None
        view_708: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_197, [4096, 2048]);  clone_197 = None
        permute_492: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_708, [1, 0])
        mm_155: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_492, view_374);  permute_492 = None
        mm_156: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_708, permute_494);  view_708 = permute_494 = None
        view_709: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_156, [32, 128, 2048]);  mm_156 = None
        convert_element_type_1361: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_709, torch.float32);  view_709 = None
        add_262: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1355, convert_element_type_1361);  convert_element_type_1355 = convert_element_type_1361 = None
        convert_element_type_1362: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_155, torch.float32);  mm_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_710: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_705, [4096, 2048]);  view_705 = None
        permute_496: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_710, [1, 0])
        mm_157: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_496, view_374);  permute_496 = view_374 = None
        mm_158: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_710, permute_498);  view_710 = permute_498 = None
        view_711: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_158, [32, 128, 2048]);  mm_158 = None
        convert_element_type_1367: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_711, torch.float32);  view_711 = None
        add_263: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_262, convert_element_type_1367);  add_262 = convert_element_type_1367 = None
        convert_element_type_1368: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_157, torch.float32);  mm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_365: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_263, primals_242);  primals_242 = None
        mul_366: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_365, 2048)
        sum_113: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_365, [2], True)
        mul_367: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_365, mul_136);  mul_365 = None
        sum_114: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True);  mul_367 = None
        mul_368: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, sum_114);  sum_114 = None
        sub_128: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_366, sum_113);  mul_366 = sum_113 = None
        sub_129: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_128, mul_368);  sub_128 = mul_368 = None
        mul_369: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_40, sub_129);  div_40 = sub_129 = None
        mul_370: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_263, mul_136);  mul_136 = None
        sum_115: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_370, [0, 1]);  mul_370 = None
        sum_116: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None
        add_264: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_261, mul_369);  add_261 = mul_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1369: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_264, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_712: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1369, [4096, 2048]);  convert_element_type_1369 = None
        mm_159: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_712, permute_500);  permute_500 = None
        permute_501: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_712, [1, 0])
        mm_160: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_501, view_372);  permute_501 = view_372 = None
        sum_117: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_712, [0], True, dtype = torch.float32);  view_712 = None
        view_713: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_117, [2048]);  sum_117 = None
        convert_element_type_1374: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_713, torch.bfloat16);  view_713 = None
        view_714: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_159, [32, 128, 8192]);  mm_159 = None
        convert_element_type_1375: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_714, torch.float32);  view_714 = None
        convert_element_type_1376: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_160, torch.float32);  mm_160 = None
        convert_element_type_1377: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1374, torch.float32);  convert_element_type_1374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_371: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [32, 128, 8192]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_132: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_371, 0.5)
        mul_371: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1375, mul_132);  mul_132 = None
        convert_element_type_656: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_371, torch.float32)
        pow_17: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_656, 3.0)
        mul_133: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_154: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_371, mul_133);  view_371 = mul_133 = None
        mul_134: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_154, 0.7978845608028654);  add_154 = None
        tanh_16: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_134);  mul_134 = None
        add_155: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_16, 1.0)
        mul_372: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1375, add_155);  convert_element_type_1375 = add_155 = None
        convert_element_type_1378: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_372, torch.bfloat16);  mul_372 = None
        mul_373: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_16, tanh_16);  tanh_16 = None
        sub_130: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_373);  mul_373 = None
        mul_374: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_371, sub_130);  mul_371 = sub_130 = None
        mul_375: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_374, 0.7978845608028654);  mul_374 = None
        convert_element_type_1379: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_375, torch.bfloat16)
        mul_376: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_375, 0.044715);  mul_375 = None
        pow_32: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_656, 2.0);  convert_element_type_656 = None
        mul_377: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_32, 3.0);  pow_32 = None
        mul_378: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_376, mul_377);  mul_376 = mul_377 = None
        convert_element_type_1380: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_378, torch.bfloat16);  mul_378 = None
        add_265: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1379, convert_element_type_1380);  convert_element_type_1379 = convert_element_type_1380 = None
        mul_379: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1378, 0.5);  convert_element_type_1378 = None
        add_266: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_265, mul_379);  add_265 = mul_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_715: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_266, [4096, 8192]);  add_266 = None
        mm_161: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_715, permute_504);  permute_504 = None
        permute_505: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_715, [1, 0])
        mm_162: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_505, view_370);  permute_505 = view_370 = None
        sum_118: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_715, [0], True, dtype = torch.float32);  view_715 = None
        view_716: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_118, [8192]);  sum_118 = None
        convert_element_type_1385: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_716, torch.bfloat16);  view_716 = None
        view_717: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_161, [32, 128, 2048]);  mm_161 = None
        convert_element_type_1386: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_717, torch.float32);  view_717 = None
        convert_element_type_1387: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_162, torch.float32);  mm_162 = None
        convert_element_type_1388: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1385, torch.float32);  convert_element_type_1385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_381: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1386, primals_236);  primals_236 = None
        mul_382: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, 2048)
        sum_119: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_381, [2], True)
        mul_383: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, mul_130);  mul_381 = None
        sum_120: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True);  mul_383 = None
        mul_384: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, sum_120);  sum_120 = None
        sub_132: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_382, sum_119);  mul_382 = sum_119 = None
        sub_133: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_132, mul_384);  sub_132 = mul_384 = None
        mul_385: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_41, sub_133);  div_41 = sub_133 = None
        mul_386: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1386, mul_130);  mul_130 = None
        sum_121: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 1]);  mul_386 = None
        sum_122: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1386, [0, 1]);  convert_element_type_1386 = None
        add_267: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_264, mul_385);  add_264 = mul_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1389: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_267, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_718: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1389, [4096, 2048]);  convert_element_type_1389 = None
        mm_163: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_718, permute_508);  permute_508 = None
        permute_509: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_718, [1, 0])
        mm_164: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_509, view_368);  permute_509 = view_368 = None
        sum_123: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_718, [0], True, dtype = torch.float32);  view_718 = None
        view_719: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_123, [2048]);  sum_123 = None
        convert_element_type_1394: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_719, torch.bfloat16);  view_719 = None
        view_720: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_163, [32, 128, 2048]);  mm_163 = None
        convert_element_type_1395: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_164, torch.float32);  mm_164 = None
        convert_element_type_1396: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1394, torch.float32);  convert_element_type_1394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_721: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_720, [32, 128, 16, 128]);  view_720 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_512: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_721, [0, 2, 1, 3]);  view_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_198: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_512, memory_format = torch.contiguous_format);  permute_512 = None
        view_722: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_198, [512, 128, 128]);  clone_198 = None
        bmm_76: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_513, view_722);  permute_513 = None
        bmm_77: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_722, permute_514);  view_722 = permute_514 = None
        view_723: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_76, [32, 16, 128, 128]);  bmm_76 = None
        view_724: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_77, [32, 16, 128, 128]);  bmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1401: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_724, torch.float32);  view_724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_387: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1401, div_16);  convert_element_type_1401 = None
        sum_124: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_387, [-1], True)
        neg_8: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_16);  div_16 = None
        fma_7: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_8, sum_124, mul_387);  neg_8 = sum_124 = mul_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1402: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_36: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_233, 2, 0, 128);  primals_233 = None
        slice_37: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_36, 3, 0, 128);  slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_36: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_37, convert_element_type_1402, full_default_31);  slice_37 = convert_element_type_1402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_725: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_36, [512, 128, 128]);  where_36 = None
        bmm_78: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_515, view_725);  permute_515 = None
        bmm_79: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_725, permute_516);  view_725 = permute_516 = None
        view_726: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_78, [32, 16, 128, 128]);  bmm_78 = None
        view_727: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_79, [32, 16, 128, 128]);  bmm_79 = None
        convert_element_type_1408: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_726, torch.float32);  view_726 = None
        permute_517: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1408, [0, 1, 3, 2]);  convert_element_type_1408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1409: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_517, torch.bfloat16);  permute_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_16: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_727, torch.bfloat16);  view_727 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_518: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_723, [0, 2, 1, 3]);  view_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_199: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
        view_728: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_199, [32, 128, 2048]);  clone_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_519: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1409, [0, 2, 1, 3]);  convert_element_type_1409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_729: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_519, [32, 128, 2048]);  permute_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_520: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_16, [0, 2, 1, 3]);  convert_element_type_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_200: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_520, memory_format = torch.contiguous_format);  permute_520 = None
        view_730: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_200, [32, 128, 2048]);  clone_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_731: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_728, [4096, 2048]);  view_728 = None
        permute_521: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_731, [1, 0])
        mm_165: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_521, view_352);  permute_521 = None
        mm_166: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_731, permute_523);  view_731 = permute_523 = None
        view_732: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_166, [32, 128, 2048]);  mm_166 = None
        convert_element_type_1415: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_732, torch.float32);  view_732 = None
        convert_element_type_1416: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_165, torch.float32);  mm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_201: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_729, memory_format = torch.contiguous_format);  view_729 = None
        view_733: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_201, [4096, 2048]);  clone_201 = None
        permute_525: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_733, [1, 0])
        mm_167: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_525, view_352);  permute_525 = None
        mm_168: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_733, permute_527);  view_733 = permute_527 = None
        view_734: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_168, [32, 128, 2048]);  mm_168 = None
        convert_element_type_1421: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_734, torch.float32);  view_734 = None
        add_268: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1415, convert_element_type_1421);  convert_element_type_1415 = convert_element_type_1421 = None
        convert_element_type_1422: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_167, torch.float32);  mm_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_735: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_730, [4096, 2048]);  view_730 = None
        permute_529: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_735, [1, 0])
        mm_169: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_529, view_352);  permute_529 = view_352 = None
        mm_170: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_735, permute_531);  view_735 = permute_531 = None
        view_736: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_170, [32, 128, 2048]);  mm_170 = None
        convert_element_type_1427: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_736, torch.float32);  view_736 = None
        add_269: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_268, convert_element_type_1427);  add_268 = convert_element_type_1427 = None
        convert_element_type_1428: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_169, torch.float32);  mm_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_389: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_269, primals_228);  primals_228 = None
        mul_390: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_389, 2048)
        sum_125: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_389, [2], True)
        mul_391: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_389, mul_128);  mul_389 = None
        sum_126: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_391, [2], True);  mul_391 = None
        mul_392: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, sum_126);  sum_126 = None
        sub_135: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_390, sum_125);  mul_390 = sum_125 = None
        sub_136: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_135, mul_392);  sub_135 = mul_392 = None
        mul_393: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_42, sub_136);  div_42 = sub_136 = None
        mul_394: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_269, mul_128);  mul_128 = None
        sum_127: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_394, [0, 1]);  mul_394 = None
        sum_128: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_269, [0, 1]);  add_269 = None
        add_270: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_267, mul_393);  add_267 = mul_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1429: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_270, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_737: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1429, [4096, 2048]);  convert_element_type_1429 = None
        mm_171: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_737, permute_533);  permute_533 = None
        permute_534: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_737, [1, 0])
        mm_172: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_534, view_350);  permute_534 = view_350 = None
        sum_129: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_737, [0], True, dtype = torch.float32);  view_737 = None
        view_738: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_129, [2048]);  sum_129 = None
        convert_element_type_1434: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_738, torch.bfloat16);  view_738 = None
        view_739: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_171, [32, 128, 8192]);  mm_171 = None
        convert_element_type_1435: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_739, torch.float32);  view_739 = None
        convert_element_type_1436: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_172, torch.float32);  mm_172 = None
        convert_element_type_1437: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1434, torch.float32);  convert_element_type_1434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_349: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [32, 128, 8192]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_124: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_349, 0.5)
        mul_395: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1435, mul_124);  mul_124 = None
        convert_element_type_617: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_349, torch.float32)
        pow_16: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_617, 3.0)
        mul_125: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_145: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_349, mul_125);  view_349 = mul_125 = None
        mul_126: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_145, 0.7978845608028654);  add_145 = None
        tanh_15: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_126);  mul_126 = None
        add_146: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
        mul_396: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1435, add_146);  convert_element_type_1435 = add_146 = None
        convert_element_type_1438: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_396, torch.bfloat16);  mul_396 = None
        mul_397: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_15, tanh_15);  tanh_15 = None
        sub_137: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_397);  mul_397 = None
        mul_398: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_395, sub_137);  mul_395 = sub_137 = None
        mul_399: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_398, 0.7978845608028654);  mul_398 = None
        convert_element_type_1439: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_399, torch.bfloat16)
        mul_400: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_399, 0.044715);  mul_399 = None
        pow_33: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_617, 2.0);  convert_element_type_617 = None
        mul_401: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_33, 3.0);  pow_33 = None
        mul_402: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
        convert_element_type_1440: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_402, torch.bfloat16);  mul_402 = None
        add_271: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1439, convert_element_type_1440);  convert_element_type_1439 = convert_element_type_1440 = None
        mul_403: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1438, 0.5);  convert_element_type_1438 = None
        add_272: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_271, mul_403);  add_271 = mul_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_740: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_272, [4096, 8192]);  add_272 = None
        mm_173: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_740, permute_537);  permute_537 = None
        permute_538: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_740, [1, 0])
        mm_174: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_538, view_348);  permute_538 = view_348 = None
        sum_130: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_740, [0], True, dtype = torch.float32);  view_740 = None
        view_741: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_130, [8192]);  sum_130 = None
        convert_element_type_1445: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_741, torch.bfloat16);  view_741 = None
        view_742: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_173, [32, 128, 2048]);  mm_173 = None
        convert_element_type_1446: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_742, torch.float32);  view_742 = None
        convert_element_type_1447: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_174, torch.float32);  mm_174 = None
        convert_element_type_1448: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1445, torch.float32);  convert_element_type_1445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_405: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1446, primals_222);  primals_222 = None
        mul_406: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_405, 2048)
        sum_131: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_405, [2], True)
        mul_407: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_405, mul_122);  mul_405 = None
        sum_132: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_407, [2], True);  mul_407 = None
        mul_408: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, sum_132);  sum_132 = None
        sub_139: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_406, sum_131);  mul_406 = sum_131 = None
        sub_140: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_139, mul_408);  sub_139 = mul_408 = None
        mul_409: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_43, sub_140);  div_43 = sub_140 = None
        mul_410: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1446, mul_122);  mul_122 = None
        sum_133: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_410, [0, 1]);  mul_410 = None
        sum_134: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1446, [0, 1]);  convert_element_type_1446 = None
        add_273: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_270, mul_409);  add_270 = mul_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1449: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_273, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_743: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1449, [4096, 2048]);  convert_element_type_1449 = None
        mm_175: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_743, permute_541);  permute_541 = None
        permute_542: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_743, [1, 0])
        mm_176: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_542, view_346);  permute_542 = view_346 = None
        sum_135: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_743, [0], True, dtype = torch.float32);  view_743 = None
        view_744: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_135, [2048]);  sum_135 = None
        convert_element_type_1454: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_744, torch.bfloat16);  view_744 = None
        view_745: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_175, [32, 128, 2048]);  mm_175 = None
        convert_element_type_1455: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_176, torch.float32);  mm_176 = None
        convert_element_type_1456: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1454, torch.float32);  convert_element_type_1454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_746: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_745, [32, 128, 16, 128]);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_545: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_746, [0, 2, 1, 3]);  view_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_202: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_545, memory_format = torch.contiguous_format);  permute_545 = None
        view_747: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_202, [512, 128, 128]);  clone_202 = None
        bmm_80: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_546, view_747);  permute_546 = None
        bmm_81: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_747, permute_547);  view_747 = permute_547 = None
        view_748: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_80, [32, 16, 128, 128]);  bmm_80 = None
        view_749: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_81, [32, 16, 128, 128]);  bmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1461: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_749, torch.float32);  view_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_411: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1461, div_15);  convert_element_type_1461 = None
        sum_136: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_411, [-1], True)
        neg_9: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_15);  div_15 = None
        fma_8: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_9, sum_136, mul_411);  neg_9 = sum_136 = mul_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1462: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_34: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_219, 2, 0, 128);  primals_219 = None
        slice_35: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_34, 3, 0, 128);  slice_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_37: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_35, convert_element_type_1462, full_default_31);  slice_35 = convert_element_type_1462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_750: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_37, [512, 128, 128]);  where_37 = None
        bmm_82: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_548, view_750);  permute_548 = None
        bmm_83: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_750, permute_549);  view_750 = permute_549 = None
        view_751: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_82, [32, 16, 128, 128]);  bmm_82 = None
        view_752: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_83, [32, 16, 128, 128]);  bmm_83 = None
        convert_element_type_1468: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_751, torch.float32);  view_751 = None
        permute_550: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1468, [0, 1, 3, 2]);  convert_element_type_1468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1469: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_550, torch.bfloat16);  permute_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_15: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_752, torch.bfloat16);  view_752 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_551: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_748, [0, 2, 1, 3]);  view_748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_203: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
        view_753: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_203, [32, 128, 2048]);  clone_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_552: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1469, [0, 2, 1, 3]);  convert_element_type_1469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_754: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_552, [32, 128, 2048]);  permute_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_553: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_15, [0, 2, 1, 3]);  convert_element_type_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_204: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_553, memory_format = torch.contiguous_format);  permute_553 = None
        view_755: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_204, [32, 128, 2048]);  clone_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_756: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_753, [4096, 2048]);  view_753 = None
        permute_554: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_756, [1, 0])
        mm_177: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_554, view_330);  permute_554 = None
        mm_178: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_756, permute_556);  view_756 = permute_556 = None
        view_757: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_178, [32, 128, 2048]);  mm_178 = None
        convert_element_type_1475: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_757, torch.float32);  view_757 = None
        convert_element_type_1476: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_177, torch.float32);  mm_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_205: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_754, memory_format = torch.contiguous_format);  view_754 = None
        view_758: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_205, [4096, 2048]);  clone_205 = None
        permute_558: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_758, [1, 0])
        mm_179: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_558, view_330);  permute_558 = None
        mm_180: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_758, permute_560);  view_758 = permute_560 = None
        view_759: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_180, [32, 128, 2048]);  mm_180 = None
        convert_element_type_1481: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_759, torch.float32);  view_759 = None
        add_274: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1475, convert_element_type_1481);  convert_element_type_1475 = convert_element_type_1481 = None
        convert_element_type_1482: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_179, torch.float32);  mm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_760: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_755, [4096, 2048]);  view_755 = None
        permute_562: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_760, [1, 0])
        mm_181: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_562, view_330);  permute_562 = view_330 = None
        mm_182: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_760, permute_564);  view_760 = permute_564 = None
        view_761: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_182, [32, 128, 2048]);  mm_182 = None
        convert_element_type_1487: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_761, torch.float32);  view_761 = None
        add_275: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_274, convert_element_type_1487);  add_274 = convert_element_type_1487 = None
        convert_element_type_1488: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_181, torch.float32);  mm_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_413: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_275, primals_214);  primals_214 = None
        mul_414: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_413, 2048)
        sum_137: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True)
        mul_415: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_413, mul_120);  mul_413 = None
        sum_138: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_415, [2], True);  mul_415 = None
        mul_416: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, sum_138);  sum_138 = None
        sub_142: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_414, sum_137);  mul_414 = sum_137 = None
        sub_143: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_142, mul_416);  sub_142 = mul_416 = None
        mul_417: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_44, sub_143);  div_44 = sub_143 = None
        mul_418: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_275, mul_120);  mul_120 = None
        sum_139: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 1]);  mul_418 = None
        sum_140: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_275, [0, 1]);  add_275 = None
        add_276: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_273, mul_417);  add_273 = mul_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1489: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_276, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_762: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1489, [4096, 2048]);  convert_element_type_1489 = None
        mm_183: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_762, permute_566);  permute_566 = None
        permute_567: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_762, [1, 0])
        mm_184: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_567, view_328);  permute_567 = view_328 = None
        sum_141: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_762, [0], True, dtype = torch.float32);  view_762 = None
        view_763: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_141, [2048]);  sum_141 = None
        convert_element_type_1494: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_763, torch.bfloat16);  view_763 = None
        view_764: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_183, [32, 128, 8192]);  mm_183 = None
        convert_element_type_1495: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_764, torch.float32);  view_764 = None
        convert_element_type_1496: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_184, torch.float32);  mm_184 = None
        convert_element_type_1497: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1494, torch.float32);  convert_element_type_1494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_327: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [32, 128, 8192]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_327, 0.5)
        mul_419: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1495, mul_116);  mul_116 = None
        convert_element_type_578: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_327, torch.float32)
        pow_15: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_578, 3.0)
        mul_117: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_136: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_327, mul_117);  view_327 = mul_117 = None
        mul_118: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_136, 0.7978845608028654);  add_136 = None
        tanh_14: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_137: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
        mul_420: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1495, add_137);  convert_element_type_1495 = add_137 = None
        convert_element_type_1498: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_420, torch.bfloat16);  mul_420 = None
        mul_421: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_14, tanh_14);  tanh_14 = None
        sub_144: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_421);  mul_421 = None
        mul_422: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_419, sub_144);  mul_419 = sub_144 = None
        mul_423: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_422, 0.7978845608028654);  mul_422 = None
        convert_element_type_1499: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_423, torch.bfloat16)
        mul_424: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_423, 0.044715);  mul_423 = None
        pow_34: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_578, 2.0);  convert_element_type_578 = None
        mul_425: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_34, 3.0);  pow_34 = None
        mul_426: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
        convert_element_type_1500: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_426, torch.bfloat16);  mul_426 = None
        add_277: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1499, convert_element_type_1500);  convert_element_type_1499 = convert_element_type_1500 = None
        mul_427: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1498, 0.5);  convert_element_type_1498 = None
        add_278: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_277, mul_427);  add_277 = mul_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_765: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_278, [4096, 8192]);  add_278 = None
        mm_185: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_765, permute_570);  permute_570 = None
        permute_571: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_765, [1, 0])
        mm_186: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_571, view_326);  permute_571 = view_326 = None
        sum_142: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_765, [0], True, dtype = torch.float32);  view_765 = None
        view_766: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_142, [8192]);  sum_142 = None
        convert_element_type_1505: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_766, torch.bfloat16);  view_766 = None
        view_767: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_185, [32, 128, 2048]);  mm_185 = None
        convert_element_type_1506: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_767, torch.float32);  view_767 = None
        convert_element_type_1507: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_186, torch.float32);  mm_186 = None
        convert_element_type_1508: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1505, torch.float32);  convert_element_type_1505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_429: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1506, primals_208);  primals_208 = None
        mul_430: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_429, 2048)
        sum_143: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True)
        mul_431: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_429, mul_114);  mul_429 = None
        sum_144: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_431, [2], True);  mul_431 = None
        mul_432: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_114, sum_144);  sum_144 = None
        sub_146: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_430, sum_143);  mul_430 = sum_143 = None
        sub_147: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_146, mul_432);  sub_146 = mul_432 = None
        mul_433: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_45, sub_147);  div_45 = sub_147 = None
        mul_434: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1506, mul_114);  mul_114 = None
        sum_145: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_434, [0, 1]);  mul_434 = None
        sum_146: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1506, [0, 1]);  convert_element_type_1506 = None
        add_279: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_276, mul_433);  add_276 = mul_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1509: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_279, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_768: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1509, [4096, 2048]);  convert_element_type_1509 = None
        mm_187: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_768, permute_574);  permute_574 = None
        permute_575: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_768, [1, 0])
        mm_188: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_575, view_324);  permute_575 = view_324 = None
        sum_147: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_768, [0], True, dtype = torch.float32);  view_768 = None
        view_769: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [2048]);  sum_147 = None
        convert_element_type_1514: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_769, torch.bfloat16);  view_769 = None
        view_770: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_187, [32, 128, 2048]);  mm_187 = None
        convert_element_type_1515: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_188, torch.float32);  mm_188 = None
        convert_element_type_1516: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1514, torch.float32);  convert_element_type_1514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_771: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_770, [32, 128, 16, 128]);  view_770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_578: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_771, [0, 2, 1, 3]);  view_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_206: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_578, memory_format = torch.contiguous_format);  permute_578 = None
        view_772: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_206, [512, 128, 128]);  clone_206 = None
        bmm_84: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_579, view_772);  permute_579 = None
        bmm_85: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_772, permute_580);  view_772 = permute_580 = None
        view_773: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_84, [32, 16, 128, 128]);  bmm_84 = None
        view_774: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_85, [32, 16, 128, 128]);  bmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1521: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_774, torch.float32);  view_774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_435: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1521, div_14);  convert_element_type_1521 = None
        sum_148: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_435, [-1], True)
        neg_10: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_14);  div_14 = None
        fma_9: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_10, sum_148, mul_435);  neg_10 = sum_148 = mul_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1522: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_32: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_205, 2, 0, 128);  primals_205 = None
        slice_33: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_32, 3, 0, 128);  slice_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_38: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_33, convert_element_type_1522, full_default_31);  slice_33 = convert_element_type_1522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_775: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_38, [512, 128, 128]);  where_38 = None
        bmm_86: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_581, view_775);  permute_581 = None
        bmm_87: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_775, permute_582);  view_775 = permute_582 = None
        view_776: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_86, [32, 16, 128, 128]);  bmm_86 = None
        view_777: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_87, [32, 16, 128, 128]);  bmm_87 = None
        convert_element_type_1528: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_776, torch.float32);  view_776 = None
        permute_583: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1528, [0, 1, 3, 2]);  convert_element_type_1528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1529: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_583, torch.bfloat16);  permute_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_14: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_777, torch.bfloat16);  view_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_584: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_773, [0, 2, 1, 3]);  view_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_207: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_584, memory_format = torch.contiguous_format);  permute_584 = None
        view_778: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_207, [32, 128, 2048]);  clone_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_585: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1529, [0, 2, 1, 3]);  convert_element_type_1529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_779: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_585, [32, 128, 2048]);  permute_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_586: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_14, [0, 2, 1, 3]);  convert_element_type_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_208: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_586, memory_format = torch.contiguous_format);  permute_586 = None
        view_780: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_208, [32, 128, 2048]);  clone_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_781: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_778, [4096, 2048]);  view_778 = None
        permute_587: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_781, [1, 0])
        mm_189: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_587, view_308);  permute_587 = None
        mm_190: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_781, permute_589);  view_781 = permute_589 = None
        view_782: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_190, [32, 128, 2048]);  mm_190 = None
        convert_element_type_1535: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_782, torch.float32);  view_782 = None
        convert_element_type_1536: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_189, torch.float32);  mm_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_209: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_779, memory_format = torch.contiguous_format);  view_779 = None
        view_783: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_209, [4096, 2048]);  clone_209 = None
        permute_591: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_783, [1, 0])
        mm_191: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_591, view_308);  permute_591 = None
        mm_192: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_783, permute_593);  view_783 = permute_593 = None
        view_784: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_192, [32, 128, 2048]);  mm_192 = None
        convert_element_type_1541: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_784, torch.float32);  view_784 = None
        add_280: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1535, convert_element_type_1541);  convert_element_type_1535 = convert_element_type_1541 = None
        convert_element_type_1542: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_191, torch.float32);  mm_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_785: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_780, [4096, 2048]);  view_780 = None
        permute_595: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_785, [1, 0])
        mm_193: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_595, view_308);  permute_595 = view_308 = None
        mm_194: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_785, permute_597);  view_785 = permute_597 = None
        view_786: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_194, [32, 128, 2048]);  mm_194 = None
        convert_element_type_1547: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_786, torch.float32);  view_786 = None
        add_281: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_280, convert_element_type_1547);  add_280 = convert_element_type_1547 = None
        convert_element_type_1548: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_193, torch.float32);  mm_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_437: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_281, primals_200);  primals_200 = None
        mul_438: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_437, 2048)
        sum_149: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_437, [2], True)
        mul_439: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_437, mul_112);  mul_437 = None
        sum_150: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_439, [2], True);  mul_439 = None
        mul_440: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, sum_150);  sum_150 = None
        sub_149: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_438, sum_149);  mul_438 = sum_149 = None
        sub_150: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_149, mul_440);  sub_149 = mul_440 = None
        mul_441: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_46, sub_150);  div_46 = sub_150 = None
        mul_442: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_281, mul_112);  mul_112 = None
        sum_151: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_442, [0, 1]);  mul_442 = None
        sum_152: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_281, [0, 1]);  add_281 = None
        add_282: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_279, mul_441);  add_279 = mul_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1549: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_282, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_787: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1549, [4096, 2048]);  convert_element_type_1549 = None
        mm_195: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_787, permute_599);  permute_599 = None
        permute_600: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_787, [1, 0])
        mm_196: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_600, view_306);  permute_600 = view_306 = None
        sum_153: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_787, [0], True, dtype = torch.float32);  view_787 = None
        view_788: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_153, [2048]);  sum_153 = None
        convert_element_type_1554: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_788, torch.bfloat16);  view_788 = None
        view_789: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_195, [32, 128, 8192]);  mm_195 = None
        convert_element_type_1555: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_789, torch.float32);  view_789 = None
        convert_element_type_1556: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_196, torch.float32);  mm_196 = None
        convert_element_type_1557: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1554, torch.float32);  convert_element_type_1554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_305: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [32, 128, 8192]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_108: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_305, 0.5)
        mul_443: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1555, mul_108);  mul_108 = None
        convert_element_type_539: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_305, torch.float32)
        pow_14: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_539, 3.0)
        mul_109: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_127: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_305, mul_109);  view_305 = mul_109 = None
        mul_110: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_127, 0.7978845608028654);  add_127 = None
        tanh_13: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_110);  mul_110 = None
        add_128: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
        mul_444: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1555, add_128);  convert_element_type_1555 = add_128 = None
        convert_element_type_1558: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_444, torch.bfloat16);  mul_444 = None
        mul_445: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
        sub_151: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_445);  mul_445 = None
        mul_446: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_443, sub_151);  mul_443 = sub_151 = None
        mul_447: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_446, 0.7978845608028654);  mul_446 = None
        convert_element_type_1559: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_447, torch.bfloat16)
        mul_448: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_447, 0.044715);  mul_447 = None
        pow_35: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_539, 2.0);  convert_element_type_539 = None
        mul_449: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_35, 3.0);  pow_35 = None
        mul_450: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_448, mul_449);  mul_448 = mul_449 = None
        convert_element_type_1560: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_450, torch.bfloat16);  mul_450 = None
        add_283: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1559, convert_element_type_1560);  convert_element_type_1559 = convert_element_type_1560 = None
        mul_451: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1558, 0.5);  convert_element_type_1558 = None
        add_284: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_283, mul_451);  add_283 = mul_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_790: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_284, [4096, 8192]);  add_284 = None
        mm_197: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_790, permute_603);  permute_603 = None
        permute_604: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_790, [1, 0])
        mm_198: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_604, view_304);  permute_604 = view_304 = None
        sum_154: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_790, [0], True, dtype = torch.float32);  view_790 = None
        view_791: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_154, [8192]);  sum_154 = None
        convert_element_type_1565: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_791, torch.bfloat16);  view_791 = None
        view_792: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_197, [32, 128, 2048]);  mm_197 = None
        convert_element_type_1566: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_792, torch.float32);  view_792 = None
        convert_element_type_1567: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_198, torch.float32);  mm_198 = None
        convert_element_type_1568: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1565, torch.float32);  convert_element_type_1565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_453: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1566, primals_194);  primals_194 = None
        mul_454: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_453, 2048)
        sum_155: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True)
        mul_455: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_453, mul_106);  mul_453 = None
        sum_156: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_455, [2], True);  mul_455 = None
        mul_456: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_106, sum_156);  sum_156 = None
        sub_153: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_454, sum_155);  mul_454 = sum_155 = None
        sub_154: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_153, mul_456);  sub_153 = mul_456 = None
        mul_457: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_47, sub_154);  div_47 = sub_154 = None
        mul_458: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1566, mul_106);  mul_106 = None
        sum_157: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 1]);  mul_458 = None
        sum_158: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1566, [0, 1]);  convert_element_type_1566 = None
        add_285: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_282, mul_457);  add_282 = mul_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1569: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_285, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_793: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1569, [4096, 2048]);  convert_element_type_1569 = None
        mm_199: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_793, permute_607);  permute_607 = None
        permute_608: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_793, [1, 0])
        mm_200: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_608, view_302);  permute_608 = view_302 = None
        sum_159: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_793, [0], True, dtype = torch.float32);  view_793 = None
        view_794: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_159, [2048]);  sum_159 = None
        convert_element_type_1574: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_794, torch.bfloat16);  view_794 = None
        view_795: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_199, [32, 128, 2048]);  mm_199 = None
        convert_element_type_1575: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_200, torch.float32);  mm_200 = None
        convert_element_type_1576: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1574, torch.float32);  convert_element_type_1574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_796: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_795, [32, 128, 16, 128]);  view_795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_611: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_796, [0, 2, 1, 3]);  view_796 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_210: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_611, memory_format = torch.contiguous_format);  permute_611 = None
        view_797: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_210, [512, 128, 128]);  clone_210 = None
        bmm_88: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_612, view_797);  permute_612 = None
        bmm_89: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_797, permute_613);  view_797 = permute_613 = None
        view_798: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_88, [32, 16, 128, 128]);  bmm_88 = None
        view_799: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_89, [32, 16, 128, 128]);  bmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1581: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_799, torch.float32);  view_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_459: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1581, div_13);  convert_element_type_1581 = None
        sum_160: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_459, [-1], True)
        neg_11: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_10: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_11, sum_160, mul_459);  neg_11 = sum_160 = mul_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1582: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_30: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_191, 2, 0, 128);  primals_191 = None
        slice_31: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_30, 3, 0, 128);  slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_39: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_31, convert_element_type_1582, full_default_31);  slice_31 = convert_element_type_1582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_800: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_39, [512, 128, 128]);  where_39 = None
        bmm_90: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_614, view_800);  permute_614 = None
        bmm_91: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_800, permute_615);  view_800 = permute_615 = None
        view_801: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_90, [32, 16, 128, 128]);  bmm_90 = None
        view_802: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_91, [32, 16, 128, 128]);  bmm_91 = None
        convert_element_type_1588: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_801, torch.float32);  view_801 = None
        permute_616: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1588, [0, 1, 3, 2]);  convert_element_type_1588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1589: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_616, torch.bfloat16);  permute_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_13: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_802, torch.bfloat16);  view_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_617: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_798, [0, 2, 1, 3]);  view_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_211: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_617, memory_format = torch.contiguous_format);  permute_617 = None
        view_803: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_211, [32, 128, 2048]);  clone_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_618: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1589, [0, 2, 1, 3]);  convert_element_type_1589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_804: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_618, [32, 128, 2048]);  permute_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_619: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_13, [0, 2, 1, 3]);  convert_element_type_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_212: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_619, memory_format = torch.contiguous_format);  permute_619 = None
        view_805: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_212, [32, 128, 2048]);  clone_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_806: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_803, [4096, 2048]);  view_803 = None
        permute_620: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_806, [1, 0])
        mm_201: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_620, view_286);  permute_620 = None
        mm_202: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_806, permute_622);  view_806 = permute_622 = None
        view_807: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_202, [32, 128, 2048]);  mm_202 = None
        convert_element_type_1595: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_807, torch.float32);  view_807 = None
        convert_element_type_1596: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_201, torch.float32);  mm_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_213: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_804, memory_format = torch.contiguous_format);  view_804 = None
        view_808: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_213, [4096, 2048]);  clone_213 = None
        permute_624: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_808, [1, 0])
        mm_203: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_624, view_286);  permute_624 = None
        mm_204: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_808, permute_626);  view_808 = permute_626 = None
        view_809: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_204, [32, 128, 2048]);  mm_204 = None
        convert_element_type_1601: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_809, torch.float32);  view_809 = None
        add_286: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1595, convert_element_type_1601);  convert_element_type_1595 = convert_element_type_1601 = None
        convert_element_type_1602: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_203, torch.float32);  mm_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_810: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_805, [4096, 2048]);  view_805 = None
        permute_628: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_810, [1, 0])
        mm_205: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_628, view_286);  permute_628 = view_286 = None
        mm_206: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_810, permute_630);  view_810 = permute_630 = None
        view_811: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_206, [32, 128, 2048]);  mm_206 = None
        convert_element_type_1607: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_811, torch.float32);  view_811 = None
        add_287: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_286, convert_element_type_1607);  add_286 = convert_element_type_1607 = None
        convert_element_type_1608: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_205, torch.float32);  mm_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_461: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_287, primals_186);  primals_186 = None
        mul_462: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_461, 2048)
        sum_161: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True)
        mul_463: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_461, mul_104);  mul_461 = None
        sum_162: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_463, [2], True);  mul_463 = None
        mul_464: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, sum_162);  sum_162 = None
        sub_156: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_462, sum_161);  mul_462 = sum_161 = None
        sub_157: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_156, mul_464);  sub_156 = mul_464 = None
        mul_465: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_48, sub_157);  div_48 = sub_157 = None
        mul_466: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_287, mul_104);  mul_104 = None
        sum_163: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_466, [0, 1]);  mul_466 = None
        sum_164: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_287, [0, 1]);  add_287 = None
        add_288: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_285, mul_465);  add_285 = mul_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1609: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_288, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_812: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1609, [4096, 2048]);  convert_element_type_1609 = None
        mm_207: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_812, permute_632);  permute_632 = None
        permute_633: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_812, [1, 0])
        mm_208: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_633, view_284);  permute_633 = view_284 = None
        sum_165: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_812, [0], True, dtype = torch.float32);  view_812 = None
        view_813: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_165, [2048]);  sum_165 = None
        convert_element_type_1614: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_813, torch.bfloat16);  view_813 = None
        view_814: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_207, [32, 128, 8192]);  mm_207 = None
        convert_element_type_1615: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_814, torch.float32);  view_814 = None
        convert_element_type_1616: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_208, torch.float32);  mm_208 = None
        convert_element_type_1617: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1614, torch.float32);  convert_element_type_1614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_283: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [32, 128, 8192]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_100: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_283, 0.5)
        mul_467: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1615, mul_100);  mul_100 = None
        convert_element_type_500: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_283, torch.float32)
        pow_13: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_500, 3.0)
        mul_101: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_118: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_283, mul_101);  view_283 = mul_101 = None
        mul_102: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_118, 0.7978845608028654);  add_118 = None
        tanh_12: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_102);  mul_102 = None
        add_119: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
        mul_468: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1615, add_119);  convert_element_type_1615 = add_119 = None
        convert_element_type_1618: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_468, torch.bfloat16);  mul_468 = None
        mul_469: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
        sub_158: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_469);  mul_469 = None
        mul_470: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_467, sub_158);  mul_467 = sub_158 = None
        mul_471: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_470, 0.7978845608028654);  mul_470 = None
        convert_element_type_1619: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_471, torch.bfloat16)
        mul_472: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_471, 0.044715);  mul_471 = None
        pow_36: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_500, 2.0);  convert_element_type_500 = None
        mul_473: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_36, 3.0);  pow_36 = None
        mul_474: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_472, mul_473);  mul_472 = mul_473 = None
        convert_element_type_1620: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_474, torch.bfloat16);  mul_474 = None
        add_289: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1619, convert_element_type_1620);  convert_element_type_1619 = convert_element_type_1620 = None
        mul_475: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1618, 0.5);  convert_element_type_1618 = None
        add_290: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_289, mul_475);  add_289 = mul_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_815: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_290, [4096, 8192]);  add_290 = None
        mm_209: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_815, permute_636);  permute_636 = None
        permute_637: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_815, [1, 0])
        mm_210: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_637, view_282);  permute_637 = view_282 = None
        sum_166: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_815, [0], True, dtype = torch.float32);  view_815 = None
        view_816: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_166, [8192]);  sum_166 = None
        convert_element_type_1625: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_816, torch.bfloat16);  view_816 = None
        view_817: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_209, [32, 128, 2048]);  mm_209 = None
        convert_element_type_1626: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_817, torch.float32);  view_817 = None
        convert_element_type_1627: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_210, torch.float32);  mm_210 = None
        convert_element_type_1628: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1625, torch.float32);  convert_element_type_1625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_477: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1626, primals_180);  primals_180 = None
        mul_478: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_477, 2048)
        sum_167: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_477, [2], True)
        mul_479: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_477, mul_98);  mul_477 = None
        sum_168: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_479, [2], True);  mul_479 = None
        mul_480: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, sum_168);  sum_168 = None
        sub_160: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_478, sum_167);  mul_478 = sum_167 = None
        sub_161: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_160, mul_480);  sub_160 = mul_480 = None
        mul_481: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_49, sub_161);  div_49 = sub_161 = None
        mul_482: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1626, mul_98);  mul_98 = None
        sum_169: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 1]);  mul_482 = None
        sum_170: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1626, [0, 1]);  convert_element_type_1626 = None
        add_291: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_288, mul_481);  add_288 = mul_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1629: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_291, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_818: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1629, [4096, 2048]);  convert_element_type_1629 = None
        mm_211: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_818, permute_640);  permute_640 = None
        permute_641: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_818, [1, 0])
        mm_212: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_641, view_280);  permute_641 = view_280 = None
        sum_171: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_818, [0], True, dtype = torch.float32);  view_818 = None
        view_819: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_171, [2048]);  sum_171 = None
        convert_element_type_1634: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_819, torch.bfloat16);  view_819 = None
        view_820: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_211, [32, 128, 2048]);  mm_211 = None
        convert_element_type_1635: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_212, torch.float32);  mm_212 = None
        convert_element_type_1636: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1634, torch.float32);  convert_element_type_1634 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_821: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_820, [32, 128, 16, 128]);  view_820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_644: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_821, [0, 2, 1, 3]);  view_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_214: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_644, memory_format = torch.contiguous_format);  permute_644 = None
        view_822: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_214, [512, 128, 128]);  clone_214 = None
        bmm_92: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_645, view_822);  permute_645 = None
        bmm_93: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_822, permute_646);  view_822 = permute_646 = None
        view_823: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_92, [32, 16, 128, 128]);  bmm_92 = None
        view_824: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_93, [32, 16, 128, 128]);  bmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1641: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_824, torch.float32);  view_824 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_483: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1641, div_12);  convert_element_type_1641 = None
        sum_172: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_483, [-1], True)
        neg_12: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_12);  div_12 = None
        fma_11: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_12, sum_172, mul_483);  neg_12 = sum_172 = mul_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1642: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_28: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_177, 2, 0, 128);  primals_177 = None
        slice_29: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_28, 3, 0, 128);  slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_40: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_29, convert_element_type_1642, full_default_31);  slice_29 = convert_element_type_1642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_825: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_40, [512, 128, 128]);  where_40 = None
        bmm_94: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_647, view_825);  permute_647 = None
        bmm_95: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_825, permute_648);  view_825 = permute_648 = None
        view_826: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_94, [32, 16, 128, 128]);  bmm_94 = None
        view_827: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_95, [32, 16, 128, 128]);  bmm_95 = None
        convert_element_type_1648: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_826, torch.float32);  view_826 = None
        permute_649: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1648, [0, 1, 3, 2]);  convert_element_type_1648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1649: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_649, torch.bfloat16);  permute_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_12: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_827, torch.bfloat16);  view_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_650: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_823, [0, 2, 1, 3]);  view_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_215: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_650, memory_format = torch.contiguous_format);  permute_650 = None
        view_828: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_215, [32, 128, 2048]);  clone_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_651: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1649, [0, 2, 1, 3]);  convert_element_type_1649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_829: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_651, [32, 128, 2048]);  permute_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_652: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_12, [0, 2, 1, 3]);  convert_element_type_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_216: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_652, memory_format = torch.contiguous_format);  permute_652 = None
        view_830: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_216, [32, 128, 2048]);  clone_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_831: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_828, [4096, 2048]);  view_828 = None
        permute_653: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_831, [1, 0])
        mm_213: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_653, view_264);  permute_653 = None
        mm_214: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_831, permute_655);  view_831 = permute_655 = None
        view_832: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_214, [32, 128, 2048]);  mm_214 = None
        convert_element_type_1655: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_832, torch.float32);  view_832 = None
        convert_element_type_1656: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_213, torch.float32);  mm_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_217: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_829, memory_format = torch.contiguous_format);  view_829 = None
        view_833: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_217, [4096, 2048]);  clone_217 = None
        permute_657: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_833, [1, 0])
        mm_215: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_657, view_264);  permute_657 = None
        mm_216: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_833, permute_659);  view_833 = permute_659 = None
        view_834: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_216, [32, 128, 2048]);  mm_216 = None
        convert_element_type_1661: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_834, torch.float32);  view_834 = None
        add_292: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1655, convert_element_type_1661);  convert_element_type_1655 = convert_element_type_1661 = None
        convert_element_type_1662: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_215, torch.float32);  mm_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_835: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_830, [4096, 2048]);  view_830 = None
        permute_661: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_835, [1, 0])
        mm_217: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_661, view_264);  permute_661 = view_264 = None
        mm_218: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_835, permute_663);  view_835 = permute_663 = None
        view_836: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_218, [32, 128, 2048]);  mm_218 = None
        convert_element_type_1667: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_836, torch.float32);  view_836 = None
        add_293: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_292, convert_element_type_1667);  add_292 = convert_element_type_1667 = None
        convert_element_type_1668: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_217, torch.float32);  mm_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_485: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_293, primals_172);  primals_172 = None
        mul_486: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_485, 2048)
        sum_173: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_485, [2], True)
        mul_487: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_485, mul_96);  mul_485 = None
        sum_174: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_487, [2], True);  mul_487 = None
        mul_488: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, sum_174);  sum_174 = None
        sub_163: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_486, sum_173);  mul_486 = sum_173 = None
        sub_164: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_163, mul_488);  sub_163 = mul_488 = None
        mul_489: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_50, sub_164);  div_50 = sub_164 = None
        mul_490: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_293, mul_96);  mul_96 = None
        sum_175: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 1]);  mul_490 = None
        sum_176: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_293, [0, 1]);  add_293 = None
        add_294: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_291, mul_489);  add_291 = mul_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1669: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_294, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_837: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1669, [4096, 2048]);  convert_element_type_1669 = None
        mm_219: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_837, permute_665);  permute_665 = None
        permute_666: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_837, [1, 0])
        mm_220: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_666, view_262);  permute_666 = view_262 = None
        sum_177: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_837, [0], True, dtype = torch.float32);  view_837 = None
        view_838: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_177, [2048]);  sum_177 = None
        convert_element_type_1674: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_838, torch.bfloat16);  view_838 = None
        view_839: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_219, [32, 128, 8192]);  mm_219 = None
        convert_element_type_1675: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_839, torch.float32);  view_839 = None
        convert_element_type_1676: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_220, torch.float32);  mm_220 = None
        convert_element_type_1677: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1674, torch.float32);  convert_element_type_1674 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_261: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [32, 128, 8192]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_491: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1675, mul_92);  mul_92 = None
        convert_element_type_461: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_261, torch.float32)
        pow_12: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_461, 3.0)
        mul_93: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_109: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_261, mul_93);  view_261 = mul_93 = None
        mul_94: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_109, 0.7978845608028654);  add_109 = None
        tanh_11: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_110: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_492: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1675, add_110);  convert_element_type_1675 = add_110 = None
        convert_element_type_1678: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_492, torch.bfloat16);  mul_492 = None
        mul_493: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_165: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_493);  mul_493 = None
        mul_494: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_491, sub_165);  mul_491 = sub_165 = None
        mul_495: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_494, 0.7978845608028654);  mul_494 = None
        convert_element_type_1679: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_495, torch.bfloat16)
        mul_496: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_495, 0.044715);  mul_495 = None
        pow_37: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_461, 2.0);  convert_element_type_461 = None
        mul_497: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_37, 3.0);  pow_37 = None
        mul_498: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_496, mul_497);  mul_496 = mul_497 = None
        convert_element_type_1680: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_498, torch.bfloat16);  mul_498 = None
        add_295: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1679, convert_element_type_1680);  convert_element_type_1679 = convert_element_type_1680 = None
        mul_499: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1678, 0.5);  convert_element_type_1678 = None
        add_296: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_295, mul_499);  add_295 = mul_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_840: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_296, [4096, 8192]);  add_296 = None
        mm_221: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_840, permute_669);  permute_669 = None
        permute_670: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_840, [1, 0])
        mm_222: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_670, view_260);  permute_670 = view_260 = None
        sum_178: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_840, [0], True, dtype = torch.float32);  view_840 = None
        view_841: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_178, [8192]);  sum_178 = None
        convert_element_type_1685: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_841, torch.bfloat16);  view_841 = None
        view_842: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_221, [32, 128, 2048]);  mm_221 = None
        convert_element_type_1686: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_842, torch.float32);  view_842 = None
        convert_element_type_1687: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_222, torch.float32);  mm_222 = None
        convert_element_type_1688: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1685, torch.float32);  convert_element_type_1685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_501: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1686, primals_166);  primals_166 = None
        mul_502: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_501, 2048)
        sum_179: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_501, [2], True)
        mul_503: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_501, mul_90);  mul_501 = None
        sum_180: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_503, [2], True);  mul_503 = None
        mul_504: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, sum_180);  sum_180 = None
        sub_167: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_502, sum_179);  mul_502 = sum_179 = None
        sub_168: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_167, mul_504);  sub_167 = mul_504 = None
        mul_505: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_51, sub_168);  div_51 = sub_168 = None
        mul_506: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1686, mul_90);  mul_90 = None
        sum_181: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_506, [0, 1]);  mul_506 = None
        sum_182: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1686, [0, 1]);  convert_element_type_1686 = None
        add_297: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_294, mul_505);  add_294 = mul_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1689: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_297, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_843: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1689, [4096, 2048]);  convert_element_type_1689 = None
        mm_223: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_843, permute_673);  permute_673 = None
        permute_674: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_843, [1, 0])
        mm_224: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_674, view_258);  permute_674 = view_258 = None
        sum_183: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_843, [0], True, dtype = torch.float32);  view_843 = None
        view_844: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_183, [2048]);  sum_183 = None
        convert_element_type_1694: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_844, torch.bfloat16);  view_844 = None
        view_845: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_223, [32, 128, 2048]);  mm_223 = None
        convert_element_type_1695: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_224, torch.float32);  mm_224 = None
        convert_element_type_1696: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1694, torch.float32);  convert_element_type_1694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_846: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_845, [32, 128, 16, 128]);  view_845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_677: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_846, [0, 2, 1, 3]);  view_846 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_218: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_677, memory_format = torch.contiguous_format);  permute_677 = None
        view_847: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_218, [512, 128, 128]);  clone_218 = None
        bmm_96: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_678, view_847);  permute_678 = None
        bmm_97: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_847, permute_679);  view_847 = permute_679 = None
        view_848: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_96, [32, 16, 128, 128]);  bmm_96 = None
        view_849: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_97, [32, 16, 128, 128]);  bmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1701: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_849, torch.float32);  view_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_507: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1701, div_11);  convert_element_type_1701 = None
        sum_184: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_507, [-1], True)
        neg_13: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_11);  div_11 = None
        fma_12: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_13, sum_184, mul_507);  neg_13 = sum_184 = mul_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1702: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_12, torch.bfloat16);  fma_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_26: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_163, 2, 0, 128);  primals_163 = None
        slice_27: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_26, 3, 0, 128);  slice_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_41: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_27, convert_element_type_1702, full_default_31);  slice_27 = convert_element_type_1702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_850: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_41, [512, 128, 128]);  where_41 = None
        bmm_98: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_680, view_850);  permute_680 = None
        bmm_99: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_850, permute_681);  view_850 = permute_681 = None
        view_851: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_98, [32, 16, 128, 128]);  bmm_98 = None
        view_852: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_99, [32, 16, 128, 128]);  bmm_99 = None
        convert_element_type_1708: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_851, torch.float32);  view_851 = None
        permute_682: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1708, [0, 1, 3, 2]);  convert_element_type_1708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1709: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_682, torch.bfloat16);  permute_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_11: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_852, torch.bfloat16);  view_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_683: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_848, [0, 2, 1, 3]);  view_848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_219: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_683, memory_format = torch.contiguous_format);  permute_683 = None
        view_853: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_219, [32, 128, 2048]);  clone_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_684: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1709, [0, 2, 1, 3]);  convert_element_type_1709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_854: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_684, [32, 128, 2048]);  permute_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_685: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_11, [0, 2, 1, 3]);  convert_element_type_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_220: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_685, memory_format = torch.contiguous_format);  permute_685 = None
        view_855: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_220, [32, 128, 2048]);  clone_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_856: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_853, [4096, 2048]);  view_853 = None
        permute_686: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_856, [1, 0])
        mm_225: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_686, view_242);  permute_686 = None
        mm_226: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_856, permute_688);  view_856 = permute_688 = None
        view_857: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_226, [32, 128, 2048]);  mm_226 = None
        convert_element_type_1715: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_857, torch.float32);  view_857 = None
        convert_element_type_1716: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_225, torch.float32);  mm_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_221: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_854, memory_format = torch.contiguous_format);  view_854 = None
        view_858: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_221, [4096, 2048]);  clone_221 = None
        permute_690: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_858, [1, 0])
        mm_227: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_690, view_242);  permute_690 = None
        mm_228: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_858, permute_692);  view_858 = permute_692 = None
        view_859: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_228, [32, 128, 2048]);  mm_228 = None
        convert_element_type_1721: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_859, torch.float32);  view_859 = None
        add_298: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1715, convert_element_type_1721);  convert_element_type_1715 = convert_element_type_1721 = None
        convert_element_type_1722: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_227, torch.float32);  mm_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_860: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_855, [4096, 2048]);  view_855 = None
        permute_694: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_860, [1, 0])
        mm_229: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_694, view_242);  permute_694 = view_242 = None
        mm_230: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_860, permute_696);  view_860 = permute_696 = None
        view_861: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_230, [32, 128, 2048]);  mm_230 = None
        convert_element_type_1727: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_861, torch.float32);  view_861 = None
        add_299: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_298, convert_element_type_1727);  add_298 = convert_element_type_1727 = None
        convert_element_type_1728: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_229, torch.float32);  mm_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_509: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_299, primals_158);  primals_158 = None
        mul_510: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_509, 2048)
        sum_185: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_509, [2], True)
        mul_511: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_509, mul_88);  mul_509 = None
        sum_186: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_511, [2], True);  mul_511 = None
        mul_512: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, sum_186);  sum_186 = None
        sub_170: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_510, sum_185);  mul_510 = sum_185 = None
        sub_171: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_170, mul_512);  sub_170 = mul_512 = None
        mul_513: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_52, sub_171);  div_52 = sub_171 = None
        mul_514: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_299, mul_88);  mul_88 = None
        sum_187: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_514, [0, 1]);  mul_514 = None
        sum_188: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_299, [0, 1]);  add_299 = None
        add_300: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_297, mul_513);  add_297 = mul_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1729: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_300, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_862: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1729, [4096, 2048]);  convert_element_type_1729 = None
        mm_231: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_862, permute_698);  permute_698 = None
        permute_699: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_862, [1, 0])
        mm_232: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_699, view_240);  permute_699 = view_240 = None
        sum_189: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_862, [0], True, dtype = torch.float32);  view_862 = None
        view_863: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_189, [2048]);  sum_189 = None
        convert_element_type_1734: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_863, torch.bfloat16);  view_863 = None
        view_864: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_231, [32, 128, 8192]);  mm_231 = None
        convert_element_type_1735: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_864, torch.float32);  view_864 = None
        convert_element_type_1736: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_232, torch.float32);  mm_232 = None
        convert_element_type_1737: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1734, torch.float32);  convert_element_type_1734 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_239: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [32, 128, 8192]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_84: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_515: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1735, mul_84);  mul_84 = None
        convert_element_type_422: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32)
        pow_11: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_422, 3.0)
        mul_85: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_100: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_239, mul_85);  view_239 = mul_85 = None
        mul_86: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_10: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_101: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_516: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1735, add_101);  convert_element_type_1735 = add_101 = None
        convert_element_type_1738: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_516, torch.bfloat16);  mul_516 = None
        mul_517: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_172: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_517);  mul_517 = None
        mul_518: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_515, sub_172);  mul_515 = sub_172 = None
        mul_519: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_518, 0.7978845608028654);  mul_518 = None
        convert_element_type_1739: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_519, torch.bfloat16)
        mul_520: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_519, 0.044715);  mul_519 = None
        pow_38: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_422, 2.0);  convert_element_type_422 = None
        mul_521: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_38, 3.0);  pow_38 = None
        mul_522: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_520, mul_521);  mul_520 = mul_521 = None
        convert_element_type_1740: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_522, torch.bfloat16);  mul_522 = None
        add_301: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1739, convert_element_type_1740);  convert_element_type_1739 = convert_element_type_1740 = None
        mul_523: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1738, 0.5);  convert_element_type_1738 = None
        add_302: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_301, mul_523);  add_301 = mul_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_865: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_302, [4096, 8192]);  add_302 = None
        mm_233: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_865, permute_702);  permute_702 = None
        permute_703: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_865, [1, 0])
        mm_234: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_703, view_238);  permute_703 = view_238 = None
        sum_190: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_865, [0], True, dtype = torch.float32);  view_865 = None
        view_866: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_190, [8192]);  sum_190 = None
        convert_element_type_1745: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_866, torch.bfloat16);  view_866 = None
        view_867: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_233, [32, 128, 2048]);  mm_233 = None
        convert_element_type_1746: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_867, torch.float32);  view_867 = None
        convert_element_type_1747: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_234, torch.float32);  mm_234 = None
        convert_element_type_1748: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1745, torch.float32);  convert_element_type_1745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_525: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1746, primals_152);  primals_152 = None
        mul_526: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_525, 2048)
        sum_191: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_525, [2], True)
        mul_527: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_525, mul_82);  mul_525 = None
        sum_192: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_527, [2], True);  mul_527 = None
        mul_528: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, sum_192);  sum_192 = None
        sub_174: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_526, sum_191);  mul_526 = sum_191 = None
        sub_175: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_174, mul_528);  sub_174 = mul_528 = None
        mul_529: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_53, sub_175);  div_53 = sub_175 = None
        mul_530: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1746, mul_82);  mul_82 = None
        sum_193: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_530, [0, 1]);  mul_530 = None
        sum_194: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1746, [0, 1]);  convert_element_type_1746 = None
        add_303: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_300, mul_529);  add_300 = mul_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1749: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_303, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_868: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1749, [4096, 2048]);  convert_element_type_1749 = None
        mm_235: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_868, permute_706);  permute_706 = None
        permute_707: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_868, [1, 0])
        mm_236: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_707, view_236);  permute_707 = view_236 = None
        sum_195: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_868, [0], True, dtype = torch.float32);  view_868 = None
        view_869: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_195, [2048]);  sum_195 = None
        convert_element_type_1754: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_869, torch.bfloat16);  view_869 = None
        view_870: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_235, [32, 128, 2048]);  mm_235 = None
        convert_element_type_1755: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_236, torch.float32);  mm_236 = None
        convert_element_type_1756: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1754, torch.float32);  convert_element_type_1754 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_871: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_870, [32, 128, 16, 128]);  view_870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_710: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_871, [0, 2, 1, 3]);  view_871 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_222: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_710, memory_format = torch.contiguous_format);  permute_710 = None
        view_872: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_222, [512, 128, 128]);  clone_222 = None
        bmm_100: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_711, view_872);  permute_711 = None
        bmm_101: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_872, permute_712);  view_872 = permute_712 = None
        view_873: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_100, [32, 16, 128, 128]);  bmm_100 = None
        view_874: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_101, [32, 16, 128, 128]);  bmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1761: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_874, torch.float32);  view_874 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_531: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1761, div_10);  convert_element_type_1761 = None
        sum_196: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_531, [-1], True)
        neg_14: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_13: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_14, sum_196, mul_531);  neg_14 = sum_196 = mul_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1762: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_13, torch.bfloat16);  fma_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_24: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_149, 2, 0, 128);  primals_149 = None
        slice_25: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_24, 3, 0, 128);  slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_42: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_25, convert_element_type_1762, full_default_31);  slice_25 = convert_element_type_1762 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_875: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_42, [512, 128, 128]);  where_42 = None
        bmm_102: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_713, view_875);  permute_713 = None
        bmm_103: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_875, permute_714);  view_875 = permute_714 = None
        view_876: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_102, [32, 16, 128, 128]);  bmm_102 = None
        view_877: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_103, [32, 16, 128, 128]);  bmm_103 = None
        convert_element_type_1768: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_876, torch.float32);  view_876 = None
        permute_715: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1768, [0, 1, 3, 2]);  convert_element_type_1768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1769: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_715, torch.bfloat16);  permute_715 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_10: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_877, torch.bfloat16);  view_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_716: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_873, [0, 2, 1, 3]);  view_873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_223: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_716, memory_format = torch.contiguous_format);  permute_716 = None
        view_878: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_223, [32, 128, 2048]);  clone_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_717: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1769, [0, 2, 1, 3]);  convert_element_type_1769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_879: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_717, [32, 128, 2048]);  permute_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_718: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_10, [0, 2, 1, 3]);  convert_element_type_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_224: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_718, memory_format = torch.contiguous_format);  permute_718 = None
        view_880: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_224, [32, 128, 2048]);  clone_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_881: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_878, [4096, 2048]);  view_878 = None
        permute_719: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_881, [1, 0])
        mm_237: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_719, view_220);  permute_719 = None
        mm_238: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_881, permute_721);  view_881 = permute_721 = None
        view_882: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_238, [32, 128, 2048]);  mm_238 = None
        convert_element_type_1775: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_882, torch.float32);  view_882 = None
        convert_element_type_1776: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_237, torch.float32);  mm_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_225: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_879, memory_format = torch.contiguous_format);  view_879 = None
        view_883: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_225, [4096, 2048]);  clone_225 = None
        permute_723: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_883, [1, 0])
        mm_239: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_723, view_220);  permute_723 = None
        mm_240: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_883, permute_725);  view_883 = permute_725 = None
        view_884: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_240, [32, 128, 2048]);  mm_240 = None
        convert_element_type_1781: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_884, torch.float32);  view_884 = None
        add_304: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1775, convert_element_type_1781);  convert_element_type_1775 = convert_element_type_1781 = None
        convert_element_type_1782: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_239, torch.float32);  mm_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_885: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_880, [4096, 2048]);  view_880 = None
        permute_727: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_885, [1, 0])
        mm_241: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_727, view_220);  permute_727 = view_220 = None
        mm_242: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_885, permute_729);  view_885 = permute_729 = None
        view_886: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_242, [32, 128, 2048]);  mm_242 = None
        convert_element_type_1787: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_886, torch.float32);  view_886 = None
        add_305: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_304, convert_element_type_1787);  add_304 = convert_element_type_1787 = None
        convert_element_type_1788: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_241, torch.float32);  mm_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_533: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_305, primals_144);  primals_144 = None
        mul_534: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_533, 2048)
        sum_197: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_533, [2], True)
        mul_535: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_533, mul_80);  mul_533 = None
        sum_198: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_535, [2], True);  mul_535 = None
        mul_536: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, sum_198);  sum_198 = None
        sub_177: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_534, sum_197);  mul_534 = sum_197 = None
        sub_178: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_177, mul_536);  sub_177 = mul_536 = None
        mul_537: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_54, sub_178);  div_54 = sub_178 = None
        mul_538: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_305, mul_80);  mul_80 = None
        sum_199: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_538, [0, 1]);  mul_538 = None
        sum_200: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_305, [0, 1]);  add_305 = None
        add_306: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_303, mul_537);  add_303 = mul_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1789: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_306, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_887: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1789, [4096, 2048]);  convert_element_type_1789 = None
        mm_243: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_887, permute_731);  permute_731 = None
        permute_732: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_887, [1, 0])
        mm_244: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_732, view_218);  permute_732 = view_218 = None
        sum_201: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_887, [0], True, dtype = torch.float32);  view_887 = None
        view_888: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_201, [2048]);  sum_201 = None
        convert_element_type_1794: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_888, torch.bfloat16);  view_888 = None
        view_889: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_243, [32, 128, 8192]);  mm_243 = None
        convert_element_type_1795: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_889, torch.float32);  view_889 = None
        convert_element_type_1796: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_244, torch.float32);  mm_244 = None
        convert_element_type_1797: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1794, torch.float32);  convert_element_type_1794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_217: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [32, 128, 8192]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_539: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1795, mul_76);  mul_76 = None
        convert_element_type_383: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_217, torch.float32)
        pow_10: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_383, 3.0)
        mul_77: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_91: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_217, mul_77);  view_217 = mul_77 = None
        mul_78: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_91, 0.7978845608028654);  add_91 = None
        tanh_9: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_92: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_540: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1795, add_92);  convert_element_type_1795 = add_92 = None
        convert_element_type_1798: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_540, torch.bfloat16);  mul_540 = None
        mul_541: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_179: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_541);  mul_541 = None
        mul_542: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_539, sub_179);  mul_539 = sub_179 = None
        mul_543: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_542, 0.7978845608028654);  mul_542 = None
        convert_element_type_1799: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_543, torch.bfloat16)
        mul_544: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_543, 0.044715);  mul_543 = None
        pow_39: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_383, 2.0);  convert_element_type_383 = None
        mul_545: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_39, 3.0);  pow_39 = None
        mul_546: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_544, mul_545);  mul_544 = mul_545 = None
        convert_element_type_1800: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_546, torch.bfloat16);  mul_546 = None
        add_307: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1799, convert_element_type_1800);  convert_element_type_1799 = convert_element_type_1800 = None
        mul_547: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1798, 0.5);  convert_element_type_1798 = None
        add_308: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_307, mul_547);  add_307 = mul_547 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_890: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_308, [4096, 8192]);  add_308 = None
        mm_245: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_890, permute_735);  permute_735 = None
        permute_736: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_890, [1, 0])
        mm_246: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_736, view_216);  permute_736 = view_216 = None
        sum_202: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_890, [0], True, dtype = torch.float32);  view_890 = None
        view_891: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_202, [8192]);  sum_202 = None
        convert_element_type_1805: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_891, torch.bfloat16);  view_891 = None
        view_892: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_245, [32, 128, 2048]);  mm_245 = None
        convert_element_type_1806: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_892, torch.float32);  view_892 = None
        convert_element_type_1807: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_246, torch.float32);  mm_246 = None
        convert_element_type_1808: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1805, torch.float32);  convert_element_type_1805 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_549: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1806, primals_138);  primals_138 = None
        mul_550: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_549, 2048)
        sum_203: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_549, [2], True)
        mul_551: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_549, mul_74);  mul_549 = None
        sum_204: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_551, [2], True);  mul_551 = None
        mul_552: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, sum_204);  sum_204 = None
        sub_181: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_550, sum_203);  mul_550 = sum_203 = None
        sub_182: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_181, mul_552);  sub_181 = mul_552 = None
        mul_553: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_55, sub_182);  div_55 = sub_182 = None
        mul_554: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1806, mul_74);  mul_74 = None
        sum_205: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_554, [0, 1]);  mul_554 = None
        sum_206: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1806, [0, 1]);  convert_element_type_1806 = None
        add_309: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_306, mul_553);  add_306 = mul_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1809: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_309, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_893: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1809, [4096, 2048]);  convert_element_type_1809 = None
        mm_247: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_893, permute_739);  permute_739 = None
        permute_740: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_893, [1, 0])
        mm_248: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_740, view_214);  permute_740 = view_214 = None
        sum_207: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_893, [0], True, dtype = torch.float32);  view_893 = None
        view_894: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_207, [2048]);  sum_207 = None
        convert_element_type_1814: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_894, torch.bfloat16);  view_894 = None
        view_895: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_247, [32, 128, 2048]);  mm_247 = None
        convert_element_type_1815: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_248, torch.float32);  mm_248 = None
        convert_element_type_1816: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1814, torch.float32);  convert_element_type_1814 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_896: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_895, [32, 128, 16, 128]);  view_895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_743: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_896, [0, 2, 1, 3]);  view_896 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_226: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_743, memory_format = torch.contiguous_format);  permute_743 = None
        view_897: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_226, [512, 128, 128]);  clone_226 = None
        bmm_104: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_744, view_897);  permute_744 = None
        bmm_105: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_897, permute_745);  view_897 = permute_745 = None
        view_898: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_104, [32, 16, 128, 128]);  bmm_104 = None
        view_899: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_105, [32, 16, 128, 128]);  bmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1821: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_899, torch.float32);  view_899 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_555: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1821, div_9);  convert_element_type_1821 = None
        sum_208: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_555, [-1], True)
        neg_15: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_9);  div_9 = None
        fma_14: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_15, sum_208, mul_555);  neg_15 = sum_208 = mul_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1822: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_14, torch.bfloat16);  fma_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_22: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_135, 2, 0, 128);  primals_135 = None
        slice_23: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_22, 3, 0, 128);  slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_43: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_23, convert_element_type_1822, full_default_31);  slice_23 = convert_element_type_1822 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_900: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_43, [512, 128, 128]);  where_43 = None
        bmm_106: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_746, view_900);  permute_746 = None
        bmm_107: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_900, permute_747);  view_900 = permute_747 = None
        view_901: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_106, [32, 16, 128, 128]);  bmm_106 = None
        view_902: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_107, [32, 16, 128, 128]);  bmm_107 = None
        convert_element_type_1828: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_901, torch.float32);  view_901 = None
        permute_748: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1828, [0, 1, 3, 2]);  convert_element_type_1828 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1829: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_748, torch.bfloat16);  permute_748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_9: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_902, torch.bfloat16);  view_902 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_749: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_898, [0, 2, 1, 3]);  view_898 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_227: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_749, memory_format = torch.contiguous_format);  permute_749 = None
        view_903: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_227, [32, 128, 2048]);  clone_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_750: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1829, [0, 2, 1, 3]);  convert_element_type_1829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_904: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_750, [32, 128, 2048]);  permute_750 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_751: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_9, [0, 2, 1, 3]);  convert_element_type_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_228: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_751, memory_format = torch.contiguous_format);  permute_751 = None
        view_905: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_228, [32, 128, 2048]);  clone_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_906: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_903, [4096, 2048]);  view_903 = None
        permute_752: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_906, [1, 0])
        mm_249: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_752, view_198);  permute_752 = None
        mm_250: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_906, permute_754);  view_906 = permute_754 = None
        view_907: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_250, [32, 128, 2048]);  mm_250 = None
        convert_element_type_1835: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_907, torch.float32);  view_907 = None
        convert_element_type_1836: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_249, torch.float32);  mm_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_229: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_904, memory_format = torch.contiguous_format);  view_904 = None
        view_908: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_229, [4096, 2048]);  clone_229 = None
        permute_756: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_908, [1, 0])
        mm_251: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_756, view_198);  permute_756 = None
        mm_252: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_908, permute_758);  view_908 = permute_758 = None
        view_909: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_252, [32, 128, 2048]);  mm_252 = None
        convert_element_type_1841: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_909, torch.float32);  view_909 = None
        add_310: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1835, convert_element_type_1841);  convert_element_type_1835 = convert_element_type_1841 = None
        convert_element_type_1842: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_251, torch.float32);  mm_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_910: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_905, [4096, 2048]);  view_905 = None
        permute_760: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_910, [1, 0])
        mm_253: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_760, view_198);  permute_760 = view_198 = None
        mm_254: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_910, permute_762);  view_910 = permute_762 = None
        view_911: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_254, [32, 128, 2048]);  mm_254 = None
        convert_element_type_1847: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_911, torch.float32);  view_911 = None
        add_311: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_310, convert_element_type_1847);  add_310 = convert_element_type_1847 = None
        convert_element_type_1848: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_253, torch.float32);  mm_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_557: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_311, primals_130);  primals_130 = None
        mul_558: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_557, 2048)
        sum_209: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_557, [2], True)
        mul_559: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_557, mul_72);  mul_557 = None
        sum_210: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_559, [2], True);  mul_559 = None
        mul_560: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, sum_210);  sum_210 = None
        sub_184: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_558, sum_209);  mul_558 = sum_209 = None
        sub_185: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_184, mul_560);  sub_184 = mul_560 = None
        mul_561: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_56, sub_185);  div_56 = sub_185 = None
        mul_562: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_311, mul_72);  mul_72 = None
        sum_211: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 1]);  mul_562 = None
        sum_212: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_311, [0, 1]);  add_311 = None
        add_312: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_309, mul_561);  add_309 = mul_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1849: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_312, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_912: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1849, [4096, 2048]);  convert_element_type_1849 = None
        mm_255: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_912, permute_764);  permute_764 = None
        permute_765: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_912, [1, 0])
        mm_256: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_765, view_196);  permute_765 = view_196 = None
        sum_213: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_912, [0], True, dtype = torch.float32);  view_912 = None
        view_913: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_213, [2048]);  sum_213 = None
        convert_element_type_1854: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_913, torch.bfloat16);  view_913 = None
        view_914: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_255, [32, 128, 8192]);  mm_255 = None
        convert_element_type_1855: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_914, torch.float32);  view_914 = None
        convert_element_type_1856: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_256, torch.float32);  mm_256 = None
        convert_element_type_1857: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1854, torch.float32);  convert_element_type_1854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_195: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [32, 128, 8192]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_563: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1855, mul_68);  mul_68 = None
        convert_element_type_344: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_195, torch.float32)
        pow_9: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_344, 3.0)
        mul_69: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_82: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_195, mul_69);  view_195 = mul_69 = None
        mul_70: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_82, 0.7978845608028654);  add_82 = None
        tanh_8: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_83: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_564: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1855, add_83);  convert_element_type_1855 = add_83 = None
        convert_element_type_1858: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_564, torch.bfloat16);  mul_564 = None
        mul_565: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_186: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_565);  mul_565 = None
        mul_566: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_563, sub_186);  mul_563 = sub_186 = None
        mul_567: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_566, 0.7978845608028654);  mul_566 = None
        convert_element_type_1859: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_567, torch.bfloat16)
        mul_568: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_567, 0.044715);  mul_567 = None
        pow_40: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_344, 2.0);  convert_element_type_344 = None
        mul_569: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_40, 3.0);  pow_40 = None
        mul_570: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
        convert_element_type_1860: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_570, torch.bfloat16);  mul_570 = None
        add_313: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1859, convert_element_type_1860);  convert_element_type_1859 = convert_element_type_1860 = None
        mul_571: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1858, 0.5);  convert_element_type_1858 = None
        add_314: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_313, mul_571);  add_313 = mul_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_915: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_314, [4096, 8192]);  add_314 = None
        mm_257: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_915, permute_768);  permute_768 = None
        permute_769: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_915, [1, 0])
        mm_258: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_769, view_194);  permute_769 = view_194 = None
        sum_214: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_915, [0], True, dtype = torch.float32);  view_915 = None
        view_916: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_214, [8192]);  sum_214 = None
        convert_element_type_1865: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_916, torch.bfloat16);  view_916 = None
        view_917: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_257, [32, 128, 2048]);  mm_257 = None
        convert_element_type_1866: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_917, torch.float32);  view_917 = None
        convert_element_type_1867: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_258, torch.float32);  mm_258 = None
        convert_element_type_1868: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1865, torch.float32);  convert_element_type_1865 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_573: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1866, primals_124);  primals_124 = None
        mul_574: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_573, 2048)
        sum_215: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_573, [2], True)
        mul_575: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_573, mul_66);  mul_573 = None
        sum_216: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_575, [2], True);  mul_575 = None
        mul_576: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, sum_216);  sum_216 = None
        sub_188: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_574, sum_215);  mul_574 = sum_215 = None
        sub_189: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_188, mul_576);  sub_188 = mul_576 = None
        mul_577: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_57, sub_189);  div_57 = sub_189 = None
        mul_578: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1866, mul_66);  mul_66 = None
        sum_217: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_578, [0, 1]);  mul_578 = None
        sum_218: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1866, [0, 1]);  convert_element_type_1866 = None
        add_315: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_312, mul_577);  add_312 = mul_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1869: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_315, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_918: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1869, [4096, 2048]);  convert_element_type_1869 = None
        mm_259: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_918, permute_772);  permute_772 = None
        permute_773: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_918, [1, 0])
        mm_260: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_773, view_192);  permute_773 = view_192 = None
        sum_219: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_918, [0], True, dtype = torch.float32);  view_918 = None
        view_919: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_219, [2048]);  sum_219 = None
        convert_element_type_1874: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_919, torch.bfloat16);  view_919 = None
        view_920: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_259, [32, 128, 2048]);  mm_259 = None
        convert_element_type_1875: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_260, torch.float32);  mm_260 = None
        convert_element_type_1876: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1874, torch.float32);  convert_element_type_1874 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_921: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_920, [32, 128, 16, 128]);  view_920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_776: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_921, [0, 2, 1, 3]);  view_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_230: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_776, memory_format = torch.contiguous_format);  permute_776 = None
        view_922: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_230, [512, 128, 128]);  clone_230 = None
        bmm_108: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_777, view_922);  permute_777 = None
        bmm_109: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_922, permute_778);  view_922 = permute_778 = None
        view_923: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_108, [32, 16, 128, 128]);  bmm_108 = None
        view_924: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_109, [32, 16, 128, 128]);  bmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1881: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_924, torch.float32);  view_924 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_579: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1881, div_8);  convert_element_type_1881 = None
        sum_220: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_579, [-1], True)
        neg_16: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_8);  div_8 = None
        fma_15: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_16, sum_220, mul_579);  neg_16 = sum_220 = mul_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1882: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_15, torch.bfloat16);  fma_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_20: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_121, 2, 0, 128);  primals_121 = None
        slice_21: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_20, 3, 0, 128);  slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_44: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_21, convert_element_type_1882, full_default_31);  slice_21 = convert_element_type_1882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_925: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_44, [512, 128, 128]);  where_44 = None
        bmm_110: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_779, view_925);  permute_779 = None
        bmm_111: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_925, permute_780);  view_925 = permute_780 = None
        view_926: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_110, [32, 16, 128, 128]);  bmm_110 = None
        view_927: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_111, [32, 16, 128, 128]);  bmm_111 = None
        convert_element_type_1888: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_926, torch.float32);  view_926 = None
        permute_781: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1888, [0, 1, 3, 2]);  convert_element_type_1888 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1889: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_781, torch.bfloat16);  permute_781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_8: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_927, torch.bfloat16);  view_927 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_782: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_923, [0, 2, 1, 3]);  view_923 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_231: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_782, memory_format = torch.contiguous_format);  permute_782 = None
        view_928: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_231, [32, 128, 2048]);  clone_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_783: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1889, [0, 2, 1, 3]);  convert_element_type_1889 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_929: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_783, [32, 128, 2048]);  permute_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_784: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_8, [0, 2, 1, 3]);  convert_element_type_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_232: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_784, memory_format = torch.contiguous_format);  permute_784 = None
        view_930: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_232, [32, 128, 2048]);  clone_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_931: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_928, [4096, 2048]);  view_928 = None
        permute_785: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_931, [1, 0])
        mm_261: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_785, view_176);  permute_785 = None
        mm_262: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_931, permute_787);  view_931 = permute_787 = None
        view_932: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_262, [32, 128, 2048]);  mm_262 = None
        convert_element_type_1895: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_932, torch.float32);  view_932 = None
        convert_element_type_1896: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_261, torch.float32);  mm_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_233: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_929, memory_format = torch.contiguous_format);  view_929 = None
        view_933: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_233, [4096, 2048]);  clone_233 = None
        permute_789: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_933, [1, 0])
        mm_263: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_789, view_176);  permute_789 = None
        mm_264: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_933, permute_791);  view_933 = permute_791 = None
        view_934: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_264, [32, 128, 2048]);  mm_264 = None
        convert_element_type_1901: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_934, torch.float32);  view_934 = None
        add_316: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1895, convert_element_type_1901);  convert_element_type_1895 = convert_element_type_1901 = None
        convert_element_type_1902: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_263, torch.float32);  mm_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_935: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_930, [4096, 2048]);  view_930 = None
        permute_793: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_935, [1, 0])
        mm_265: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_793, view_176);  permute_793 = view_176 = None
        mm_266: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_935, permute_795);  view_935 = permute_795 = None
        view_936: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_266, [32, 128, 2048]);  mm_266 = None
        convert_element_type_1907: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_936, torch.float32);  view_936 = None
        add_317: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_316, convert_element_type_1907);  add_316 = convert_element_type_1907 = None
        convert_element_type_1908: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_265, torch.float32);  mm_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_581: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_317, primals_116);  primals_116 = None
        mul_582: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_581, 2048)
        sum_221: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_581, [2], True)
        mul_583: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_581, mul_64);  mul_581 = None
        sum_222: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_583, [2], True);  mul_583 = None
        mul_584: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, sum_222);  sum_222 = None
        sub_191: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_582, sum_221);  mul_582 = sum_221 = None
        sub_192: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_191, mul_584);  sub_191 = mul_584 = None
        mul_585: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_58, sub_192);  div_58 = sub_192 = None
        mul_586: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_317, mul_64);  mul_64 = None
        sum_223: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_586, [0, 1]);  mul_586 = None
        sum_224: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_317, [0, 1]);  add_317 = None
        add_318: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_315, mul_585);  add_315 = mul_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1909: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_318, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_937: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1909, [4096, 2048]);  convert_element_type_1909 = None
        mm_267: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_937, permute_797);  permute_797 = None
        permute_798: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_937, [1, 0])
        mm_268: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_798, view_174);  permute_798 = view_174 = None
        sum_225: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_937, [0], True, dtype = torch.float32);  view_937 = None
        view_938: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_225, [2048]);  sum_225 = None
        convert_element_type_1914: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_938, torch.bfloat16);  view_938 = None
        view_939: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_267, [32, 128, 8192]);  mm_267 = None
        convert_element_type_1915: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_939, torch.float32);  view_939 = None
        convert_element_type_1916: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_268, torch.float32);  mm_268 = None
        convert_element_type_1917: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1914, torch.float32);  convert_element_type_1914 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_173: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [32, 128, 8192]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_587: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1915, mul_60);  mul_60 = None
        convert_element_type_305: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32)
        pow_8: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_305, 3.0)
        mul_61: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_73: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_173, mul_61);  view_173 = mul_61 = None
        mul_62: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_7: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_74: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_588: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1915, add_74);  convert_element_type_1915 = add_74 = None
        convert_element_type_1918: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_588, torch.bfloat16);  mul_588 = None
        mul_589: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_193: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_589);  mul_589 = None
        mul_590: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_587, sub_193);  mul_587 = sub_193 = None
        mul_591: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_590, 0.7978845608028654);  mul_590 = None
        convert_element_type_1919: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_591, torch.bfloat16)
        mul_592: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_591, 0.044715);  mul_591 = None
        pow_41: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_305, 2.0);  convert_element_type_305 = None
        mul_593: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_41, 3.0);  pow_41 = None
        mul_594: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
        convert_element_type_1920: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_594, torch.bfloat16);  mul_594 = None
        add_319: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1919, convert_element_type_1920);  convert_element_type_1919 = convert_element_type_1920 = None
        mul_595: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1918, 0.5);  convert_element_type_1918 = None
        add_320: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_319, mul_595);  add_319 = mul_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_940: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_320, [4096, 8192]);  add_320 = None
        mm_269: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_940, permute_801);  permute_801 = None
        permute_802: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_940, [1, 0])
        mm_270: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_802, view_172);  permute_802 = view_172 = None
        sum_226: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_940, [0], True, dtype = torch.float32);  view_940 = None
        view_941: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_226, [8192]);  sum_226 = None
        convert_element_type_1925: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_941, torch.bfloat16);  view_941 = None
        view_942: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_269, [32, 128, 2048]);  mm_269 = None
        convert_element_type_1926: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_942, torch.float32);  view_942 = None
        convert_element_type_1927: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_270, torch.float32);  mm_270 = None
        convert_element_type_1928: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1925, torch.float32);  convert_element_type_1925 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_597: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1926, primals_110);  primals_110 = None
        mul_598: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_597, 2048)
        sum_227: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_597, [2], True)
        mul_599: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_597, mul_58);  mul_597 = None
        sum_228: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_599, [2], True);  mul_599 = None
        mul_600: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, sum_228);  sum_228 = None
        sub_195: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_598, sum_227);  mul_598 = sum_227 = None
        sub_196: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_195, mul_600);  sub_195 = mul_600 = None
        mul_601: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_59, sub_196);  div_59 = sub_196 = None
        mul_602: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1926, mul_58);  mul_58 = None
        sum_229: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_602, [0, 1]);  mul_602 = None
        sum_230: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1926, [0, 1]);  convert_element_type_1926 = None
        add_321: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_318, mul_601);  add_318 = mul_601 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1929: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_321, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_943: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1929, [4096, 2048]);  convert_element_type_1929 = None
        mm_271: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_943, permute_805);  permute_805 = None
        permute_806: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_943, [1, 0])
        mm_272: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_806, view_170);  permute_806 = view_170 = None
        sum_231: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_943, [0], True, dtype = torch.float32);  view_943 = None
        view_944: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_231, [2048]);  sum_231 = None
        convert_element_type_1934: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_944, torch.bfloat16);  view_944 = None
        view_945: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_271, [32, 128, 2048]);  mm_271 = None
        convert_element_type_1935: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_272, torch.float32);  mm_272 = None
        convert_element_type_1936: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1934, torch.float32);  convert_element_type_1934 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_946: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_945, [32, 128, 16, 128]);  view_945 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_809: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_946, [0, 2, 1, 3]);  view_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_234: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_809, memory_format = torch.contiguous_format);  permute_809 = None
        view_947: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_234, [512, 128, 128]);  clone_234 = None
        bmm_112: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_810, view_947);  permute_810 = None
        bmm_113: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_947, permute_811);  view_947 = permute_811 = None
        view_948: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_112, [32, 16, 128, 128]);  bmm_112 = None
        view_949: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_113, [32, 16, 128, 128]);  bmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1941: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_949, torch.float32);  view_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_603: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1941, div_7);  convert_element_type_1941 = None
        sum_232: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_603, [-1], True)
        neg_17: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_16: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_17, sum_232, mul_603);  neg_17 = sum_232 = mul_603 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1942: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_16, torch.bfloat16);  fma_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_18: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_107, 2, 0, 128);  primals_107 = None
        slice_19: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_18, 3, 0, 128);  slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_45: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_19, convert_element_type_1942, full_default_31);  slice_19 = convert_element_type_1942 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_950: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_45, [512, 128, 128]);  where_45 = None
        bmm_114: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_812, view_950);  permute_812 = None
        bmm_115: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_950, permute_813);  view_950 = permute_813 = None
        view_951: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_114, [32, 16, 128, 128]);  bmm_114 = None
        view_952: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_115, [32, 16, 128, 128]);  bmm_115 = None
        convert_element_type_1948: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_951, torch.float32);  view_951 = None
        permute_814: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1948, [0, 1, 3, 2]);  convert_element_type_1948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1949: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_814, torch.bfloat16);  permute_814 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_7: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_952, torch.bfloat16);  view_952 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_815: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_948, [0, 2, 1, 3]);  view_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_235: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_815, memory_format = torch.contiguous_format);  permute_815 = None
        view_953: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_235, [32, 128, 2048]);  clone_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_816: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1949, [0, 2, 1, 3]);  convert_element_type_1949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_954: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_816, [32, 128, 2048]);  permute_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_817: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_7, [0, 2, 1, 3]);  convert_element_type_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_236: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_817, memory_format = torch.contiguous_format);  permute_817 = None
        view_955: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_236, [32, 128, 2048]);  clone_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_956: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_953, [4096, 2048]);  view_953 = None
        permute_818: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_956, [1, 0])
        mm_273: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_818, view_154);  permute_818 = None
        mm_274: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_956, permute_820);  view_956 = permute_820 = None
        view_957: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_274, [32, 128, 2048]);  mm_274 = None
        convert_element_type_1955: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_957, torch.float32);  view_957 = None
        convert_element_type_1956: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_273, torch.float32);  mm_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_237: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_954, memory_format = torch.contiguous_format);  view_954 = None
        view_958: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_237, [4096, 2048]);  clone_237 = None
        permute_822: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_958, [1, 0])
        mm_275: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_822, view_154);  permute_822 = None
        mm_276: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_958, permute_824);  view_958 = permute_824 = None
        view_959: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_276, [32, 128, 2048]);  mm_276 = None
        convert_element_type_1961: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_959, torch.float32);  view_959 = None
        add_322: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1955, convert_element_type_1961);  convert_element_type_1955 = convert_element_type_1961 = None
        convert_element_type_1962: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_275, torch.float32);  mm_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_960: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_955, [4096, 2048]);  view_955 = None
        permute_826: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_960, [1, 0])
        mm_277: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_826, view_154);  permute_826 = view_154 = None
        mm_278: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_960, permute_828);  view_960 = permute_828 = None
        view_961: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_278, [32, 128, 2048]);  mm_278 = None
        convert_element_type_1967: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_961, torch.float32);  view_961 = None
        add_323: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_322, convert_element_type_1967);  add_322 = convert_element_type_1967 = None
        convert_element_type_1968: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_277, torch.float32);  mm_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_605: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_323, primals_102);  primals_102 = None
        mul_606: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_605, 2048)
        sum_233: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_605, [2], True)
        mul_607: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_605, mul_56);  mul_605 = None
        sum_234: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_607, [2], True);  mul_607 = None
        mul_608: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, sum_234);  sum_234 = None
        sub_198: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_606, sum_233);  mul_606 = sum_233 = None
        sub_199: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_198, mul_608);  sub_198 = mul_608 = None
        mul_609: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_60, sub_199);  div_60 = sub_199 = None
        mul_610: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_323, mul_56);  mul_56 = None
        sum_235: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_610, [0, 1]);  mul_610 = None
        sum_236: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_323, [0, 1]);  add_323 = None
        add_324: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_321, mul_609);  add_321 = mul_609 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1969: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_324, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_962: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1969, [4096, 2048]);  convert_element_type_1969 = None
        mm_279: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_962, permute_830);  permute_830 = None
        permute_831: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_962, [1, 0])
        mm_280: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_831, view_152);  permute_831 = view_152 = None
        sum_237: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_962, [0], True, dtype = torch.float32);  view_962 = None
        view_963: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_237, [2048]);  sum_237 = None
        convert_element_type_1974: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_963, torch.bfloat16);  view_963 = None
        view_964: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_279, [32, 128, 8192]);  mm_279 = None
        convert_element_type_1975: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_964, torch.float32);  view_964 = None
        convert_element_type_1976: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_280, torch.float32);  mm_280 = None
        convert_element_type_1977: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1974, torch.float32);  convert_element_type_1974 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_151: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [32, 128, 8192]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_52: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_151, 0.5)
        mul_611: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1975, mul_52);  mul_52 = None
        convert_element_type_266: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_151, torch.float32)
        pow_7: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_266, 3.0)
        mul_53: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_64: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_151, mul_53);  view_151 = mul_53 = None
        mul_54: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_64, 0.7978845608028654);  add_64 = None
        tanh_6: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_65: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_612: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1975, add_65);  convert_element_type_1975 = add_65 = None
        convert_element_type_1978: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_612, torch.bfloat16);  mul_612 = None
        mul_613: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_200: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_613);  mul_613 = None
        mul_614: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_611, sub_200);  mul_611 = sub_200 = None
        mul_615: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_614, 0.7978845608028654);  mul_614 = None
        convert_element_type_1979: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_615, torch.bfloat16)
        mul_616: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_615, 0.044715);  mul_615 = None
        pow_42: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_266, 2.0);  convert_element_type_266 = None
        mul_617: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_42, 3.0);  pow_42 = None
        mul_618: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_616, mul_617);  mul_616 = mul_617 = None
        convert_element_type_1980: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_618, torch.bfloat16);  mul_618 = None
        add_325: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1979, convert_element_type_1980);  convert_element_type_1979 = convert_element_type_1980 = None
        mul_619: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1978, 0.5);  convert_element_type_1978 = None
        add_326: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_325, mul_619);  add_325 = mul_619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_965: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_326, [4096, 8192]);  add_326 = None
        mm_281: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_965, permute_834);  permute_834 = None
        permute_835: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_965, [1, 0])
        mm_282: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_835, view_150);  permute_835 = view_150 = None
        sum_238: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_965, [0], True, dtype = torch.float32);  view_965 = None
        view_966: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_238, [8192]);  sum_238 = None
        convert_element_type_1985: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_966, torch.bfloat16);  view_966 = None
        view_967: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_281, [32, 128, 2048]);  mm_281 = None
        convert_element_type_1986: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_967, torch.float32);  view_967 = None
        convert_element_type_1987: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_282, torch.float32);  mm_282 = None
        convert_element_type_1988: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1985, torch.float32);  convert_element_type_1985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_621: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1986, primals_96);  primals_96 = None
        mul_622: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_621, 2048)
        sum_239: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_621, [2], True)
        mul_623: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_621, mul_50);  mul_621 = None
        sum_240: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_623, [2], True);  mul_623 = None
        mul_624: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, sum_240);  sum_240 = None
        sub_202: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_622, sum_239);  mul_622 = sum_239 = None
        sub_203: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_202, mul_624);  sub_202 = mul_624 = None
        mul_625: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_61, sub_203);  div_61 = sub_203 = None
        mul_626: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1986, mul_50);  mul_50 = None
        sum_241: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 1]);  mul_626 = None
        sum_242: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1986, [0, 1]);  convert_element_type_1986 = None
        add_327: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_324, mul_625);  add_324 = mul_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1989: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_327, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_968: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1989, [4096, 2048]);  convert_element_type_1989 = None
        mm_283: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_968, permute_838);  permute_838 = None
        permute_839: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_968, [1, 0])
        mm_284: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_839, view_148);  permute_839 = view_148 = None
        sum_243: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_968, [0], True, dtype = torch.float32);  view_968 = None
        view_969: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_243, [2048]);  sum_243 = None
        convert_element_type_1994: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_969, torch.bfloat16);  view_969 = None
        view_970: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_283, [32, 128, 2048]);  mm_283 = None
        convert_element_type_1995: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_284, torch.float32);  mm_284 = None
        convert_element_type_1996: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1994, torch.float32);  convert_element_type_1994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_971: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_970, [32, 128, 16, 128]);  view_970 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_842: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_971, [0, 2, 1, 3]);  view_971 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_238: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_842, memory_format = torch.contiguous_format);  permute_842 = None
        view_972: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_238, [512, 128, 128]);  clone_238 = None
        bmm_116: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_843, view_972);  permute_843 = None
        bmm_117: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_972, permute_844);  view_972 = permute_844 = None
        view_973: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_116, [32, 16, 128, 128]);  bmm_116 = None
        view_974: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_117, [32, 16, 128, 128]);  bmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2001: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_974, torch.float32);  view_974 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_627: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2001, div_6);  convert_element_type_2001 = None
        sum_244: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_627, [-1], True)
        neg_18: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_6);  div_6 = None
        fma_17: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_18, sum_244, mul_627);  neg_18 = sum_244 = mul_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2002: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_17, torch.bfloat16);  fma_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_16: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_93, 2, 0, 128);  primals_93 = None
        slice_17: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_16, 3, 0, 128);  slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_46: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_17, convert_element_type_2002, full_default_31);  slice_17 = convert_element_type_2002 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_975: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_46, [512, 128, 128]);  where_46 = None
        bmm_118: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_845, view_975);  permute_845 = None
        bmm_119: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_975, permute_846);  view_975 = permute_846 = None
        view_976: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_118, [32, 16, 128, 128]);  bmm_118 = None
        view_977: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_119, [32, 16, 128, 128]);  bmm_119 = None
        convert_element_type_2008: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_976, torch.float32);  view_976 = None
        permute_847: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2008, [0, 1, 3, 2]);  convert_element_type_2008 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2009: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_847, torch.bfloat16);  permute_847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_6: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_977, torch.bfloat16);  view_977 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_848: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_973, [0, 2, 1, 3]);  view_973 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_239: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_848, memory_format = torch.contiguous_format);  permute_848 = None
        view_978: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_239, [32, 128, 2048]);  clone_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_849: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2009, [0, 2, 1, 3]);  convert_element_type_2009 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_979: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_849, [32, 128, 2048]);  permute_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_850: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_6, [0, 2, 1, 3]);  convert_element_type_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_240: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_850, memory_format = torch.contiguous_format);  permute_850 = None
        view_980: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_240, [32, 128, 2048]);  clone_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_981: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_978, [4096, 2048]);  view_978 = None
        permute_851: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_981, [1, 0])
        mm_285: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_851, view_132);  permute_851 = None
        mm_286: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_981, permute_853);  view_981 = permute_853 = None
        view_982: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_286, [32, 128, 2048]);  mm_286 = None
        convert_element_type_2015: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_982, torch.float32);  view_982 = None
        convert_element_type_2016: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_285, torch.float32);  mm_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_241: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_979, memory_format = torch.contiguous_format);  view_979 = None
        view_983: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_241, [4096, 2048]);  clone_241 = None
        permute_855: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_983, [1, 0])
        mm_287: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_855, view_132);  permute_855 = None
        mm_288: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_983, permute_857);  view_983 = permute_857 = None
        view_984: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_288, [32, 128, 2048]);  mm_288 = None
        convert_element_type_2021: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_984, torch.float32);  view_984 = None
        add_328: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2015, convert_element_type_2021);  convert_element_type_2015 = convert_element_type_2021 = None
        convert_element_type_2022: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_287, torch.float32);  mm_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_985: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_980, [4096, 2048]);  view_980 = None
        permute_859: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_985, [1, 0])
        mm_289: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_859, view_132);  permute_859 = view_132 = None
        mm_290: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_985, permute_861);  view_985 = permute_861 = None
        view_986: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_290, [32, 128, 2048]);  mm_290 = None
        convert_element_type_2027: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_986, torch.float32);  view_986 = None
        add_329: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_328, convert_element_type_2027);  add_328 = convert_element_type_2027 = None
        convert_element_type_2028: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_289, torch.float32);  mm_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_629: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_329, primals_88);  primals_88 = None
        mul_630: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_629, 2048)
        sum_245: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_629, [2], True)
        mul_631: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_629, mul_48);  mul_629 = None
        sum_246: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_631, [2], True);  mul_631 = None
        mul_632: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, sum_246);  sum_246 = None
        sub_205: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_630, sum_245);  mul_630 = sum_245 = None
        sub_206: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_205, mul_632);  sub_205 = mul_632 = None
        mul_633: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_62, sub_206);  div_62 = sub_206 = None
        mul_634: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_329, mul_48);  mul_48 = None
        sum_247: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_634, [0, 1]);  mul_634 = None
        sum_248: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_329, [0, 1]);  add_329 = None
        add_330: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_327, mul_633);  add_327 = mul_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_2029: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_330, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_987: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2029, [4096, 2048]);  convert_element_type_2029 = None
        mm_291: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_987, permute_863);  permute_863 = None
        permute_864: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_987, [1, 0])
        mm_292: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_864, view_130);  permute_864 = view_130 = None
        sum_249: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_987, [0], True, dtype = torch.float32);  view_987 = None
        view_988: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_249, [2048]);  sum_249 = None
        convert_element_type_2034: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_988, torch.bfloat16);  view_988 = None
        view_989: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_291, [32, 128, 8192]);  mm_291 = None
        convert_element_type_2035: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_989, torch.float32);  view_989 = None
        convert_element_type_2036: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_292, torch.float32);  mm_292 = None
        convert_element_type_2037: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2034, torch.float32);  convert_element_type_2034 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_129: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [32, 128, 8192]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_635: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2035, mul_44);  mul_44 = None
        convert_element_type_227: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_129, torch.float32)
        pow_6: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_227, 3.0)
        mul_45: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_55: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_129, mul_45);  view_129 = mul_45 = None
        mul_46: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_5: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_56: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_636: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2035, add_56);  convert_element_type_2035 = add_56 = None
        convert_element_type_2038: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_636, torch.bfloat16);  mul_636 = None
        mul_637: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_207: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_637);  mul_637 = None
        mul_638: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_635, sub_207);  mul_635 = sub_207 = None
        mul_639: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_638, 0.7978845608028654);  mul_638 = None
        convert_element_type_2039: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_639, torch.bfloat16)
        mul_640: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_639, 0.044715);  mul_639 = None
        pow_43: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_227, 2.0);  convert_element_type_227 = None
        mul_641: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_43, 3.0);  pow_43 = None
        mul_642: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_640, mul_641);  mul_640 = mul_641 = None
        convert_element_type_2040: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_642, torch.bfloat16);  mul_642 = None
        add_331: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2039, convert_element_type_2040);  convert_element_type_2039 = convert_element_type_2040 = None
        mul_643: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2038, 0.5);  convert_element_type_2038 = None
        add_332: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_331, mul_643);  add_331 = mul_643 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_990: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_332, [4096, 8192]);  add_332 = None
        mm_293: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_990, permute_867);  permute_867 = None
        permute_868: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_990, [1, 0])
        mm_294: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_868, view_128);  permute_868 = view_128 = None
        sum_250: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_990, [0], True, dtype = torch.float32);  view_990 = None
        view_991: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_250, [8192]);  sum_250 = None
        convert_element_type_2045: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_991, torch.bfloat16);  view_991 = None
        view_992: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_293, [32, 128, 2048]);  mm_293 = None
        convert_element_type_2046: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_992, torch.float32);  view_992 = None
        convert_element_type_2047: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_294, torch.float32);  mm_294 = None
        convert_element_type_2048: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2045, torch.float32);  convert_element_type_2045 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_645: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2046, primals_82);  primals_82 = None
        mul_646: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_645, 2048)
        sum_251: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_645, [2], True)
        mul_647: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_645, mul_42);  mul_645 = None
        sum_252: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_647, [2], True);  mul_647 = None
        mul_648: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, sum_252);  sum_252 = None
        sub_209: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_646, sum_251);  mul_646 = sum_251 = None
        sub_210: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_209, mul_648);  sub_209 = mul_648 = None
        mul_649: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_63, sub_210);  div_63 = sub_210 = None
        mul_650: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2046, mul_42);  mul_42 = None
        sum_253: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_650, [0, 1]);  mul_650 = None
        sum_254: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2046, [0, 1]);  convert_element_type_2046 = None
        add_333: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_330, mul_649);  add_330 = mul_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_2049: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_333, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_993: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2049, [4096, 2048]);  convert_element_type_2049 = None
        mm_295: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_993, permute_871);  permute_871 = None
        permute_872: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_993, [1, 0])
        mm_296: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_872, view_126);  permute_872 = view_126 = None
        sum_255: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_993, [0], True, dtype = torch.float32);  view_993 = None
        view_994: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_255, [2048]);  sum_255 = None
        convert_element_type_2054: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_994, torch.bfloat16);  view_994 = None
        view_995: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_295, [32, 128, 2048]);  mm_295 = None
        convert_element_type_2055: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_296, torch.float32);  mm_296 = None
        convert_element_type_2056: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2054, torch.float32);  convert_element_type_2054 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_996: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_995, [32, 128, 16, 128]);  view_995 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_875: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_996, [0, 2, 1, 3]);  view_996 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_242: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_875, memory_format = torch.contiguous_format);  permute_875 = None
        view_997: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_242, [512, 128, 128]);  clone_242 = None
        bmm_120: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_876, view_997);  permute_876 = None
        bmm_121: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_997, permute_877);  view_997 = permute_877 = None
        view_998: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_120, [32, 16, 128, 128]);  bmm_120 = None
        view_999: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_121, [32, 16, 128, 128]);  bmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2061: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_999, torch.float32);  view_999 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_651: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2061, div_5);  convert_element_type_2061 = None
        sum_256: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_651, [-1], True)
        neg_19: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_18: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_19, sum_256, mul_651);  neg_19 = sum_256 = mul_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2062: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_18, torch.bfloat16);  fma_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_14: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_79, 2, 0, 128);  primals_79 = None
        slice_15: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_14, 3, 0, 128);  slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_47: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_15, convert_element_type_2062, full_default_31);  slice_15 = convert_element_type_2062 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1000: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_47, [512, 128, 128]);  where_47 = None
        bmm_122: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_878, view_1000);  permute_878 = None
        bmm_123: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1000, permute_879);  view_1000 = permute_879 = None
        view_1001: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_122, [32, 16, 128, 128]);  bmm_122 = None
        view_1002: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_123, [32, 16, 128, 128]);  bmm_123 = None
        convert_element_type_2068: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1001, torch.float32);  view_1001 = None
        permute_880: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2068, [0, 1, 3, 2]);  convert_element_type_2068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2069: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_880, torch.bfloat16);  permute_880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_5: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1002, torch.bfloat16);  view_1002 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_881: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_998, [0, 2, 1, 3]);  view_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_243: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_881, memory_format = torch.contiguous_format);  permute_881 = None
        view_1003: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_243, [32, 128, 2048]);  clone_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_882: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2069, [0, 2, 1, 3]);  convert_element_type_2069 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1004: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_882, [32, 128, 2048]);  permute_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_883: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_5, [0, 2, 1, 3]);  convert_element_type_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_244: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_883, memory_format = torch.contiguous_format);  permute_883 = None
        view_1005: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_244, [32, 128, 2048]);  clone_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1006: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1003, [4096, 2048]);  view_1003 = None
        permute_884: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1006, [1, 0])
        mm_297: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_884, view_110);  permute_884 = None
        mm_298: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1006, permute_886);  view_1006 = permute_886 = None
        view_1007: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_298, [32, 128, 2048]);  mm_298 = None
        convert_element_type_2075: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1007, torch.float32);  view_1007 = None
        convert_element_type_2076: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_297, torch.float32);  mm_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_245: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_1004, memory_format = torch.contiguous_format);  view_1004 = None
        view_1008: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_245, [4096, 2048]);  clone_245 = None
        permute_888: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1008, [1, 0])
        mm_299: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_888, view_110);  permute_888 = None
        mm_300: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1008, permute_890);  view_1008 = permute_890 = None
        view_1009: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_300, [32, 128, 2048]);  mm_300 = None
        convert_element_type_2081: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1009, torch.float32);  view_1009 = None
        add_334: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2075, convert_element_type_2081);  convert_element_type_2075 = convert_element_type_2081 = None
        convert_element_type_2082: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_299, torch.float32);  mm_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1010: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1005, [4096, 2048]);  view_1005 = None
        permute_892: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1010, [1, 0])
        mm_301: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_892, view_110);  permute_892 = view_110 = None
        mm_302: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1010, permute_894);  view_1010 = permute_894 = None
        view_1011: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_302, [32, 128, 2048]);  mm_302 = None
        convert_element_type_2087: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1011, torch.float32);  view_1011 = None
        add_335: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_334, convert_element_type_2087);  add_334 = convert_element_type_2087 = None
        convert_element_type_2088: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_301, torch.float32);  mm_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_653: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_335, primals_74);  primals_74 = None
        mul_654: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_653, 2048)
        sum_257: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_653, [2], True)
        mul_655: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_653, mul_40);  mul_653 = None
        sum_258: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_655, [2], True);  mul_655 = None
        mul_656: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, sum_258);  sum_258 = None
        sub_212: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_654, sum_257);  mul_654 = sum_257 = None
        sub_213: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_212, mul_656);  sub_212 = mul_656 = None
        mul_657: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_64, sub_213);  div_64 = sub_213 = None
        mul_658: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_335, mul_40);  mul_40 = None
        sum_259: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_658, [0, 1]);  mul_658 = None
        sum_260: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_335, [0, 1]);  add_335 = None
        add_336: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_333, mul_657);  add_333 = mul_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_2089: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_336, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_1012: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2089, [4096, 2048]);  convert_element_type_2089 = None
        mm_303: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_1012, permute_896);  permute_896 = None
        permute_897: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1012, [1, 0])
        mm_304: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_897, view_108);  permute_897 = view_108 = None
        sum_261: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1012, [0], True, dtype = torch.float32);  view_1012 = None
        view_1013: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_261, [2048]);  sum_261 = None
        convert_element_type_2094: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1013, torch.bfloat16);  view_1013 = None
        view_1014: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_303, [32, 128, 8192]);  mm_303 = None
        convert_element_type_2095: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1014, torch.float32);  view_1014 = None
        convert_element_type_2096: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_304, torch.float32);  mm_304 = None
        convert_element_type_2097: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2094, torch.float32);  convert_element_type_2094 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_107: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [32, 128, 8192]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        mul_659: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2095, mul_36);  mul_36 = None
        convert_element_type_188: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.float32)
        pow_5: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_188, 3.0)
        mul_37: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_46: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_107, mul_37);  view_107 = mul_37 = None
        mul_38: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_4: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_47: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_660: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2095, add_47);  convert_element_type_2095 = add_47 = None
        convert_element_type_2098: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_660, torch.bfloat16);  mul_660 = None
        mul_661: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_214: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_661);  mul_661 = None
        mul_662: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_659, sub_214);  mul_659 = sub_214 = None
        mul_663: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_662, 0.7978845608028654);  mul_662 = None
        convert_element_type_2099: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_663, torch.bfloat16)
        mul_664: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_663, 0.044715);  mul_663 = None
        pow_44: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_188, 2.0);  convert_element_type_188 = None
        mul_665: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_44, 3.0);  pow_44 = None
        mul_666: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_664, mul_665);  mul_664 = mul_665 = None
        convert_element_type_2100: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_666, torch.bfloat16);  mul_666 = None
        add_337: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2099, convert_element_type_2100);  convert_element_type_2099 = convert_element_type_2100 = None
        mul_667: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2098, 0.5);  convert_element_type_2098 = None
        add_338: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_337, mul_667);  add_337 = mul_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_1015: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_338, [4096, 8192]);  add_338 = None
        mm_305: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1015, permute_900);  permute_900 = None
        permute_901: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_1015, [1, 0])
        mm_306: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_901, view_106);  permute_901 = view_106 = None
        sum_262: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1015, [0], True, dtype = torch.float32);  view_1015 = None
        view_1016: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_262, [8192]);  sum_262 = None
        convert_element_type_2105: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1016, torch.bfloat16);  view_1016 = None
        view_1017: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_305, [32, 128, 2048]);  mm_305 = None
        convert_element_type_2106: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1017, torch.float32);  view_1017 = None
        convert_element_type_2107: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_306, torch.float32);  mm_306 = None
        convert_element_type_2108: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2105, torch.float32);  convert_element_type_2105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_669: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2106, primals_68);  primals_68 = None
        mul_670: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_669, 2048)
        sum_263: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_669, [2], True)
        mul_671: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_669, mul_34);  mul_669 = None
        sum_264: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_671, [2], True);  mul_671 = None
        mul_672: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, sum_264);  sum_264 = None
        sub_216: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_670, sum_263);  mul_670 = sum_263 = None
        sub_217: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_216, mul_672);  sub_216 = mul_672 = None
        mul_673: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_65, sub_217);  div_65 = sub_217 = None
        mul_674: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2106, mul_34);  mul_34 = None
        sum_265: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_674, [0, 1]);  mul_674 = None
        sum_266: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2106, [0, 1]);  convert_element_type_2106 = None
        add_339: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_336, mul_673);  add_336 = mul_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_2109: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_339, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_1018: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2109, [4096, 2048]);  convert_element_type_2109 = None
        mm_307: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1018, permute_904);  permute_904 = None
        permute_905: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1018, [1, 0])
        mm_308: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_905, view_104);  permute_905 = view_104 = None
        sum_267: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1018, [0], True, dtype = torch.float32);  view_1018 = None
        view_1019: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_267, [2048]);  sum_267 = None
        convert_element_type_2114: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1019, torch.bfloat16);  view_1019 = None
        view_1020: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_307, [32, 128, 2048]);  mm_307 = None
        convert_element_type_2115: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_308, torch.float32);  mm_308 = None
        convert_element_type_2116: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2114, torch.float32);  convert_element_type_2114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_1021: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1020, [32, 128, 16, 128]);  view_1020 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_908: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_1021, [0, 2, 1, 3]);  view_1021 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_246: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_908, memory_format = torch.contiguous_format);  permute_908 = None
        view_1022: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_246, [512, 128, 128]);  clone_246 = None
        bmm_124: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_909, view_1022);  permute_909 = None
        bmm_125: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1022, permute_910);  view_1022 = permute_910 = None
        view_1023: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_124, [32, 16, 128, 128]);  bmm_124 = None
        view_1024: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_125, [32, 16, 128, 128]);  bmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2121: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1024, torch.float32);  view_1024 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_675: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2121, div_4);  convert_element_type_2121 = None
        sum_268: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_675, [-1], True)
        neg_20: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_19: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_20, sum_268, mul_675);  neg_20 = sum_268 = mul_675 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2122: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_19, torch.bfloat16);  fma_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_12: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_65, 2, 0, 128);  primals_65 = None
        slice_13: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_12, 3, 0, 128);  slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_48: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_13, convert_element_type_2122, full_default_31);  slice_13 = convert_element_type_2122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1025: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_48, [512, 128, 128]);  where_48 = None
        bmm_126: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_911, view_1025);  permute_911 = None
        bmm_127: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1025, permute_912);  view_1025 = permute_912 = None
        view_1026: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_126, [32, 16, 128, 128]);  bmm_126 = None
        view_1027: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_127, [32, 16, 128, 128]);  bmm_127 = None
        convert_element_type_2128: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1026, torch.float32);  view_1026 = None
        permute_913: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2128, [0, 1, 3, 2]);  convert_element_type_2128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2129: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_913, torch.bfloat16);  permute_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_4: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1027, torch.bfloat16);  view_1027 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_914: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_1023, [0, 2, 1, 3]);  view_1023 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_247: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_914, memory_format = torch.contiguous_format);  permute_914 = None
        view_1028: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_247, [32, 128, 2048]);  clone_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_915: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2129, [0, 2, 1, 3]);  convert_element_type_2129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1029: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_915, [32, 128, 2048]);  permute_915 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_916: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_4, [0, 2, 1, 3]);  convert_element_type_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_248: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_916, memory_format = torch.contiguous_format);  permute_916 = None
        view_1030: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_248, [32, 128, 2048]);  clone_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1031: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1028, [4096, 2048]);  view_1028 = None
        permute_917: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1031, [1, 0])
        mm_309: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_917, view_88);  permute_917 = None
        mm_310: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1031, permute_919);  view_1031 = permute_919 = None
        view_1032: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_310, [32, 128, 2048]);  mm_310 = None
        convert_element_type_2135: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1032, torch.float32);  view_1032 = None
        convert_element_type_2136: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_309, torch.float32);  mm_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_249: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_1029, memory_format = torch.contiguous_format);  view_1029 = None
        view_1033: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_249, [4096, 2048]);  clone_249 = None
        permute_921: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1033, [1, 0])
        mm_311: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_921, view_88);  permute_921 = None
        mm_312: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1033, permute_923);  view_1033 = permute_923 = None
        view_1034: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_312, [32, 128, 2048]);  mm_312 = None
        convert_element_type_2141: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1034, torch.float32);  view_1034 = None
        add_340: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2135, convert_element_type_2141);  convert_element_type_2135 = convert_element_type_2141 = None
        convert_element_type_2142: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_311, torch.float32);  mm_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1035: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1030, [4096, 2048]);  view_1030 = None
        permute_925: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1035, [1, 0])
        mm_313: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_925, view_88);  permute_925 = view_88 = None
        mm_314: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1035, permute_927);  view_1035 = permute_927 = None
        view_1036: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_314, [32, 128, 2048]);  mm_314 = None
        convert_element_type_2147: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1036, torch.float32);  view_1036 = None
        add_341: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_340, convert_element_type_2147);  add_340 = convert_element_type_2147 = None
        convert_element_type_2148: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_313, torch.float32);  mm_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_677: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_341, primals_60);  primals_60 = None
        mul_678: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_677, 2048)
        sum_269: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_677, [2], True)
        mul_679: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_677, mul_32);  mul_677 = None
        sum_270: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_679, [2], True);  mul_679 = None
        mul_680: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, sum_270);  sum_270 = None
        sub_219: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_678, sum_269);  mul_678 = sum_269 = None
        sub_220: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_219, mul_680);  sub_219 = mul_680 = None
        mul_681: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_66, sub_220);  div_66 = sub_220 = None
        mul_682: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_341, mul_32);  mul_32 = None
        sum_271: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_682, [0, 1]);  mul_682 = None
        sum_272: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_341, [0, 1]);  add_341 = None
        add_342: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_339, mul_681);  add_339 = mul_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_2149: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_342, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_1037: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2149, [4096, 2048]);  convert_element_type_2149 = None
        mm_315: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_1037, permute_929);  permute_929 = None
        permute_930: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1037, [1, 0])
        mm_316: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_930, view_86);  permute_930 = view_86 = None
        sum_273: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1037, [0], True, dtype = torch.float32);  view_1037 = None
        view_1038: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_273, [2048]);  sum_273 = None
        convert_element_type_2154: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1038, torch.bfloat16);  view_1038 = None
        view_1039: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_315, [32, 128, 8192]);  mm_315 = None
        convert_element_type_2155: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1039, torch.float32);  view_1039 = None
        convert_element_type_2156: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_316, torch.float32);  mm_316 = None
        convert_element_type_2157: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2154, torch.float32);  convert_element_type_2154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_85: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [32, 128, 8192]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_85, 0.5)
        mul_683: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2155, mul_28);  mul_28 = None
        convert_element_type_149: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_85, torch.float32)
        pow_4: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_149, 3.0)
        mul_29: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_37: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_85, mul_29);  view_85 = mul_29 = None
        mul_30: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_37, 0.7978845608028654);  add_37 = None
        tanh_3: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_38: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_684: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2155, add_38);  convert_element_type_2155 = add_38 = None
        convert_element_type_2158: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_684, torch.bfloat16);  mul_684 = None
        mul_685: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_221: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_685);  mul_685 = None
        mul_686: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_683, sub_221);  mul_683 = sub_221 = None
        mul_687: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_686, 0.7978845608028654);  mul_686 = None
        convert_element_type_2159: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_687, torch.bfloat16)
        mul_688: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_687, 0.044715);  mul_687 = None
        pow_45: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_149, 2.0);  convert_element_type_149 = None
        mul_689: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_45, 3.0);  pow_45 = None
        mul_690: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_688, mul_689);  mul_688 = mul_689 = None
        convert_element_type_2160: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_690, torch.bfloat16);  mul_690 = None
        add_343: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2159, convert_element_type_2160);  convert_element_type_2159 = convert_element_type_2160 = None
        mul_691: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2158, 0.5);  convert_element_type_2158 = None
        add_344: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_343, mul_691);  add_343 = mul_691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_1040: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_344, [4096, 8192]);  add_344 = None
        mm_317: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1040, permute_933);  permute_933 = None
        permute_934: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_1040, [1, 0])
        mm_318: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_934, view_84);  permute_934 = view_84 = None
        sum_274: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1040, [0], True, dtype = torch.float32);  view_1040 = None
        view_1041: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_274, [8192]);  sum_274 = None
        convert_element_type_2165: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1041, torch.bfloat16);  view_1041 = None
        view_1042: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_317, [32, 128, 2048]);  mm_317 = None
        convert_element_type_2166: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1042, torch.float32);  view_1042 = None
        convert_element_type_2167: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_318, torch.float32);  mm_318 = None
        convert_element_type_2168: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2165, torch.float32);  convert_element_type_2165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_693: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2166, primals_54);  primals_54 = None
        mul_694: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_693, 2048)
        sum_275: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_693, [2], True)
        mul_695: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_693, mul_26);  mul_693 = None
        sum_276: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_695, [2], True);  mul_695 = None
        mul_696: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, sum_276);  sum_276 = None
        sub_223: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_694, sum_275);  mul_694 = sum_275 = None
        sub_224: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_223, mul_696);  sub_223 = mul_696 = None
        mul_697: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_67, sub_224);  div_67 = sub_224 = None
        mul_698: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2166, mul_26);  mul_26 = None
        sum_277: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_698, [0, 1]);  mul_698 = None
        sum_278: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2166, [0, 1]);  convert_element_type_2166 = None
        add_345: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_342, mul_697);  add_342 = mul_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_2169: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_345, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_1043: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2169, [4096, 2048]);  convert_element_type_2169 = None
        mm_319: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1043, permute_937);  permute_937 = None
        permute_938: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1043, [1, 0])
        mm_320: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_938, view_82);  permute_938 = view_82 = None
        sum_279: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1043, [0], True, dtype = torch.float32);  view_1043 = None
        view_1044: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_279, [2048]);  sum_279 = None
        convert_element_type_2174: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1044, torch.bfloat16);  view_1044 = None
        view_1045: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_319, [32, 128, 2048]);  mm_319 = None
        convert_element_type_2175: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_320, torch.float32);  mm_320 = None
        convert_element_type_2176: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2174, torch.float32);  convert_element_type_2174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_1046: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1045, [32, 128, 16, 128]);  view_1045 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_941: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_1046, [0, 2, 1, 3]);  view_1046 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_250: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_941, memory_format = torch.contiguous_format);  permute_941 = None
        view_1047: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_250, [512, 128, 128]);  clone_250 = None
        bmm_128: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_942, view_1047);  permute_942 = None
        bmm_129: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1047, permute_943);  view_1047 = permute_943 = None
        view_1048: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_128, [32, 16, 128, 128]);  bmm_128 = None
        view_1049: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_129, [32, 16, 128, 128]);  bmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2181: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1049, torch.float32);  view_1049 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_699: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2181, div_3);  convert_element_type_2181 = None
        sum_280: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_699, [-1], True)
        neg_21: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_20: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_21, sum_280, mul_699);  neg_21 = sum_280 = mul_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2182: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_20, torch.bfloat16);  fma_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_10: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_51, 2, 0, 128);  primals_51 = None
        slice_11: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_10, 3, 0, 128);  slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_49: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_11, convert_element_type_2182, full_default_31);  slice_11 = convert_element_type_2182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1050: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_49, [512, 128, 128]);  where_49 = None
        bmm_130: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_944, view_1050);  permute_944 = None
        bmm_131: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1050, permute_945);  view_1050 = permute_945 = None
        view_1051: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_130, [32, 16, 128, 128]);  bmm_130 = None
        view_1052: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_131, [32, 16, 128, 128]);  bmm_131 = None
        convert_element_type_2188: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1051, torch.float32);  view_1051 = None
        permute_946: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2188, [0, 1, 3, 2]);  convert_element_type_2188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2189: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_946, torch.bfloat16);  permute_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_3: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1052, torch.bfloat16);  view_1052 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_947: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_1048, [0, 2, 1, 3]);  view_1048 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_251: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_947, memory_format = torch.contiguous_format);  permute_947 = None
        view_1053: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_251, [32, 128, 2048]);  clone_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_948: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2189, [0, 2, 1, 3]);  convert_element_type_2189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1054: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_948, [32, 128, 2048]);  permute_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_949: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_3, [0, 2, 1, 3]);  convert_element_type_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_252: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_949, memory_format = torch.contiguous_format);  permute_949 = None
        view_1055: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_252, [32, 128, 2048]);  clone_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1056: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1053, [4096, 2048]);  view_1053 = None
        permute_950: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1056, [1, 0])
        mm_321: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_950, view_66);  permute_950 = None
        mm_322: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1056, permute_952);  view_1056 = permute_952 = None
        view_1057: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_322, [32, 128, 2048]);  mm_322 = None
        convert_element_type_2195: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1057, torch.float32);  view_1057 = None
        convert_element_type_2196: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_321, torch.float32);  mm_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_253: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_1054, memory_format = torch.contiguous_format);  view_1054 = None
        view_1058: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_253, [4096, 2048]);  clone_253 = None
        permute_954: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1058, [1, 0])
        mm_323: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_954, view_66);  permute_954 = None
        mm_324: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1058, permute_956);  view_1058 = permute_956 = None
        view_1059: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_324, [32, 128, 2048]);  mm_324 = None
        convert_element_type_2201: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1059, torch.float32);  view_1059 = None
        add_346: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2195, convert_element_type_2201);  convert_element_type_2195 = convert_element_type_2201 = None
        convert_element_type_2202: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_323, torch.float32);  mm_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1060: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1055, [4096, 2048]);  view_1055 = None
        permute_958: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1060, [1, 0])
        mm_325: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_958, view_66);  permute_958 = view_66 = None
        mm_326: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1060, permute_960);  view_1060 = permute_960 = None
        view_1061: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_326, [32, 128, 2048]);  mm_326 = None
        convert_element_type_2207: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1061, torch.float32);  view_1061 = None
        add_347: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_346, convert_element_type_2207);  add_346 = convert_element_type_2207 = None
        convert_element_type_2208: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_325, torch.float32);  mm_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_701: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_347, primals_46);  primals_46 = None
        mul_702: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_701, 2048)
        sum_281: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_701, [2], True)
        mul_703: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_701, mul_24);  mul_701 = None
        sum_282: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_703, [2], True);  mul_703 = None
        mul_704: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, sum_282);  sum_282 = None
        sub_226: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_702, sum_281);  mul_702 = sum_281 = None
        sub_227: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_226, mul_704);  sub_226 = mul_704 = None
        mul_705: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_68, sub_227);  div_68 = sub_227 = None
        mul_706: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_347, mul_24);  mul_24 = None
        sum_283: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_706, [0, 1]);  mul_706 = None
        sum_284: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_347, [0, 1]);  add_347 = None
        add_348: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_345, mul_705);  add_345 = mul_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_2209: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_348, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_1062: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2209, [4096, 2048]);  convert_element_type_2209 = None
        mm_327: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_1062, permute_962);  permute_962 = None
        permute_963: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1062, [1, 0])
        mm_328: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_963, view_64);  permute_963 = view_64 = None
        sum_285: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1062, [0], True, dtype = torch.float32);  view_1062 = None
        view_1063: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_285, [2048]);  sum_285 = None
        convert_element_type_2214: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1063, torch.bfloat16);  view_1063 = None
        view_1064: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_327, [32, 128, 8192]);  mm_327 = None
        convert_element_type_2215: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1064, torch.float32);  view_1064 = None
        convert_element_type_2216: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_328, torch.float32);  mm_328 = None
        convert_element_type_2217: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2214, torch.float32);  convert_element_type_2214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_63: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [32, 128, 8192]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_707: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2215, mul_20);  mul_20 = None
        convert_element_type_110: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32)
        pow_3: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_110, 3.0)
        mul_21: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_28: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_63, mul_21);  view_63 = mul_21 = None
        mul_22: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_28, 0.7978845608028654);  add_28 = None
        tanh_2: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_29: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_708: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2215, add_29);  convert_element_type_2215 = add_29 = None
        convert_element_type_2218: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_708, torch.bfloat16);  mul_708 = None
        mul_709: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_228: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_709);  mul_709 = None
        mul_710: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_707, sub_228);  mul_707 = sub_228 = None
        mul_711: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_710, 0.7978845608028654);  mul_710 = None
        convert_element_type_2219: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_711, torch.bfloat16)
        mul_712: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_711, 0.044715);  mul_711 = None
        pow_46: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_110, 2.0);  convert_element_type_110 = None
        mul_713: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_46, 3.0);  pow_46 = None
        mul_714: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_712, mul_713);  mul_712 = mul_713 = None
        convert_element_type_2220: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_714, torch.bfloat16);  mul_714 = None
        add_349: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2219, convert_element_type_2220);  convert_element_type_2219 = convert_element_type_2220 = None
        mul_715: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2218, 0.5);  convert_element_type_2218 = None
        add_350: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_349, mul_715);  add_349 = mul_715 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_1065: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_350, [4096, 8192]);  add_350 = None
        mm_329: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1065, permute_966);  permute_966 = None
        permute_967: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_1065, [1, 0])
        mm_330: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_967, view_62);  permute_967 = view_62 = None
        sum_286: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1065, [0], True, dtype = torch.float32);  view_1065 = None
        view_1066: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_286, [8192]);  sum_286 = None
        convert_element_type_2225: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1066, torch.bfloat16);  view_1066 = None
        view_1067: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_329, [32, 128, 2048]);  mm_329 = None
        convert_element_type_2226: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1067, torch.float32);  view_1067 = None
        convert_element_type_2227: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_330, torch.float32);  mm_330 = None
        convert_element_type_2228: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2225, torch.float32);  convert_element_type_2225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_717: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2226, primals_40);  primals_40 = None
        mul_718: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_717, 2048)
        sum_287: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_717, [2], True)
        mul_719: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_717, mul_18);  mul_717 = None
        sum_288: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_719, [2], True);  mul_719 = None
        mul_720: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, sum_288);  sum_288 = None
        sub_230: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_718, sum_287);  mul_718 = sum_287 = None
        sub_231: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_230, mul_720);  sub_230 = mul_720 = None
        mul_721: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_69, sub_231);  div_69 = sub_231 = None
        mul_722: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2226, mul_18);  mul_18 = None
        sum_289: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_722, [0, 1]);  mul_722 = None
        sum_290: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2226, [0, 1]);  convert_element_type_2226 = None
        add_351: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_348, mul_721);  add_348 = mul_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_2229: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_351, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_1068: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2229, [4096, 2048]);  convert_element_type_2229 = None
        mm_331: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1068, permute_970);  permute_970 = None
        permute_971: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1068, [1, 0])
        mm_332: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_971, view_60);  permute_971 = view_60 = None
        sum_291: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1068, [0], True, dtype = torch.float32);  view_1068 = None
        view_1069: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_291, [2048]);  sum_291 = None
        convert_element_type_2234: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1069, torch.bfloat16);  view_1069 = None
        view_1070: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_331, [32, 128, 2048]);  mm_331 = None
        convert_element_type_2235: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_332, torch.float32);  mm_332 = None
        convert_element_type_2236: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2234, torch.float32);  convert_element_type_2234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_1071: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1070, [32, 128, 16, 128]);  view_1070 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_974: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_1071, [0, 2, 1, 3]);  view_1071 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_254: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_974, memory_format = torch.contiguous_format);  permute_974 = None
        view_1072: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_254, [512, 128, 128]);  clone_254 = None
        bmm_132: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_975, view_1072);  permute_975 = None
        bmm_133: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1072, permute_976);  view_1072 = permute_976 = None
        view_1073: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_132, [32, 16, 128, 128]);  bmm_132 = None
        view_1074: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_133, [32, 16, 128, 128]);  bmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2241: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1074, torch.float32);  view_1074 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_723: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2241, div_2);  convert_element_type_2241 = None
        sum_292: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_723, [-1], True)
        neg_22: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_2);  div_2 = None
        fma_21: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_22, sum_292, mul_723);  neg_22 = sum_292 = mul_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2242: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_21, torch.bfloat16);  fma_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_8: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_37, 2, 0, 128);  primals_37 = None
        slice_9: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_8, 3, 0, 128);  slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_50: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_9, convert_element_type_2242, full_default_31);  slice_9 = convert_element_type_2242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1075: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_50, [512, 128, 128]);  where_50 = None
        bmm_134: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_977, view_1075);  permute_977 = None
        bmm_135: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1075, permute_978);  view_1075 = permute_978 = None
        view_1076: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_134, [32, 16, 128, 128]);  bmm_134 = None
        view_1077: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_135, [32, 16, 128, 128]);  bmm_135 = None
        convert_element_type_2248: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1076, torch.float32);  view_1076 = None
        permute_979: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2248, [0, 1, 3, 2]);  convert_element_type_2248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2249: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_979, torch.bfloat16);  permute_979 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_2: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1077, torch.bfloat16);  view_1077 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_980: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_1073, [0, 2, 1, 3]);  view_1073 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_255: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_980, memory_format = torch.contiguous_format);  permute_980 = None
        view_1078: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_255, [32, 128, 2048]);  clone_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_981: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2249, [0, 2, 1, 3]);  convert_element_type_2249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1079: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_981, [32, 128, 2048]);  permute_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_982: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_2, [0, 2, 1, 3]);  convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_256: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_982, memory_format = torch.contiguous_format);  permute_982 = None
        view_1080: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_256, [32, 128, 2048]);  clone_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1081: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1078, [4096, 2048]);  view_1078 = None
        permute_983: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1081, [1, 0])
        mm_333: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_983, view_44);  permute_983 = None
        mm_334: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1081, permute_985);  view_1081 = permute_985 = None
        view_1082: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_334, [32, 128, 2048]);  mm_334 = None
        convert_element_type_2255: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1082, torch.float32);  view_1082 = None
        convert_element_type_2256: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_333, torch.float32);  mm_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_257: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_1079, memory_format = torch.contiguous_format);  view_1079 = None
        view_1083: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_257, [4096, 2048]);  clone_257 = None
        permute_987: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1083, [1, 0])
        mm_335: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_987, view_44);  permute_987 = None
        mm_336: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1083, permute_989);  view_1083 = permute_989 = None
        view_1084: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_336, [32, 128, 2048]);  mm_336 = None
        convert_element_type_2261: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1084, torch.float32);  view_1084 = None
        add_352: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2255, convert_element_type_2261);  convert_element_type_2255 = convert_element_type_2261 = None
        convert_element_type_2262: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_335, torch.float32);  mm_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1085: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1080, [4096, 2048]);  view_1080 = None
        permute_991: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1085, [1, 0])
        mm_337: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_991, view_44);  permute_991 = view_44 = None
        mm_338: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1085, permute_993);  view_1085 = permute_993 = None
        view_1086: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_338, [32, 128, 2048]);  mm_338 = None
        convert_element_type_2267: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1086, torch.float32);  view_1086 = None
        add_353: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_352, convert_element_type_2267);  add_352 = convert_element_type_2267 = None
        convert_element_type_2268: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_337, torch.float32);  mm_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_725: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_353, primals_32);  primals_32 = None
        mul_726: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_725, 2048)
        sum_293: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_725, [2], True)
        mul_727: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_725, mul_16);  mul_725 = None
        sum_294: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_727, [2], True);  mul_727 = None
        mul_728: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, sum_294);  sum_294 = None
        sub_233: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_726, sum_293);  mul_726 = sum_293 = None
        sub_234: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_233, mul_728);  sub_233 = mul_728 = None
        mul_729: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_70, sub_234);  div_70 = sub_234 = None
        mul_730: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_353, mul_16);  mul_16 = None
        sum_295: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_730, [0, 1]);  mul_730 = None
        sum_296: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_353, [0, 1]);  add_353 = None
        add_354: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_351, mul_729);  add_351 = mul_729 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_2269: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_354, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_1087: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2269, [4096, 2048]);  convert_element_type_2269 = None
        mm_339: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_1087, permute_995);  permute_995 = None
        permute_996: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1087, [1, 0])
        mm_340: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_996, view_42);  permute_996 = view_42 = None
        sum_297: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1087, [0], True, dtype = torch.float32);  view_1087 = None
        view_1088: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_297, [2048]);  sum_297 = None
        convert_element_type_2274: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1088, torch.bfloat16);  view_1088 = None
        view_1089: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_339, [32, 128, 8192]);  mm_339 = None
        convert_element_type_2275: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1089, torch.float32);  view_1089 = None
        convert_element_type_2276: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_340, torch.float32);  mm_340 = None
        convert_element_type_2277: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2274, torch.float32);  convert_element_type_2274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_41: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [32, 128, 8192]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_731: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2275, mul_12);  mul_12 = None
        convert_element_type_71: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32)
        pow_2: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_71, 3.0)
        mul_13: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_19: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_41, mul_13);  view_41 = mul_13 = None
        mul_14: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_19, 0.7978845608028654);  add_19 = None
        tanh_1: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_20: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_732: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2275, add_20);  convert_element_type_2275 = add_20 = None
        convert_element_type_2278: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_732, torch.bfloat16);  mul_732 = None
        mul_733: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_235: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_733);  mul_733 = None
        mul_734: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_731, sub_235);  mul_731 = sub_235 = None
        mul_735: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_734, 0.7978845608028654);  mul_734 = None
        convert_element_type_2279: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_735, torch.bfloat16)
        mul_736: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_735, 0.044715);  mul_735 = None
        pow_47: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_71, 2.0);  convert_element_type_71 = None
        mul_737: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_47, 3.0);  pow_47 = None
        mul_738: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_736, mul_737);  mul_736 = mul_737 = None
        convert_element_type_2280: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_738, torch.bfloat16);  mul_738 = None
        add_355: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2279, convert_element_type_2280);  convert_element_type_2279 = convert_element_type_2280 = None
        mul_739: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2278, 0.5);  convert_element_type_2278 = None
        add_356: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_355, mul_739);  add_355 = mul_739 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_1090: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_356, [4096, 8192]);  add_356 = None
        mm_341: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1090, permute_999);  permute_999 = None
        permute_1000: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_1090, [1, 0])
        mm_342: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1000, view_40);  permute_1000 = view_40 = None
        sum_298: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1090, [0], True, dtype = torch.float32);  view_1090 = None
        view_1091: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_298, [8192]);  sum_298 = None
        convert_element_type_2285: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1091, torch.bfloat16);  view_1091 = None
        view_1092: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_341, [32, 128, 2048]);  mm_341 = None
        convert_element_type_2286: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1092, torch.float32);  view_1092 = None
        convert_element_type_2287: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_342, torch.float32);  mm_342 = None
        convert_element_type_2288: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2285, torch.float32);  convert_element_type_2285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_741: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2286, primals_26);  primals_26 = None
        mul_742: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_741, 2048)
        sum_299: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_741, [2], True)
        mul_743: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_741, mul_10);  mul_741 = None
        sum_300: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_743, [2], True);  mul_743 = None
        mul_744: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, sum_300);  sum_300 = None
        sub_237: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_742, sum_299);  mul_742 = sum_299 = None
        sub_238: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_237, mul_744);  sub_237 = mul_744 = None
        mul_745: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_71, sub_238);  div_71 = sub_238 = None
        mul_746: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2286, mul_10);  mul_10 = None
        sum_301: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_746, [0, 1]);  mul_746 = None
        sum_302: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2286, [0, 1]);  convert_element_type_2286 = None
        add_357: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_354, mul_745);  add_354 = mul_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_2289: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_357, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_1093: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2289, [4096, 2048]);  convert_element_type_2289 = None
        mm_343: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1093, permute_1003);  permute_1003 = None
        permute_1004: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1093, [1, 0])
        mm_344: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1004, view_38);  permute_1004 = view_38 = None
        sum_303: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1093, [0], True, dtype = torch.float32);  view_1093 = None
        view_1094: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_303, [2048]);  sum_303 = None
        convert_element_type_2294: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1094, torch.bfloat16);  view_1094 = None
        view_1095: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_343, [32, 128, 2048]);  mm_343 = None
        convert_element_type_2295: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_344, torch.float32);  mm_344 = None
        convert_element_type_2296: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2294, torch.float32);  convert_element_type_2294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_1096: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1095, [32, 128, 16, 128]);  view_1095 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1007: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_1096, [0, 2, 1, 3]);  view_1096 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_258: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_1007, memory_format = torch.contiguous_format);  permute_1007 = None
        view_1097: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_258, [512, 128, 128]);  clone_258 = None
        bmm_136: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1008, view_1097);  permute_1008 = None
        bmm_137: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1097, permute_1009);  view_1097 = permute_1009 = None
        view_1098: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_136, [32, 16, 128, 128]);  bmm_136 = None
        view_1099: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_137, [32, 16, 128, 128]);  bmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2301: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1099, torch.float32);  view_1099 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_747: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2301, div_1);  convert_element_type_2301 = None
        sum_304: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_747, [-1], True)
        neg_23: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_22: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_23, sum_304, mul_747);  neg_23 = sum_304 = mul_747 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2302: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_22, torch.bfloat16);  fma_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_6: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_23, 2, 0, 128);  primals_23 = None
        slice_7: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_6, 3, 0, 128);  slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_51: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_7, convert_element_type_2302, full_default_31);  slice_7 = convert_element_type_2302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1100: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_51, [512, 128, 128]);  where_51 = None
        bmm_138: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1010, view_1100);  permute_1010 = None
        bmm_139: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1100, permute_1011);  view_1100 = permute_1011 = None
        view_1101: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_138, [32, 16, 128, 128]);  bmm_138 = None
        view_1102: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_139, [32, 16, 128, 128]);  bmm_139 = None
        convert_element_type_2308: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1101, torch.float32);  view_1101 = None
        permute_1012: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2308, [0, 1, 3, 2]);  convert_element_type_2308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2309: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1012, torch.bfloat16);  permute_1012 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_1: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1102, torch.bfloat16);  view_1102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1013: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_1098, [0, 2, 1, 3]);  view_1098 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_259: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_1013, memory_format = torch.contiguous_format);  permute_1013 = None
        view_1103: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_259, [32, 128, 2048]);  clone_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1014: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2309, [0, 2, 1, 3]);  convert_element_type_2309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1104: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_1014, [32, 128, 2048]);  permute_1014 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1015: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_1, [0, 2, 1, 3]);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_260: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_1015, memory_format = torch.contiguous_format);  permute_1015 = None
        view_1105: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_260, [32, 128, 2048]);  clone_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1106: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1103, [4096, 2048]);  view_1103 = None
        permute_1016: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1106, [1, 0])
        mm_345: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1016, view_22);  permute_1016 = None
        mm_346: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1106, permute_1018);  view_1106 = permute_1018 = None
        view_1107: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_346, [32, 128, 2048]);  mm_346 = None
        convert_element_type_2315: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1107, torch.float32);  view_1107 = None
        convert_element_type_2316: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_345, torch.float32);  mm_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_261: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_1104, memory_format = torch.contiguous_format);  view_1104 = None
        view_1108: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_261, [4096, 2048]);  clone_261 = None
        permute_1020: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1108, [1, 0])
        mm_347: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1020, view_22);  permute_1020 = None
        mm_348: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1108, permute_1022);  view_1108 = permute_1022 = None
        view_1109: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_348, [32, 128, 2048]);  mm_348 = None
        convert_element_type_2321: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1109, torch.float32);  view_1109 = None
        add_358: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2315, convert_element_type_2321);  convert_element_type_2315 = convert_element_type_2321 = None
        convert_element_type_2322: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_347, torch.float32);  mm_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1110: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1105, [4096, 2048]);  view_1105 = None
        permute_1024: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1110, [1, 0])
        mm_349: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1024, view_22);  permute_1024 = view_22 = None
        mm_350: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1110, permute_1026);  view_1110 = permute_1026 = None
        view_1111: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_350, [32, 128, 2048]);  mm_350 = None
        convert_element_type_2327: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1111, torch.float32);  view_1111 = None
        add_359: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_358, convert_element_type_2327);  add_358 = convert_element_type_2327 = None
        convert_element_type_2328: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_349, torch.float32);  mm_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_749: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_359, primals_18);  primals_18 = None
        mul_750: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_749, 2048)
        sum_305: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_749, [2], True)
        mul_751: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_749, mul_8);  mul_749 = None
        sum_306: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_751, [2], True);  mul_751 = None
        mul_752: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, sum_306);  sum_306 = None
        sub_240: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_750, sum_305);  mul_750 = sum_305 = None
        sub_241: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_240, mul_752);  sub_240 = mul_752 = None
        mul_753: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_72, sub_241);  div_72 = sub_241 = None
        mul_754: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_359, mul_8);  mul_8 = None
        sum_307: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_754, [0, 1]);  mul_754 = None
        sum_308: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_359, [0, 1]);  add_359 = None
        add_360: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_357, mul_753);  add_357 = mul_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_2329: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_360, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_1112: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2329, [4096, 2048]);  convert_element_type_2329 = None
        mm_351: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_1112, permute_1028);  permute_1028 = None
        permute_1029: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1112, [1, 0])
        mm_352: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_1029, view_20);  permute_1029 = view_20 = None
        sum_309: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1112, [0], True, dtype = torch.float32);  view_1112 = None
        view_1113: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_309, [2048]);  sum_309 = None
        convert_element_type_2334: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1113, torch.bfloat16);  view_1113 = None
        view_1114: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_351, [32, 128, 8192]);  mm_351 = None
        convert_element_type_2335: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1114, torch.float32);  view_1114 = None
        convert_element_type_2336: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_352, torch.float32);  mm_352 = None
        convert_element_type_2337: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2334, torch.float32);  convert_element_type_2334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_19: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [32, 128, 8192]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_755: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2335, mul_4);  mul_4 = None
        convert_element_type_32: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32)
        pow_1: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_32, 3.0)
        mul_5: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_10: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_19, mul_5);  view_19 = mul_5 = None
        mul_6: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_10, 0.7978845608028654);  add_10 = None
        tanh: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_11: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_756: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2335, add_11);  convert_element_type_2335 = add_11 = None
        convert_element_type_2338: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_756, torch.bfloat16);  mul_756 = None
        mul_757: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_242: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_757);  mul_757 = None
        mul_758: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_755, sub_242);  mul_755 = sub_242 = None
        mul_759: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_758, 0.7978845608028654);  mul_758 = None
        convert_element_type_2339: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_759, torch.bfloat16)
        mul_760: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_759, 0.044715);  mul_759 = None
        pow_48: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_32, 2.0);  convert_element_type_32 = None
        mul_761: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_48, 3.0);  pow_48 = None
        mul_762: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_760, mul_761);  mul_760 = mul_761 = None
        convert_element_type_2340: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_762, torch.bfloat16);  mul_762 = None
        add_361: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2339, convert_element_type_2340);  convert_element_type_2339 = convert_element_type_2340 = None
        mul_763: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2338, 0.5);  convert_element_type_2338 = None
        add_362: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_361, mul_763);  add_361 = mul_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_1115: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_362, [4096, 8192]);  add_362 = None
        mm_353: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1115, permute_1032);  permute_1032 = None
        permute_1033: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_1115, [1, 0])
        mm_354: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1033, view_18);  permute_1033 = view_18 = None
        sum_310: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1115, [0], True, dtype = torch.float32);  view_1115 = None
        view_1116: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_310, [8192]);  sum_310 = None
        convert_element_type_2345: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1116, torch.bfloat16);  view_1116 = None
        view_1117: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_353, [32, 128, 2048]);  mm_353 = None
        convert_element_type_2346: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1117, torch.float32);  view_1117 = None
        convert_element_type_2347: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_354, torch.float32);  mm_354 = None
        convert_element_type_2348: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2345, torch.float32);  convert_element_type_2345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_765: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2346, primals_12);  primals_12 = None
        mul_766: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_765, 2048)
        sum_311: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_765, [2], True)
        mul_767: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_765, mul_2);  mul_765 = None
        sum_312: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_767, [2], True);  mul_767 = None
        mul_768: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, sum_312);  sum_312 = None
        sub_244: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_766, sum_311);  mul_766 = sum_311 = None
        sub_245: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_244, mul_768);  sub_244 = mul_768 = None
        mul_769: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_73, sub_245);  div_73 = sub_245 = None
        mul_770: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2346, mul_2);  mul_2 = None
        sum_313: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_770, [0, 1]);  mul_770 = None
        sum_314: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2346, [0, 1]);  convert_element_type_2346 = None
        add_363: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_360, mul_769);  add_360 = mul_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_2349: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_363, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_1118: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2349, [4096, 2048]);  convert_element_type_2349 = None
        mm_355: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1118, permute_1036);  permute_1036 = None
        permute_1037: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1118, [1, 0])
        mm_356: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1037, view_16);  permute_1037 = view_16 = None
        sum_315: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1118, [0], True, dtype = torch.float32);  view_1118 = None
        view_1119: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_315, [2048]);  sum_315 = None
        convert_element_type_2354: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1119, torch.bfloat16);  view_1119 = None
        view_1120: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_355, [32, 128, 2048]);  mm_355 = None
        convert_element_type_2355: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_356, torch.float32);  mm_356 = None
        convert_element_type_2356: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2354, torch.float32);  convert_element_type_2354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_1121: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1120, [32, 128, 16, 128]);  view_1120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1040: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_1121, [0, 2, 1, 3]);  view_1121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_262: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_1040, memory_format = torch.contiguous_format);  permute_1040 = None
        view_1122: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_262, [512, 128, 128]);  clone_262 = None
        bmm_140: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1041, view_1122);  permute_1041 = None
        bmm_141: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1122, permute_1042);  view_1122 = permute_1042 = None
        view_1123: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_140, [32, 16, 128, 128]);  bmm_140 = None
        view_1124: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_141, [32, 16, 128, 128]);  bmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2361: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1124, torch.float32);  view_1124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        iota_1: "i64[32][1]cuda:0" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:360 in _non_vmap_expansion_sdpa, code: batch_indices = batch_indices[:, None, None, None]
        unsqueeze_1: "i64[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        unsqueeze_2: "i64[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_3: "i64[32, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_5: "i64[1, 1, 128][128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, 1)
        unsqueeze_6: "i64[1, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_9: "i64[1, 1, 1, 128][128, 128, 128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 2);  unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:52 in and_mask, code: result = q_idx.new_ones((), dtype=torch.bool)
        full: "b8[][]cuda:0" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_9, unsqueeze_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and: "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(full, le);  full = le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:168 in inner_mask, code: return packed_sequence_mask[batch_idx, q_idx] == packed_sequence_mask[batch_idx, kv_idx]
        index: "i64[32, 1, 128, 1][128, 128, 1, 1]cuda:0" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_6]);  unsqueeze_6 = None
        index_1: "i64[32, 1, 1, 128][128, 128, 128, 1]cuda:0" = torch.ops.aten.index.Tensor(cumsum, [unsqueeze_3, unsqueeze_9]);  cumsum = unsqueeze_3 = unsqueeze_9 = None
        eq: "b8[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:54 in and_mask, code: result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
        bitwise_and_1: "b8[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_1: "b8[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.expand.default(bitwise_and_1, [32, -1, 128, 128]);  bitwise_and_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(expand_1, full_default, full_default_1);  expand_1 = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_11: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [32, 16, 128, 128]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_4: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_9, 2, 0, 128);  primals_9 = None
        slice_5: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_4, 3, 0, 128);  slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:118 in _attn, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        full_default_2: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_1: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_5, view_11, full_default_2);  view_11 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        add_6: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(where_1, where);  where_1 = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        sub_3: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_6, amax);  add_6 = amax = None
        exp: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        div: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_771: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2361, div);  convert_element_type_2361 = None
        sum_316: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_771, [-1], True)
        neg_24: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div);  div = None
        fma_23: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_24, sum_316, mul_771);  neg_24 = sum_316 = mul_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2362: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_23, torch.bfloat16);  fma_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_52: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_5, convert_element_type_2362, full_default_31);  slice_5 = convert_element_type_2362 = full_default_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1125: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_52, [512, 128, 128]);  where_52 = None
        bmm_142: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1043, view_1125);  permute_1043 = None
        bmm_143: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1125, permute_1044);  view_1125 = permute_1044 = None
        view_1126: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_142, [32, 16, 128, 128]);  bmm_142 = None
        view_1127: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_143, [32, 16, 128, 128]);  bmm_143 = None
        convert_element_type_2368: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1126, torch.float32);  view_1126 = None
        permute_1045: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2368, [0, 1, 3, 2]);  convert_element_type_2368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2369: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1045, torch.bfloat16);  permute_1045 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1127, torch.bfloat16);  view_1127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1046: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_1123, [0, 2, 1, 3]);  view_1123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_263: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_1046, memory_format = torch.contiguous_format);  permute_1046 = None
        view_1128: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_263, [32, 128, 2048]);  clone_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1047: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2369, [0, 2, 1, 3]);  convert_element_type_2369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1129: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_1047, [32, 128, 2048]);  permute_1047 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1048: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default, [0, 2, 1, 3]);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_264: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_1048, memory_format = torch.contiguous_format);  permute_1048 = None
        view_1130: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_264, [32, 128, 2048]);  clone_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1131: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1128, [4096, 2048]);  view_1128 = None
        permute_1049: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1131, [1, 0])
        mm_357: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1049, view);  permute_1049 = None
        mm_358: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1131, permute_1051);  view_1131 = permute_1051 = None
        view_1132: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_358, [32, 128, 2048]);  mm_358 = None
        convert_element_type_2375: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1132, torch.float32);  view_1132 = None
        convert_element_type_2376: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_357, torch.float32);  mm_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_265: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_1129, memory_format = torch.contiguous_format);  view_1129 = None
        view_1133: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_265, [4096, 2048]);  clone_265 = None
        permute_1053: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1133, [1, 0])
        mm_359: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1053, view);  permute_1053 = None
        mm_360: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1133, permute_1055);  view_1133 = permute_1055 = None
        view_1134: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_360, [32, 128, 2048]);  mm_360 = None
        convert_element_type_2381: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1134, torch.float32);  view_1134 = None
        add_364: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2375, convert_element_type_2381);  convert_element_type_2375 = convert_element_type_2381 = None
        convert_element_type_2382: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_359, torch.float32);  mm_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1135: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1130, [4096, 2048]);  view_1130 = None
        permute_1057: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1135, [1, 0])
        mm_361: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1057, view);  permute_1057 = view = None
        mm_362: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1135, permute_1059);  view_1135 = permute_1059 = None
        view_1136: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_362, [32, 128, 2048]);  mm_362 = None
        convert_element_type_2387: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1136, torch.float32);  view_1136 = None
        add_365: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_364, convert_element_type_2387);  add_364 = convert_element_type_2387 = None
        convert_element_type_2388: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_361, torch.float32);  mm_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_773: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_365, primals_4);  primals_4 = None
        mul_774: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_773, 2048)
        sum_317: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_773, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:463 in forward, code: hidden_states = inputs_embeds + position_embeds
        add_3: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_2: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_3, getitem_1);  add_3 = getitem_1 = None
        mul: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_775: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_773, mul);  mul_773 = None
        sum_318: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_775, [2], True);  mul_775 = None
        mul_776: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_318);  sum_318 = None
        sub_247: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_774, sum_317);  mul_774 = sum_317 = None
        sub_248: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_247, mul_776);  sub_247 = mul_776 = None
        div_74: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 2048);  rsqrt = None
        mul_777: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_74, sub_248);  div_74 = sub_248 = None
        mul_778: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_365, mul);  mul = None
        sum_319: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_778, [0, 1]);  mul_778 = None
        sum_320: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_365, [0, 1]);  add_365 = None
        add_366: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_363, mul_777);  add_363 = mul_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:463 in forward, code: hidden_states = inputs_embeds + position_embeds
        sum_321: "f32[1, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_366, [0], True, dtype = torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:462 in forward, code: position_embeds = self.wpe(position_ids)
        full_default_55: "b8[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.full.default([1, 128, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.full.default([2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_56, full_default_55, [unsqueeze], sum_321);  full_default_56 = full_default_55 = unsqueeze = sum_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:444 in forward, code: inputs_embeds = self.wte(input_ids)
        ge_1: "b8[32, 128][128, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_1, 0)
        lt_1: "b8[32, 128][128, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_1, 50257)
        bitwise_and_4: "b8[32, 128][128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_7: "b8[32, 128][128, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_1, -1)
        bitwise_and_5: "b8[32, 128][128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, ne_7);  bitwise_and_4 = ne_7 = None
        unsqueeze_13: "b8[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_5, -1);  bitwise_and_5 = None
        full_default_57: "f32[50257, 2048][2048, 1]cuda:0" = torch.ops.aten.full.default([50257, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_1: "f32[50257, 2048][2048, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_57, unsqueeze_13, [primals_1], add_366);  full_default_57 = unsqueeze_13 = primals_1 = add_366 = None
        add_367: "f32[50257, 2048][2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_948, _unsafe_masked_index_put_accumulate_1);  convert_element_type_948 = _unsafe_masked_index_put_accumulate_1 = None
        return (None, add_367, _unsafe_masked_index_put_accumulate, sum_319, sum_320, convert_element_type_2388, convert_element_type_2382, convert_element_type_2376, None, convert_element_type_2355, convert_element_type_2356, sum_313, sum_314, convert_element_type_2347, convert_element_type_2348, convert_element_type_2336, convert_element_type_2337, sum_307, sum_308, convert_element_type_2328, convert_element_type_2322, convert_element_type_2316, None, convert_element_type_2295, convert_element_type_2296, sum_301, sum_302, convert_element_type_2287, convert_element_type_2288, convert_element_type_2276, convert_element_type_2277, sum_295, sum_296, convert_element_type_2268, convert_element_type_2262, convert_element_type_2256, None, convert_element_type_2235, convert_element_type_2236, sum_289, sum_290, convert_element_type_2227, convert_element_type_2228, convert_element_type_2216, convert_element_type_2217, sum_283, sum_284, convert_element_type_2208, convert_element_type_2202, convert_element_type_2196, None, convert_element_type_2175, convert_element_type_2176, sum_277, sum_278, convert_element_type_2167, convert_element_type_2168, convert_element_type_2156, convert_element_type_2157, sum_271, sum_272, convert_element_type_2148, convert_element_type_2142, convert_element_type_2136, None, convert_element_type_2115, convert_element_type_2116, sum_265, sum_266, convert_element_type_2107, convert_element_type_2108, convert_element_type_2096, convert_element_type_2097, sum_259, sum_260, convert_element_type_2088, convert_element_type_2082, convert_element_type_2076, None, convert_element_type_2055, convert_element_type_2056, sum_253, sum_254, convert_element_type_2047, convert_element_type_2048, convert_element_type_2036, convert_element_type_2037, sum_247, sum_248, convert_element_type_2028, convert_element_type_2022, convert_element_type_2016, None, convert_element_type_1995, convert_element_type_1996, sum_241, sum_242, convert_element_type_1987, convert_element_type_1988, convert_element_type_1976, convert_element_type_1977, sum_235, sum_236, convert_element_type_1968, convert_element_type_1962, convert_element_type_1956, None, convert_element_type_1935, convert_element_type_1936, sum_229, sum_230, convert_element_type_1927, convert_element_type_1928, convert_element_type_1916, convert_element_type_1917, sum_223, sum_224, convert_element_type_1908, convert_element_type_1902, convert_element_type_1896, None, convert_element_type_1875, convert_element_type_1876, sum_217, sum_218, convert_element_type_1867, convert_element_type_1868, convert_element_type_1856, convert_element_type_1857, sum_211, sum_212, convert_element_type_1848, convert_element_type_1842, convert_element_type_1836, None, convert_element_type_1815, convert_element_type_1816, sum_205, sum_206, convert_element_type_1807, convert_element_type_1808, convert_element_type_1796, convert_element_type_1797, sum_199, sum_200, convert_element_type_1788, convert_element_type_1782, convert_element_type_1776, None, convert_element_type_1755, convert_element_type_1756, sum_193, sum_194, convert_element_type_1747, convert_element_type_1748, convert_element_type_1736, convert_element_type_1737, sum_187, sum_188, convert_element_type_1728, convert_element_type_1722, convert_element_type_1716, None, convert_element_type_1695, convert_element_type_1696, sum_181, sum_182, convert_element_type_1687, convert_element_type_1688, convert_element_type_1676, convert_element_type_1677, sum_175, sum_176, convert_element_type_1668, convert_element_type_1662, convert_element_type_1656, None, convert_element_type_1635, convert_element_type_1636, sum_169, sum_170, convert_element_type_1627, convert_element_type_1628, convert_element_type_1616, convert_element_type_1617, sum_163, sum_164, convert_element_type_1608, convert_element_type_1602, convert_element_type_1596, None, convert_element_type_1575, convert_element_type_1576, sum_157, sum_158, convert_element_type_1567, convert_element_type_1568, convert_element_type_1556, convert_element_type_1557, sum_151, sum_152, convert_element_type_1548, convert_element_type_1542, convert_element_type_1536, None, convert_element_type_1515, convert_element_type_1516, sum_145, sum_146, convert_element_type_1507, convert_element_type_1508, convert_element_type_1496, convert_element_type_1497, sum_139, sum_140, convert_element_type_1488, convert_element_type_1482, convert_element_type_1476, None, convert_element_type_1455, convert_element_type_1456, sum_133, sum_134, convert_element_type_1447, convert_element_type_1448, convert_element_type_1436, convert_element_type_1437, sum_127, sum_128, convert_element_type_1428, convert_element_type_1422, convert_element_type_1416, None, convert_element_type_1395, convert_element_type_1396, sum_121, sum_122, convert_element_type_1387, convert_element_type_1388, convert_element_type_1376, convert_element_type_1377, sum_115, sum_116, convert_element_type_1368, convert_element_type_1362, convert_element_type_1356, None, convert_element_type_1335, convert_element_type_1336, sum_109, sum_110, convert_element_type_1327, convert_element_type_1328, convert_element_type_1316, convert_element_type_1317, sum_103, sum_104, convert_element_type_1308, convert_element_type_1302, convert_element_type_1296, None, convert_element_type_1275, convert_element_type_1276, sum_97, sum_98, convert_element_type_1267, convert_element_type_1268, convert_element_type_1256, convert_element_type_1257, sum_91, sum_92, convert_element_type_1248, convert_element_type_1242, convert_element_type_1236, None, convert_element_type_1215, convert_element_type_1216, sum_85, sum_86, convert_element_type_1207, convert_element_type_1208, convert_element_type_1196, convert_element_type_1197, sum_79, sum_80, convert_element_type_1188, convert_element_type_1182, convert_element_type_1176, None, convert_element_type_1155, convert_element_type_1156, sum_73, sum_74, convert_element_type_1147, convert_element_type_1148, convert_element_type_1136, convert_element_type_1137, sum_67, sum_68, convert_element_type_1128, convert_element_type_1122, convert_element_type_1116, None, convert_element_type_1095, convert_element_type_1096, sum_61, sum_62, convert_element_type_1087, convert_element_type_1088, convert_element_type_1076, convert_element_type_1077, sum_55, sum_56, convert_element_type_1068, convert_element_type_1062, convert_element_type_1056, None, convert_element_type_1035, convert_element_type_1036, sum_49, sum_50, convert_element_type_1027, convert_element_type_1028, convert_element_type_1016, convert_element_type_1017, sum_43, sum_44, convert_element_type_1008, convert_element_type_1002, convert_element_type_996, None, convert_element_type_975, convert_element_type_976, sum_37, sum_38, convert_element_type_967, convert_element_type_968, convert_element_type_956, convert_element_type_957, sum_31, sum_32, None)
