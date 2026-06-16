class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 128][128, 1]cuda:0", primals_4: "f32[2048][1]cuda:0", primals_9: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_12: "f32[2048][1]cuda:0", primals_18: "f32[2048][1]cuda:0", primals_23: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_26: "f32[2048][1]cuda:0", primals_32: "f32[2048][1]cuda:0", primals_37: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_40: "f32[2048][1]cuda:0", primals_46: "f32[2048][1]cuda:0", primals_51: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_54: "f32[2048][1]cuda:0", primals_60: "f32[2048][1]cuda:0", primals_65: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_68: "f32[2048][1]cuda:0", primals_74: "f32[2048][1]cuda:0", primals_79: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_82: "f32[2048][1]cuda:0", primals_88: "f32[2048][1]cuda:0", primals_93: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_96: "f32[2048][1]cuda:0", primals_102: "f32[2048][1]cuda:0", primals_107: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_110: "f32[2048][1]cuda:0", primals_116: "f32[2048][1]cuda:0", primals_121: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_124: "f32[2048][1]cuda:0", primals_130: "f32[2048][1]cuda:0", primals_135: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_138: "f32[2048][1]cuda:0", primals_144: "f32[2048][1]cuda:0", primals_149: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_152: "f32[2048][1]cuda:0", primals_158: "f32[2048][1]cuda:0", primals_163: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_166: "f32[2048][1]cuda:0", primals_172: "f32[2048][1]cuda:0", primals_177: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_180: "f32[2048][1]cuda:0", primals_186: "f32[2048][1]cuda:0", primals_191: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_194: "f32[2048][1]cuda:0", primals_200: "f32[2048][1]cuda:0", primals_205: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_208: "f32[2048][1]cuda:0", primals_214: "f32[2048][1]cuda:0", primals_219: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_222: "f32[2048][1]cuda:0", primals_228: "f32[2048][1]cuda:0", primals_233: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_236: "f32[2048][1]cuda:0", primals_242: "f32[2048][1]cuda:0", primals_247: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_250: "f32[2048][1]cuda:0", primals_256: "f32[2048][1]cuda:0", primals_261: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_264: "f32[2048][1]cuda:0", primals_270: "f32[2048][1]cuda:0", primals_275: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_278: "f32[2048][1]cuda:0", primals_284: "f32[2048][1]cuda:0", primals_289: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_292: "f32[2048][1]cuda:0", primals_298: "f32[2048][1]cuda:0", primals_303: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_306: "f32[2048][1]cuda:0", primals_312: "f32[2048][1]cuda:0", primals_317: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_320: "f32[2048][1]cuda:0", primals_326: "f32[2048][1]cuda:0", primals_331: "b8[1, 1, 2048, 2048][4194304, 4194304, 2048, 1]cuda:0", primals_334: "f32[2048][1]cuda:0", primals_340: "f32[2048][1]cuda:0", embedding: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", unsqueeze: "i64[1, 128][128, 1]cuda:0", cumsum: "i64[32, 128][128, 1]cuda:0", iota_1: "i64[32][1]cuda:0", embedding_1: "f32[1, 128, 2048][262144, 2048, 1]cuda:0", getitem_1: "f32[32, 128, 1][128, 1, 1]cuda:0", rsqrt: "f32[32, 128, 1][128, 1, 1]cuda:0", view: "bf16[4096, 2048][2048, 1]cuda:0", bmm: "bf16[512, 128, 128][16384, 128, 1]cuda:0", amax: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0", sum_1: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0", view_16: "bf16[4096, 2048][2048, 1]cuda:0", mul_2: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_18: "bf16[4096, 2048][2048, 1]cuda:0", addmm_1: "bf16[4096, 8192][8192, 1]cuda:0", view_20: "bf16[4096, 8192][8192, 1]cuda:0", mul_8: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_22: "bf16[4096, 2048][2048, 1]cuda:0", div_1: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_38: "bf16[4096, 2048][2048, 1]cuda:0", mul_10: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_40: "bf16[4096, 2048][2048, 1]cuda:0", addmm_4: "bf16[4096, 8192][8192, 1]cuda:0", view_42: "bf16[4096, 8192][8192, 1]cuda:0", mul_16: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_44: "bf16[4096, 2048][2048, 1]cuda:0", div_2: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_60: "bf16[4096, 2048][2048, 1]cuda:0", mul_18: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_62: "bf16[4096, 2048][2048, 1]cuda:0", addmm_7: "bf16[4096, 8192][8192, 1]cuda:0", view_64: "bf16[4096, 8192][8192, 1]cuda:0", mul_24: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_66: "bf16[4096, 2048][2048, 1]cuda:0", div_3: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_82: "bf16[4096, 2048][2048, 1]cuda:0", mul_26: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_84: "bf16[4096, 2048][2048, 1]cuda:0", addmm_10: "bf16[4096, 8192][8192, 1]cuda:0", view_86: "bf16[4096, 8192][8192, 1]cuda:0", mul_32: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_88: "bf16[4096, 2048][2048, 1]cuda:0", div_4: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_104: "bf16[4096, 2048][2048, 1]cuda:0", mul_34: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_106: "bf16[4096, 2048][2048, 1]cuda:0", addmm_13: "bf16[4096, 8192][8192, 1]cuda:0", view_108: "bf16[4096, 8192][8192, 1]cuda:0", mul_40: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_110: "bf16[4096, 2048][2048, 1]cuda:0", div_5: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_126: "bf16[4096, 2048][2048, 1]cuda:0", mul_42: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_128: "bf16[4096, 2048][2048, 1]cuda:0", addmm_16: "bf16[4096, 8192][8192, 1]cuda:0", view_130: "bf16[4096, 8192][8192, 1]cuda:0", mul_48: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_132: "bf16[4096, 2048][2048, 1]cuda:0", div_6: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_148: "bf16[4096, 2048][2048, 1]cuda:0", mul_50: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_150: "bf16[4096, 2048][2048, 1]cuda:0", addmm_19: "bf16[4096, 8192][8192, 1]cuda:0", view_152: "bf16[4096, 8192][8192, 1]cuda:0", mul_56: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_154: "bf16[4096, 2048][2048, 1]cuda:0", div_7: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_170: "bf16[4096, 2048][2048, 1]cuda:0", mul_58: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_172: "bf16[4096, 2048][2048, 1]cuda:0", addmm_22: "bf16[4096, 8192][8192, 1]cuda:0", view_174: "bf16[4096, 8192][8192, 1]cuda:0", mul_64: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_176: "bf16[4096, 2048][2048, 1]cuda:0", div_8: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_192: "bf16[4096, 2048][2048, 1]cuda:0", mul_66: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_194: "bf16[4096, 2048][2048, 1]cuda:0", addmm_25: "bf16[4096, 8192][8192, 1]cuda:0", view_196: "bf16[4096, 8192][8192, 1]cuda:0", mul_72: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_198: "bf16[4096, 2048][2048, 1]cuda:0", div_9: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_214: "bf16[4096, 2048][2048, 1]cuda:0", mul_74: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_216: "bf16[4096, 2048][2048, 1]cuda:0", addmm_28: "bf16[4096, 8192][8192, 1]cuda:0", view_218: "bf16[4096, 8192][8192, 1]cuda:0", mul_80: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_220: "bf16[4096, 2048][2048, 1]cuda:0", div_10: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_236: "bf16[4096, 2048][2048, 1]cuda:0", mul_82: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_238: "bf16[4096, 2048][2048, 1]cuda:0", addmm_31: "bf16[4096, 8192][8192, 1]cuda:0", view_240: "bf16[4096, 8192][8192, 1]cuda:0", mul_88: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_242: "bf16[4096, 2048][2048, 1]cuda:0", div_11: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_258: "bf16[4096, 2048][2048, 1]cuda:0", mul_90: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_260: "bf16[4096, 2048][2048, 1]cuda:0", addmm_34: "bf16[4096, 8192][8192, 1]cuda:0", view_262: "bf16[4096, 8192][8192, 1]cuda:0", mul_96: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_264: "bf16[4096, 2048][2048, 1]cuda:0", div_12: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_280: "bf16[4096, 2048][2048, 1]cuda:0", mul_98: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_282: "bf16[4096, 2048][2048, 1]cuda:0", addmm_37: "bf16[4096, 8192][8192, 1]cuda:0", view_284: "bf16[4096, 8192][8192, 1]cuda:0", mul_104: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_286: "bf16[4096, 2048][2048, 1]cuda:0", div_13: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_302: "bf16[4096, 2048][2048, 1]cuda:0", mul_106: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_304: "bf16[4096, 2048][2048, 1]cuda:0", addmm_40: "bf16[4096, 8192][8192, 1]cuda:0", view_306: "bf16[4096, 8192][8192, 1]cuda:0", mul_112: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_308: "bf16[4096, 2048][2048, 1]cuda:0", div_14: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_324: "bf16[4096, 2048][2048, 1]cuda:0", mul_114: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_326: "bf16[4096, 2048][2048, 1]cuda:0", addmm_43: "bf16[4096, 8192][8192, 1]cuda:0", view_328: "bf16[4096, 8192][8192, 1]cuda:0", mul_120: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_330: "bf16[4096, 2048][2048, 1]cuda:0", div_15: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_346: "bf16[4096, 2048][2048, 1]cuda:0", mul_122: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_348: "bf16[4096, 2048][2048, 1]cuda:0", addmm_46: "bf16[4096, 8192][8192, 1]cuda:0", view_350: "bf16[4096, 8192][8192, 1]cuda:0", mul_128: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_352: "bf16[4096, 2048][2048, 1]cuda:0", div_16: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_368: "bf16[4096, 2048][2048, 1]cuda:0", mul_130: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_370: "bf16[4096, 2048][2048, 1]cuda:0", addmm_49: "bf16[4096, 8192][8192, 1]cuda:0", view_372: "bf16[4096, 8192][8192, 1]cuda:0", mul_136: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_374: "bf16[4096, 2048][2048, 1]cuda:0", div_17: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_390: "bf16[4096, 2048][2048, 1]cuda:0", mul_138: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_392: "bf16[4096, 2048][2048, 1]cuda:0", addmm_52: "bf16[4096, 8192][8192, 1]cuda:0", view_394: "bf16[4096, 8192][8192, 1]cuda:0", mul_144: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_396: "bf16[4096, 2048][2048, 1]cuda:0", div_18: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_412: "bf16[4096, 2048][2048, 1]cuda:0", mul_146: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_414: "bf16[4096, 2048][2048, 1]cuda:0", addmm_55: "bf16[4096, 8192][8192, 1]cuda:0", view_416: "bf16[4096, 8192][8192, 1]cuda:0", mul_152: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_418: "bf16[4096, 2048][2048, 1]cuda:0", div_19: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_434: "bf16[4096, 2048][2048, 1]cuda:0", mul_154: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_436: "bf16[4096, 2048][2048, 1]cuda:0", addmm_58: "bf16[4096, 8192][8192, 1]cuda:0", view_438: "bf16[4096, 8192][8192, 1]cuda:0", mul_160: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_440: "bf16[4096, 2048][2048, 1]cuda:0", div_20: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_456: "bf16[4096, 2048][2048, 1]cuda:0", mul_162: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_458: "bf16[4096, 2048][2048, 1]cuda:0", addmm_61: "bf16[4096, 8192][8192, 1]cuda:0", view_460: "bf16[4096, 8192][8192, 1]cuda:0", mul_168: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_462: "bf16[4096, 2048][2048, 1]cuda:0", div_21: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_478: "bf16[4096, 2048][2048, 1]cuda:0", mul_170: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_480: "bf16[4096, 2048][2048, 1]cuda:0", addmm_64: "bf16[4096, 8192][8192, 1]cuda:0", view_482: "bf16[4096, 8192][8192, 1]cuda:0", mul_176: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_484: "bf16[4096, 2048][2048, 1]cuda:0", div_22: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_500: "bf16[4096, 2048][2048, 1]cuda:0", mul_178: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_502: "bf16[4096, 2048][2048, 1]cuda:0", addmm_67: "bf16[4096, 8192][8192, 1]cuda:0", view_504: "bf16[4096, 8192][8192, 1]cuda:0", mul_184: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_506: "bf16[4096, 2048][2048, 1]cuda:0", div_23: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0", view_522: "bf16[4096, 2048][2048, 1]cuda:0", mul_186: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_524: "bf16[4096, 2048][2048, 1]cuda:0", addmm_70: "bf16[4096, 8192][8192, 1]cuda:0", view_526: "bf16[4096, 8192][8192, 1]cuda:0", mul_192: "f32[32, 128, 2048][262144, 2048, 1]cuda:0", view_529: "bf16[4096, 2048][2048, 1]cuda:0", argmax: "i64[32][1]cuda:0", index_2: "bf16[32, 2][2, 1]cuda:0", convert_element_type_944: "f32[][]cuda:0", ne_5: "b8[32, 1][1, 1]cuda:0", eq_tensor: "b8[32, 2][2, 1]cuda:0", permute_267: "bf16[2, 2048][2048, 1]cuda:0", div_26: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_269: "bf16[2048, 8192][8192, 1]cuda:0", permute_273: "bf16[8192, 2048][2048, 1]cuda:0", div_27: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_277: "bf16[2048, 2048][2048, 1]cuda:0", permute_282: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_283: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_284: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_285: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_292: "bf16[2048, 2048][2048, 1]cuda:0", permute_296: "bf16[2048, 2048][2048, 1]cuda:0", permute_300: "bf16[2048, 2048][2048, 1]cuda:0", div_28: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_302: "bf16[2048, 8192][8192, 1]cuda:0", permute_306: "bf16[8192, 2048][2048, 1]cuda:0", div_29: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_310: "bf16[2048, 2048][2048, 1]cuda:0", permute_315: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_316: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_317: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_318: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_325: "bf16[2048, 2048][2048, 1]cuda:0", permute_329: "bf16[2048, 2048][2048, 1]cuda:0", permute_333: "bf16[2048, 2048][2048, 1]cuda:0", div_30: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_335: "bf16[2048, 8192][8192, 1]cuda:0", permute_339: "bf16[8192, 2048][2048, 1]cuda:0", div_31: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_343: "bf16[2048, 2048][2048, 1]cuda:0", permute_348: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_349: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_350: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_351: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_358: "bf16[2048, 2048][2048, 1]cuda:0", permute_362: "bf16[2048, 2048][2048, 1]cuda:0", permute_366: "bf16[2048, 2048][2048, 1]cuda:0", div_32: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_368: "bf16[2048, 8192][8192, 1]cuda:0", permute_372: "bf16[8192, 2048][2048, 1]cuda:0", div_33: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_376: "bf16[2048, 2048][2048, 1]cuda:0", permute_381: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_382: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_383: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_384: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_391: "bf16[2048, 2048][2048, 1]cuda:0", permute_395: "bf16[2048, 2048][2048, 1]cuda:0", permute_399: "bf16[2048, 2048][2048, 1]cuda:0", div_34: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_401: "bf16[2048, 8192][8192, 1]cuda:0", permute_405: "bf16[8192, 2048][2048, 1]cuda:0", div_35: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_409: "bf16[2048, 2048][2048, 1]cuda:0", permute_414: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_415: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_416: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_417: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_424: "bf16[2048, 2048][2048, 1]cuda:0", permute_428: "bf16[2048, 2048][2048, 1]cuda:0", permute_432: "bf16[2048, 2048][2048, 1]cuda:0", div_36: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_434: "bf16[2048, 8192][8192, 1]cuda:0", permute_438: "bf16[8192, 2048][2048, 1]cuda:0", div_37: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_442: "bf16[2048, 2048][2048, 1]cuda:0", permute_447: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_448: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_449: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_450: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_457: "bf16[2048, 2048][2048, 1]cuda:0", permute_461: "bf16[2048, 2048][2048, 1]cuda:0", permute_465: "bf16[2048, 2048][2048, 1]cuda:0", div_38: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_467: "bf16[2048, 8192][8192, 1]cuda:0", permute_471: "bf16[8192, 2048][2048, 1]cuda:0", div_39: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_475: "bf16[2048, 2048][2048, 1]cuda:0", permute_480: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_481: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_482: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_483: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_490: "bf16[2048, 2048][2048, 1]cuda:0", permute_494: "bf16[2048, 2048][2048, 1]cuda:0", permute_498: "bf16[2048, 2048][2048, 1]cuda:0", div_40: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_500: "bf16[2048, 8192][8192, 1]cuda:0", permute_504: "bf16[8192, 2048][2048, 1]cuda:0", div_41: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_508: "bf16[2048, 2048][2048, 1]cuda:0", permute_513: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_514: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_515: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_516: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_523: "bf16[2048, 2048][2048, 1]cuda:0", permute_527: "bf16[2048, 2048][2048, 1]cuda:0", permute_531: "bf16[2048, 2048][2048, 1]cuda:0", div_42: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_533: "bf16[2048, 8192][8192, 1]cuda:0", permute_537: "bf16[8192, 2048][2048, 1]cuda:0", div_43: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_541: "bf16[2048, 2048][2048, 1]cuda:0", permute_546: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_547: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_548: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_549: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_556: "bf16[2048, 2048][2048, 1]cuda:0", permute_560: "bf16[2048, 2048][2048, 1]cuda:0", permute_564: "bf16[2048, 2048][2048, 1]cuda:0", div_44: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_566: "bf16[2048, 8192][8192, 1]cuda:0", permute_570: "bf16[8192, 2048][2048, 1]cuda:0", div_45: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_574: "bf16[2048, 2048][2048, 1]cuda:0", permute_579: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_580: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_581: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_582: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_589: "bf16[2048, 2048][2048, 1]cuda:0", permute_593: "bf16[2048, 2048][2048, 1]cuda:0", permute_597: "bf16[2048, 2048][2048, 1]cuda:0", div_46: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_599: "bf16[2048, 8192][8192, 1]cuda:0", permute_603: "bf16[8192, 2048][2048, 1]cuda:0", div_47: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_607: "bf16[2048, 2048][2048, 1]cuda:0", permute_612: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_613: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_614: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_615: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_622: "bf16[2048, 2048][2048, 1]cuda:0", permute_626: "bf16[2048, 2048][2048, 1]cuda:0", permute_630: "bf16[2048, 2048][2048, 1]cuda:0", div_48: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_632: "bf16[2048, 8192][8192, 1]cuda:0", permute_636: "bf16[8192, 2048][2048, 1]cuda:0", div_49: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_640: "bf16[2048, 2048][2048, 1]cuda:0", permute_645: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_646: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_647: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_648: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_655: "bf16[2048, 2048][2048, 1]cuda:0", permute_659: "bf16[2048, 2048][2048, 1]cuda:0", permute_663: "bf16[2048, 2048][2048, 1]cuda:0", div_50: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_665: "bf16[2048, 8192][8192, 1]cuda:0", permute_669: "bf16[8192, 2048][2048, 1]cuda:0", div_51: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_673: "bf16[2048, 2048][2048, 1]cuda:0", permute_678: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_679: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_680: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_681: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_688: "bf16[2048, 2048][2048, 1]cuda:0", permute_692: "bf16[2048, 2048][2048, 1]cuda:0", permute_696: "bf16[2048, 2048][2048, 1]cuda:0", div_52: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_698: "bf16[2048, 8192][8192, 1]cuda:0", permute_702: "bf16[8192, 2048][2048, 1]cuda:0", div_53: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_706: "bf16[2048, 2048][2048, 1]cuda:0", permute_711: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_712: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_713: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_714: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_721: "bf16[2048, 2048][2048, 1]cuda:0", permute_725: "bf16[2048, 2048][2048, 1]cuda:0", permute_729: "bf16[2048, 2048][2048, 1]cuda:0", div_54: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_731: "bf16[2048, 8192][8192, 1]cuda:0", permute_735: "bf16[8192, 2048][2048, 1]cuda:0", div_55: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_739: "bf16[2048, 2048][2048, 1]cuda:0", permute_744: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_745: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_746: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_747: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_754: "bf16[2048, 2048][2048, 1]cuda:0", permute_758: "bf16[2048, 2048][2048, 1]cuda:0", permute_762: "bf16[2048, 2048][2048, 1]cuda:0", div_56: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_764: "bf16[2048, 8192][8192, 1]cuda:0", permute_768: "bf16[8192, 2048][2048, 1]cuda:0", div_57: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_772: "bf16[2048, 2048][2048, 1]cuda:0", permute_777: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_778: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_779: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_780: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_787: "bf16[2048, 2048][2048, 1]cuda:0", permute_791: "bf16[2048, 2048][2048, 1]cuda:0", permute_795: "bf16[2048, 2048][2048, 1]cuda:0", div_58: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_797: "bf16[2048, 8192][8192, 1]cuda:0", permute_801: "bf16[8192, 2048][2048, 1]cuda:0", div_59: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_805: "bf16[2048, 2048][2048, 1]cuda:0", permute_810: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_811: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_812: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_813: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_820: "bf16[2048, 2048][2048, 1]cuda:0", permute_824: "bf16[2048, 2048][2048, 1]cuda:0", permute_828: "bf16[2048, 2048][2048, 1]cuda:0", div_60: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_830: "bf16[2048, 8192][8192, 1]cuda:0", permute_834: "bf16[8192, 2048][2048, 1]cuda:0", div_61: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_838: "bf16[2048, 2048][2048, 1]cuda:0", permute_843: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_844: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_845: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_846: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_853: "bf16[2048, 2048][2048, 1]cuda:0", permute_857: "bf16[2048, 2048][2048, 1]cuda:0", permute_861: "bf16[2048, 2048][2048, 1]cuda:0", div_62: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_863: "bf16[2048, 8192][8192, 1]cuda:0", permute_867: "bf16[8192, 2048][2048, 1]cuda:0", div_63: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_871: "bf16[2048, 2048][2048, 1]cuda:0", permute_876: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_877: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_878: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_879: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_886: "bf16[2048, 2048][2048, 1]cuda:0", permute_890: "bf16[2048, 2048][2048, 1]cuda:0", permute_894: "bf16[2048, 2048][2048, 1]cuda:0", div_64: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_896: "bf16[2048, 8192][8192, 1]cuda:0", permute_900: "bf16[8192, 2048][2048, 1]cuda:0", div_65: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_904: "bf16[2048, 2048][2048, 1]cuda:0", permute_909: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_910: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_911: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_912: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_919: "bf16[2048, 2048][2048, 1]cuda:0", permute_923: "bf16[2048, 2048][2048, 1]cuda:0", permute_927: "bf16[2048, 2048][2048, 1]cuda:0", div_66: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_929: "bf16[2048, 8192][8192, 1]cuda:0", permute_933: "bf16[8192, 2048][2048, 1]cuda:0", div_67: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_937: "bf16[2048, 2048][2048, 1]cuda:0", permute_942: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_943: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_944: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_945: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_952: "bf16[2048, 2048][2048, 1]cuda:0", permute_956: "bf16[2048, 2048][2048, 1]cuda:0", permute_960: "bf16[2048, 2048][2048, 1]cuda:0", div_68: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_962: "bf16[2048, 8192][8192, 1]cuda:0", permute_966: "bf16[8192, 2048][2048, 1]cuda:0", div_69: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_970: "bf16[2048, 2048][2048, 1]cuda:0", permute_975: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_976: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_977: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_978: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_985: "bf16[2048, 2048][2048, 1]cuda:0", permute_989: "bf16[2048, 2048][2048, 1]cuda:0", permute_993: "bf16[2048, 2048][2048, 1]cuda:0", div_70: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_995: "bf16[2048, 8192][8192, 1]cuda:0", permute_999: "bf16[8192, 2048][2048, 1]cuda:0", div_71: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_1003: "bf16[2048, 2048][2048, 1]cuda:0", permute_1008: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_1009: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_1010: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_1011: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_1018: "bf16[2048, 2048][2048, 1]cuda:0", permute_1022: "bf16[2048, 2048][2048, 1]cuda:0", permute_1026: "bf16[2048, 2048][2048, 1]cuda:0", div_72: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_1028: "bf16[2048, 8192][8192, 1]cuda:0", permute_1032: "bf16[8192, 2048][2048, 1]cuda:0", div_73: "f32[32, 128, 1][128, 1, 1]cuda:0", permute_1036: "bf16[2048, 2048][2048, 1]cuda:0", permute_1041: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_1042: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_1043: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_1044: "bf16[512, 128, 128][16384, 1, 128]cuda:0", permute_1051: "bf16[2048, 2048][2048, 1]cuda:0", permute_1055: "bf16[2048, 2048][2048, 1]cuda:0", permute_1059: "bf16[2048, 2048][2048, 1]cuda:0", tangents_1: "f32[][]cuda:0", tangents_2: "bf16[32, 2][2, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:718 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        div_25: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_944);  tangents_1 = convert_element_type_944 = None

        # No stacktrace found for following nodes
        scalar_tensor_default: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:718 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        where_self: "f32[32, 2][2, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:718 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        where_28: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_5, div_25, full_default);  ne_5 = div_25 = None
        mul_195: "f32[32, 2][2, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self, where_28);  where_self = where_28 = None
        convert_element_type_945: "bf16[32, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_195, torch.bfloat16);  mul_195 = None
        convert_element_type_946: "f32[32, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_945, torch.float32);  convert_element_type_945 = None
        convert_element_type_941: "f32[32, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(index_2, torch.float32);  index_2 = None
        amax_24: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_941, [1], True)
        sub_75: "f32[32, 2][2, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_941, amax_24);  convert_element_type_941 = amax_24 = None
        exp_24: "f32[32, 2][2, 1]cuda:0" = torch.ops.aten.exp.default(sub_75)
        sum_25: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_76: "f32[32, 2][2, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_75, log);  sub_75 = log = None
        convert_element_type_942: "bf16[32, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_76, torch.bfloat16);  sub_76 = None
        convert_element_type_943: "f32[32, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_942, torch.float32);  convert_element_type_942 = None
        exp_25: "f32[32, 2][2, 1]cuda:0" = torch.ops.aten.exp.default(convert_element_type_943);  convert_element_type_943 = None
        sum_28: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_946, [1], True)
        mul_196: "f32[32, 2][2, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
        sub_77: "f32[32, 2][2, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_946, mul_196);  convert_element_type_946 = mul_196 = None
        convert_element_type_948: "bf16[32, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_77, torch.bfloat16);  sub_77 = None
        add_222: "bf16[32, 2][2, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, convert_element_type_948);  tangents_2 = convert_element_type_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:698 in forward, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        full_default_31: "bf16[32, 128, 2][256, 2, 1]cuda:0" = torch.ops.aten.full.default([32, 128, 2], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "bf16[32, 128, 2][256, 2, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_31, [iota_1, argmax], add_222, True);  full_default_31 = argmax = add_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:675 in forward, code: logits = self.score(hidden_states)
        view_534: "bf16[4096, 2][2, 1]cuda:0" = torch.ops.aten.reshape.default(index_put, [4096, 2]);  index_put = None
        permute_265: "bf16[2, 4096][1, 2]cuda:0" = torch.ops.aten.permute.default(view_534, [1, 0])
        mm_73: "bf16[2, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_265, view_529);  permute_265 = view_529 = None
        mm_74: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_534, permute_267);  view_534 = permute_267 = None
        view_535: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [32, 128, 2048]);  mm_74 = None
        convert_element_type_953: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_535, torch.float32);  view_535 = None
        convert_element_type_954: "f32[2, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:492 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_198: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_953, primals_340);  primals_340 = None
        mul_199: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_198, 2048)
        sum_29: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_198, [2], True)
        mul_200: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_198, mul_192);  mul_198 = None
        sum_30: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_200, [2], True);  mul_200 = None
        mul_201: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, sum_30);  sum_30 = None
        sub_79: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_199, sum_29);  mul_199 = sum_29 = None
        sub_80: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_79, mul_201);  sub_79 = mul_201 = None
        mul_202: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_26, sub_80);  div_26 = sub_80 = None
        mul_203: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_953, mul_192);  mul_192 = None
        sum_31: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_203, [0, 1]);  mul_203 = None
        sum_32: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_953, [0, 1]);  convert_element_type_953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_955: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_202, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_537: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_955, [4096, 2048]);  convert_element_type_955 = None
        mm_75: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_537, permute_269);  permute_269 = None
        permute_270: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_537, [1, 0])
        mm_76: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_270, view_526);  permute_270 = view_526 = None
        sum_33: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_537, [0], True, dtype = torch.float32);  view_537 = None
        view_538: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_33, [2048]);  sum_33 = None
        convert_element_type_960: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_538, torch.bfloat16);  view_538 = None
        view_539: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_75, [32, 128, 8192]);  mm_75 = None
        convert_element_type_961: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_539, torch.float32);  view_539 = None
        convert_element_type_962: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_76, torch.float32);  mm_76 = None
        convert_element_type_963: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_960, torch.float32);  convert_element_type_960 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_525: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [32, 128, 8192]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_188: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_525, 0.5)
        mul_204: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_961, mul_188);  mul_188 = None
        convert_element_type_929: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_525, torch.float32)
        pow_24: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_929, 3.0)
        mul_189: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_217: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_525, mul_189);  view_525 = mul_189 = None
        mul_190: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_217, 0.7978845608028654);  add_217 = None
        tanh_23: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_190);  mul_190 = None
        add_218: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_23, 1.0)
        mul_205: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_961, add_218);  convert_element_type_961 = add_218 = None
        convert_element_type_964: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_205, torch.bfloat16);  mul_205 = None
        mul_206: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_23, tanh_23);  tanh_23 = None
        sub_81: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_206);  mul_206 = None
        mul_207: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_204, sub_81);  mul_204 = sub_81 = None
        mul_208: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_207, 0.7978845608028654);  mul_207 = None
        convert_element_type_965: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_208, torch.bfloat16)
        mul_209: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_208, 0.044715);  mul_208 = None
        pow_25: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_929, 2.0);  convert_element_type_929 = None
        mul_210: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_25, 3.0);  pow_25 = None
        mul_211: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_209, mul_210);  mul_209 = mul_210 = None
        convert_element_type_966: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_211, torch.bfloat16);  mul_211 = None
        add_223: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_965, convert_element_type_966);  convert_element_type_965 = convert_element_type_966 = None
        mul_212: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_964, 0.5);  convert_element_type_964 = None
        add_224: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_223, mul_212);  add_223 = mul_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_540: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_224, [4096, 8192]);  add_224 = None
        mm_77: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_540, permute_273);  permute_273 = None
        permute_274: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_540, [1, 0])
        mm_78: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_274, view_524);  permute_274 = view_524 = None
        sum_34: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_540, [0], True, dtype = torch.float32);  view_540 = None
        view_541: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_34, [8192]);  sum_34 = None
        convert_element_type_971: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_541, torch.bfloat16);  view_541 = None
        view_542: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_77, [32, 128, 2048]);  mm_77 = None
        convert_element_type_972: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_542, torch.float32);  view_542 = None
        convert_element_type_973: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_78, torch.float32);  mm_78 = None
        convert_element_type_974: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_971, torch.float32);  convert_element_type_971 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_214: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_972, primals_334);  primals_334 = None
        mul_215: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_214, 2048)
        sum_35: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_214, [2], True)
        mul_216: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_214, mul_186);  mul_214 = None
        sum_36: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_216, [2], True);  mul_216 = None
        mul_217: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_186, sum_36);  sum_36 = None
        sub_83: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_215, sum_35);  mul_215 = sum_35 = None
        sub_84: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_83, mul_217);  sub_83 = mul_217 = None
        mul_218: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_27, sub_84);  div_27 = sub_84 = None
        mul_219: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_972, mul_186);  mul_186 = None
        sum_37: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_219, [0, 1]);  mul_219 = None
        sum_38: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_972, [0, 1]);  convert_element_type_972 = None
        add_225: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_202, mul_218);  mul_202 = mul_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_975: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_225, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_543: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_975, [4096, 2048]);  convert_element_type_975 = None
        mm_79: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_543, permute_277);  permute_277 = None
        permute_278: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_543, [1, 0])
        mm_80: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_278, view_522);  permute_278 = view_522 = None
        sum_39: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_543, [0], True, dtype = torch.float32);  view_543 = None
        view_544: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_39, [2048]);  sum_39 = None
        convert_element_type_980: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_544, torch.bfloat16);  view_544 = None
        view_545: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_79, [32, 128, 2048]);  mm_79 = None
        convert_element_type_981: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_80, torch.float32);  mm_80 = None
        convert_element_type_982: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_980, torch.float32);  convert_element_type_980 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_546: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_545, [32, 128, 16, 128]);  view_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_281: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_169: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_281, memory_format = torch.contiguous_format);  permute_281 = None
        view_547: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_169, [512, 128, 128]);  clone_169 = None
        bmm_48: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_282, view_547);  permute_282 = None
        bmm_49: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_547, permute_283);  view_547 = permute_283 = None
        view_548: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_48, [32, 16, 128, 128]);  bmm_48 = None
        view_549: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_49, [32, 16, 128, 128]);  bmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_987: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_549, torch.float32);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_220: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_987, div_23);  convert_element_type_987 = None
        sum_40: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_220, [-1], True)
        neg_1: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_23);  div_23 = None
        fma: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_1, sum_40, mul_220);  neg_1 = sum_40 = mul_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_988: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        full_default_32: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_50: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_331, 2, 0, 128);  primals_331 = None
        slice_51: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_50, 3, 0, 128);  slice_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_29: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_51, convert_element_type_988, full_default_32);  slice_51 = convert_element_type_988 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_550: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_29, [512, 128, 128]);  where_29 = None
        bmm_50: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_284, view_550);  permute_284 = None
        bmm_51: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_550, permute_285);  view_550 = permute_285 = None
        view_551: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_50, [32, 16, 128, 128]);  bmm_50 = None
        view_552: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_51, [32, 16, 128, 128]);  bmm_51 = None
        convert_element_type_994: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_551, torch.float32);  view_551 = None
        permute_286: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_994, [0, 1, 3, 2]);  convert_element_type_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_995: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_286, torch.bfloat16);  permute_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_23: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_552, torch.bfloat16);  view_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_287: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_548, [0, 2, 1, 3]);  view_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_170: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
        view_553: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_170, [32, 128, 2048]);  clone_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_288: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_995, [0, 2, 1, 3]);  convert_element_type_995 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_554: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_288, [32, 128, 2048]);  permute_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_289: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_23, [0, 2, 1, 3]);  convert_element_type_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_171: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
        view_555: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_171, [32, 128, 2048]);  clone_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_556: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_553, [4096, 2048]);  view_553 = None
        permute_290: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_556, [1, 0])
        mm_81: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_290, view_506);  permute_290 = None
        mm_82: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_556, permute_292);  view_556 = permute_292 = None
        view_557: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [32, 128, 2048]);  mm_82 = None
        convert_element_type_1001: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_557, torch.float32);  view_557 = None
        convert_element_type_1002: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_172: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_554, memory_format = torch.contiguous_format);  view_554 = None
        view_558: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_172, [4096, 2048]);  clone_172 = None
        permute_294: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_558, [1, 0])
        mm_83: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_294, view_506);  permute_294 = None
        mm_84: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_558, permute_296);  view_558 = permute_296 = None
        view_559: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [32, 128, 2048]);  mm_84 = None
        convert_element_type_1007: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_559, torch.float32);  view_559 = None
        add_226: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1001, convert_element_type_1007);  convert_element_type_1001 = convert_element_type_1007 = None
        convert_element_type_1008: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_560: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_555, [4096, 2048]);  view_555 = None
        permute_298: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_560, [1, 0])
        mm_85: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_298, view_506);  permute_298 = view_506 = None
        mm_86: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_560, permute_300);  view_560 = permute_300 = None
        view_561: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [32, 128, 2048]);  mm_86 = None
        convert_element_type_1013: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_561, torch.float32);  view_561 = None
        add_227: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_226, convert_element_type_1013);  add_226 = convert_element_type_1013 = None
        convert_element_type_1014: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_222: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_227, primals_326);  primals_326 = None
        mul_223: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_222, 2048)
        sum_41: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
        mul_224: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_222, mul_184);  mul_222 = None
        sum_42: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
        mul_225: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, sum_42);  sum_42 = None
        sub_86: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_223, sum_41);  mul_223 = sum_41 = None
        sub_87: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_86, mul_225);  sub_86 = mul_225 = None
        mul_226: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_28, sub_87);  div_28 = sub_87 = None
        mul_227: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_227, mul_184);  mul_184 = None
        sum_43: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
        sum_44: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_227, [0, 1]);  add_227 = None
        add_228: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_225, mul_226);  add_225 = mul_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1015: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_228, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_562: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1015, [4096, 2048]);  convert_element_type_1015 = None
        mm_87: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_562, permute_302);  permute_302 = None
        permute_303: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_562, [1, 0])
        mm_88: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_303, view_504);  permute_303 = view_504 = None
        sum_45: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_562, [0], True, dtype = torch.float32);  view_562 = None
        view_563: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_45, [2048]);  sum_45 = None
        convert_element_type_1020: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_563, torch.bfloat16);  view_563 = None
        view_564: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_87, [32, 128, 8192]);  mm_87 = None
        convert_element_type_1021: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_564, torch.float32);  view_564 = None
        convert_element_type_1022: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_88, torch.float32);  mm_88 = None
        convert_element_type_1023: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1020, torch.float32);  convert_element_type_1020 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_503: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [32, 128, 8192]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_180: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_503, 0.5)
        mul_228: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1021, mul_180);  mul_180 = None
        convert_element_type_890: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_503, torch.float32)
        pow_23: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_890, 3.0)
        mul_181: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_208: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_503, mul_181);  view_503 = mul_181 = None
        mul_182: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_208, 0.7978845608028654);  add_208 = None
        tanh_22: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_182);  mul_182 = None
        add_209: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_22, 1.0)
        mul_229: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1021, add_209);  convert_element_type_1021 = add_209 = None
        convert_element_type_1024: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_229, torch.bfloat16);  mul_229 = None
        mul_230: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_22, tanh_22);  tanh_22 = None
        sub_88: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_230);  mul_230 = None
        mul_231: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_228, sub_88);  mul_228 = sub_88 = None
        mul_232: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_231, 0.7978845608028654);  mul_231 = None
        convert_element_type_1025: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_232, torch.bfloat16)
        mul_233: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_232, 0.044715);  mul_232 = None
        pow_26: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_890, 2.0);  convert_element_type_890 = None
        mul_234: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_26, 3.0);  pow_26 = None
        mul_235: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_233, mul_234);  mul_233 = mul_234 = None
        convert_element_type_1026: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_235, torch.bfloat16);  mul_235 = None
        add_229: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1025, convert_element_type_1026);  convert_element_type_1025 = convert_element_type_1026 = None
        mul_236: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1024, 0.5);  convert_element_type_1024 = None
        add_230: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_229, mul_236);  add_229 = mul_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_565: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_230, [4096, 8192]);  add_230 = None
        mm_89: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_565, permute_306);  permute_306 = None
        permute_307: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_565, [1, 0])
        mm_90: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_307, view_502);  permute_307 = view_502 = None
        sum_46: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_565, [0], True, dtype = torch.float32);  view_565 = None
        view_566: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_46, [8192]);  sum_46 = None
        convert_element_type_1031: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_566, torch.bfloat16);  view_566 = None
        view_567: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_89, [32, 128, 2048]);  mm_89 = None
        convert_element_type_1032: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_567, torch.float32);  view_567 = None
        convert_element_type_1033: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_90, torch.float32);  mm_90 = None
        convert_element_type_1034: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1031, torch.float32);  convert_element_type_1031 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_238: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1032, primals_320);  primals_320 = None
        mul_239: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_238, 2048)
        sum_47: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True)
        mul_240: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_238, mul_178);  mul_238 = None
        sum_48: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True);  mul_240 = None
        mul_241: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, sum_48);  sum_48 = None
        sub_90: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_239, sum_47);  mul_239 = sum_47 = None
        sub_91: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, mul_241);  sub_90 = mul_241 = None
        mul_242: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_29, sub_91);  div_29 = sub_91 = None
        mul_243: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1032, mul_178);  mul_178 = None
        sum_49: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1]);  mul_243 = None
        sum_50: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1032, [0, 1]);  convert_element_type_1032 = None
        add_231: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_228, mul_242);  add_228 = mul_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1035: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_231, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_568: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1035, [4096, 2048]);  convert_element_type_1035 = None
        mm_91: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_568, permute_310);  permute_310 = None
        permute_311: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_568, [1, 0])
        mm_92: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_311, view_500);  permute_311 = view_500 = None
        sum_51: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_568, [0], True, dtype = torch.float32);  view_568 = None
        view_569: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_51, [2048]);  sum_51 = None
        convert_element_type_1040: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_569, torch.bfloat16);  view_569 = None
        view_570: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_91, [32, 128, 2048]);  mm_91 = None
        convert_element_type_1041: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        convert_element_type_1042: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1040, torch.float32);  convert_element_type_1040 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_571: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_570, [32, 128, 16, 128]);  view_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_314: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_571, [0, 2, 1, 3]);  view_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_173: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
        view_572: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_173, [512, 128, 128]);  clone_173 = None
        bmm_52: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_315, view_572);  permute_315 = None
        bmm_53: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_572, permute_316);  view_572 = permute_316 = None
        view_573: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_52, [32, 16, 128, 128]);  bmm_52 = None
        view_574: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_53, [32, 16, 128, 128]);  bmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1047: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_574, torch.float32);  view_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_244: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1047, div_22);  convert_element_type_1047 = None
        sum_52: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_244, [-1], True)
        neg_2: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_22);  div_22 = None
        fma_1: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_2, sum_52, mul_244);  neg_2 = sum_52 = mul_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1048: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_48: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_317, 2, 0, 128);  primals_317 = None
        slice_49: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_48, 3, 0, 128);  slice_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_30: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_49, convert_element_type_1048, full_default_32);  slice_49 = convert_element_type_1048 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_575: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_30, [512, 128, 128]);  where_30 = None
        bmm_54: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_317, view_575);  permute_317 = None
        bmm_55: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_575, permute_318);  view_575 = permute_318 = None
        view_576: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_54, [32, 16, 128, 128]);  bmm_54 = None
        view_577: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_55, [32, 16, 128, 128]);  bmm_55 = None
        convert_element_type_1054: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_576, torch.float32);  view_576 = None
        permute_319: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1054, [0, 1, 3, 2]);  convert_element_type_1054 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1055: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_319, torch.bfloat16);  permute_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_22: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_577, torch.bfloat16);  view_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_320: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_174: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
        view_578: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_174, [32, 128, 2048]);  clone_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_321: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1055, [0, 2, 1, 3]);  convert_element_type_1055 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_579: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_321, [32, 128, 2048]);  permute_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_322: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_22, [0, 2, 1, 3]);  convert_element_type_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_175: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
        view_580: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_175, [32, 128, 2048]);  clone_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_581: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_578, [4096, 2048]);  view_578 = None
        permute_323: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_581, [1, 0])
        mm_93: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_323, view_484);  permute_323 = None
        mm_94: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_581, permute_325);  view_581 = permute_325 = None
        view_582: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [32, 128, 2048]);  mm_94 = None
        convert_element_type_1061: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_582, torch.float32);  view_582 = None
        convert_element_type_1062: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_176: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_579, memory_format = torch.contiguous_format);  view_579 = None
        view_583: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_176, [4096, 2048]);  clone_176 = None
        permute_327: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_583, [1, 0])
        mm_95: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_327, view_484);  permute_327 = None
        mm_96: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_583, permute_329);  view_583 = permute_329 = None
        view_584: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [32, 128, 2048]);  mm_96 = None
        convert_element_type_1067: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_584, torch.float32);  view_584 = None
        add_232: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1061, convert_element_type_1067);  convert_element_type_1061 = convert_element_type_1067 = None
        convert_element_type_1068: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_585: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_580, [4096, 2048]);  view_580 = None
        permute_331: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_585, [1, 0])
        mm_97: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_331, view_484);  permute_331 = view_484 = None
        mm_98: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_585, permute_333);  view_585 = permute_333 = None
        view_586: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [32, 128, 2048]);  mm_98 = None
        convert_element_type_1073: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_586, torch.float32);  view_586 = None
        add_233: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_232, convert_element_type_1073);  add_232 = convert_element_type_1073 = None
        convert_element_type_1074: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_246: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_233, primals_312);  primals_312 = None
        mul_247: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_246, 2048)
        sum_53: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_246, [2], True)
        mul_248: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_246, mul_176);  mul_246 = None
        sum_54: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_248, [2], True);  mul_248 = None
        mul_249: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, sum_54);  sum_54 = None
        sub_93: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_247, sum_53);  mul_247 = sum_53 = None
        sub_94: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_93, mul_249);  sub_93 = mul_249 = None
        mul_250: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_30, sub_94);  div_30 = sub_94 = None
        mul_251: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_233, mul_176);  mul_176 = None
        sum_55: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_251, [0, 1]);  mul_251 = None
        sum_56: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None
        add_234: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_231, mul_250);  add_231 = mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1075: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_234, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_587: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1075, [4096, 2048]);  convert_element_type_1075 = None
        mm_99: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_587, permute_335);  permute_335 = None
        permute_336: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_587, [1, 0])
        mm_100: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_336, view_482);  permute_336 = view_482 = None
        sum_57: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_587, [0], True, dtype = torch.float32);  view_587 = None
        view_588: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_57, [2048]);  sum_57 = None
        convert_element_type_1080: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_588, torch.bfloat16);  view_588 = None
        view_589: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_99, [32, 128, 8192]);  mm_99 = None
        convert_element_type_1081: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_589, torch.float32);  view_589 = None
        convert_element_type_1082: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_100, torch.float32);  mm_100 = None
        convert_element_type_1083: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1080, torch.float32);  convert_element_type_1080 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_481: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [32, 128, 8192]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_172: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_481, 0.5)
        mul_252: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1081, mul_172);  mul_172 = None
        convert_element_type_851: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_481, torch.float32)
        pow_22: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_851, 3.0)
        mul_173: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_199: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_481, mul_173);  view_481 = mul_173 = None
        mul_174: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_199, 0.7978845608028654);  add_199 = None
        tanh_21: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_174);  mul_174 = None
        add_200: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_21, 1.0)
        mul_253: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1081, add_200);  convert_element_type_1081 = add_200 = None
        convert_element_type_1084: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_253, torch.bfloat16);  mul_253 = None
        mul_254: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_21, tanh_21);  tanh_21 = None
        sub_95: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_254);  mul_254 = None
        mul_255: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, sub_95);  mul_252 = sub_95 = None
        mul_256: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_255, 0.7978845608028654);  mul_255 = None
        convert_element_type_1085: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_256, torch.bfloat16)
        mul_257: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_256, 0.044715);  mul_256 = None
        pow_27: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_851, 2.0);  convert_element_type_851 = None
        mul_258: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_27, 3.0);  pow_27 = None
        mul_259: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_257, mul_258);  mul_257 = mul_258 = None
        convert_element_type_1086: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_259, torch.bfloat16);  mul_259 = None
        add_235: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1085, convert_element_type_1086);  convert_element_type_1085 = convert_element_type_1086 = None
        mul_260: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1084, 0.5);  convert_element_type_1084 = None
        add_236: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_235, mul_260);  add_235 = mul_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_590: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_236, [4096, 8192]);  add_236 = None
        mm_101: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_590, permute_339);  permute_339 = None
        permute_340: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_590, [1, 0])
        mm_102: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_340, view_480);  permute_340 = view_480 = None
        sum_58: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_590, [0], True, dtype = torch.float32);  view_590 = None
        view_591: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_58, [8192]);  sum_58 = None
        convert_element_type_1091: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_591, torch.bfloat16);  view_591 = None
        view_592: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_101, [32, 128, 2048]);  mm_101 = None
        convert_element_type_1092: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_592, torch.float32);  view_592 = None
        convert_element_type_1093: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_102, torch.float32);  mm_102 = None
        convert_element_type_1094: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1091, torch.float32);  convert_element_type_1091 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_262: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1092, primals_306);  primals_306 = None
        mul_263: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_262, 2048)
        sum_59: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_262, [2], True)
        mul_264: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_262, mul_170);  mul_262 = None
        sum_60: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_264, [2], True);  mul_264 = None
        mul_265: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_170, sum_60);  sum_60 = None
        sub_97: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_263, sum_59);  mul_263 = sum_59 = None
        sub_98: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_97, mul_265);  sub_97 = mul_265 = None
        mul_266: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_31, sub_98);  div_31 = sub_98 = None
        mul_267: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1092, mul_170);  mul_170 = None
        sum_61: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_267, [0, 1]);  mul_267 = None
        sum_62: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1092, [0, 1]);  convert_element_type_1092 = None
        add_237: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_234, mul_266);  add_234 = mul_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1095: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_237, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_593: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1095, [4096, 2048]);  convert_element_type_1095 = None
        mm_103: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_593, permute_343);  permute_343 = None
        permute_344: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_593, [1, 0])
        mm_104: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_344, view_478);  permute_344 = view_478 = None
        sum_63: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_593, [0], True, dtype = torch.float32);  view_593 = None
        view_594: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [2048]);  sum_63 = None
        convert_element_type_1100: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_594, torch.bfloat16);  view_594 = None
        view_595: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_103, [32, 128, 2048]);  mm_103 = None
        convert_element_type_1101: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_104, torch.float32);  mm_104 = None
        convert_element_type_1102: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1100, torch.float32);  convert_element_type_1100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_596: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_595, [32, 128, 16, 128]);  view_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_347: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_177: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
        view_597: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_177, [512, 128, 128]);  clone_177 = None
        bmm_56: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_348, view_597);  permute_348 = None
        bmm_57: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_597, permute_349);  view_597 = permute_349 = None
        view_598: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_56, [32, 16, 128, 128]);  bmm_56 = None
        view_599: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_57, [32, 16, 128, 128]);  bmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1107: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_599, torch.float32);  view_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_268: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1107, div_21);  convert_element_type_1107 = None
        sum_64: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_268, [-1], True)
        neg_3: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_21);  div_21 = None
        fma_2: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_3, sum_64, mul_268);  neg_3 = sum_64 = mul_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1108: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_46: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_303, 2, 0, 128);  primals_303 = None
        slice_47: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_46, 3, 0, 128);  slice_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_31: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_47, convert_element_type_1108, full_default_32);  slice_47 = convert_element_type_1108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_600: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_31, [512, 128, 128]);  where_31 = None
        bmm_58: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_350, view_600);  permute_350 = None
        bmm_59: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_600, permute_351);  view_600 = permute_351 = None
        view_601: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_58, [32, 16, 128, 128]);  bmm_58 = None
        view_602: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_59, [32, 16, 128, 128]);  bmm_59 = None
        convert_element_type_1114: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_601, torch.float32);  view_601 = None
        permute_352: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1114, [0, 1, 3, 2]);  convert_element_type_1114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1115: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_352, torch.bfloat16);  permute_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_21: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_602, torch.bfloat16);  view_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_353: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_178: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
        view_603: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_178, [32, 128, 2048]);  clone_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_354: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1115, [0, 2, 1, 3]);  convert_element_type_1115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_604: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_354, [32, 128, 2048]);  permute_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_355: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_21, [0, 2, 1, 3]);  convert_element_type_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_179: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
        view_605: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_179, [32, 128, 2048]);  clone_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_606: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_603, [4096, 2048]);  view_603 = None
        permute_356: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_606, [1, 0])
        mm_105: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_356, view_462);  permute_356 = None
        mm_106: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_606, permute_358);  view_606 = permute_358 = None
        view_607: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [32, 128, 2048]);  mm_106 = None
        convert_element_type_1121: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_607, torch.float32);  view_607 = None
        convert_element_type_1122: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_180: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_604, memory_format = torch.contiguous_format);  view_604 = None
        view_608: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_180, [4096, 2048]);  clone_180 = None
        permute_360: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_608, [1, 0])
        mm_107: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_360, view_462);  permute_360 = None
        mm_108: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_608, permute_362);  view_608 = permute_362 = None
        view_609: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [32, 128, 2048]);  mm_108 = None
        convert_element_type_1127: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_609, torch.float32);  view_609 = None
        add_238: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1121, convert_element_type_1127);  convert_element_type_1121 = convert_element_type_1127 = None
        convert_element_type_1128: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_610: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_605, [4096, 2048]);  view_605 = None
        permute_364: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_610, [1, 0])
        mm_109: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_364, view_462);  permute_364 = view_462 = None
        mm_110: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_610, permute_366);  view_610 = permute_366 = None
        view_611: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [32, 128, 2048]);  mm_110 = None
        convert_element_type_1133: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_611, torch.float32);  view_611 = None
        add_239: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_238, convert_element_type_1133);  add_238 = convert_element_type_1133 = None
        convert_element_type_1134: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_270: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_239, primals_298);  primals_298 = None
        mul_271: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_270, 2048)
        sum_65: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_270, [2], True)
        mul_272: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_270, mul_168);  mul_270 = None
        sum_66: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True);  mul_272 = None
        mul_273: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, sum_66);  sum_66 = None
        sub_100: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_271, sum_65);  mul_271 = sum_65 = None
        sub_101: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_100, mul_273);  sub_100 = mul_273 = None
        mul_274: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_32, sub_101);  div_32 = sub_101 = None
        mul_275: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_239, mul_168);  mul_168 = None
        sum_67: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1]);  mul_275 = None
        sum_68: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_239, [0, 1]);  add_239 = None
        add_240: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_237, mul_274);  add_237 = mul_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1135: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_240, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_612: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1135, [4096, 2048]);  convert_element_type_1135 = None
        mm_111: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_612, permute_368);  permute_368 = None
        permute_369: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_612, [1, 0])
        mm_112: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_369, view_460);  permute_369 = view_460 = None
        sum_69: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_612, [0], True, dtype = torch.float32);  view_612 = None
        view_613: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_69, [2048]);  sum_69 = None
        convert_element_type_1140: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_613, torch.bfloat16);  view_613 = None
        view_614: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_111, [32, 128, 8192]);  mm_111 = None
        convert_element_type_1141: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_614, torch.float32);  view_614 = None
        convert_element_type_1142: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_112, torch.float32);  mm_112 = None
        convert_element_type_1143: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1140, torch.float32);  convert_element_type_1140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_459: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [32, 128, 8192]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_164: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_459, 0.5)
        mul_276: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1141, mul_164);  mul_164 = None
        convert_element_type_812: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_459, torch.float32)
        pow_21: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_812, 3.0)
        mul_165: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_190: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_459, mul_165);  view_459 = mul_165 = None
        mul_166: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_190, 0.7978845608028654);  add_190 = None
        tanh_20: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
        add_191: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_20, 1.0)
        mul_277: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1141, add_191);  convert_element_type_1141 = add_191 = None
        convert_element_type_1144: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_277, torch.bfloat16);  mul_277 = None
        mul_278: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_20, tanh_20);  tanh_20 = None
        sub_102: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_278);  mul_278 = None
        mul_279: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_276, sub_102);  mul_276 = sub_102 = None
        mul_280: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_279, 0.7978845608028654);  mul_279 = None
        convert_element_type_1145: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_280, torch.bfloat16)
        mul_281: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_280, 0.044715);  mul_280 = None
        pow_28: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_812, 2.0);  convert_element_type_812 = None
        mul_282: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_28, 3.0);  pow_28 = None
        mul_283: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
        convert_element_type_1146: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_283, torch.bfloat16);  mul_283 = None
        add_241: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1145, convert_element_type_1146);  convert_element_type_1145 = convert_element_type_1146 = None
        mul_284: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1144, 0.5);  convert_element_type_1144 = None
        add_242: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_241, mul_284);  add_241 = mul_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_615: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_242, [4096, 8192]);  add_242 = None
        mm_113: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_615, permute_372);  permute_372 = None
        permute_373: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_615, [1, 0])
        mm_114: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_373, view_458);  permute_373 = view_458 = None
        sum_70: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_615, [0], True, dtype = torch.float32);  view_615 = None
        view_616: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [8192]);  sum_70 = None
        convert_element_type_1151: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_616, torch.bfloat16);  view_616 = None
        view_617: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_113, [32, 128, 2048]);  mm_113 = None
        convert_element_type_1152: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_617, torch.float32);  view_617 = None
        convert_element_type_1153: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_114, torch.float32);  mm_114 = None
        convert_element_type_1154: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1151, torch.float32);  convert_element_type_1151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_286: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1152, primals_292);  primals_292 = None
        mul_287: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_286, 2048)
        sum_71: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_286, [2], True)
        mul_288: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_286, mul_162);  mul_286 = None
        sum_72: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True);  mul_288 = None
        mul_289: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, sum_72);  sum_72 = None
        sub_104: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_287, sum_71);  mul_287 = sum_71 = None
        sub_105: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_104, mul_289);  sub_104 = mul_289 = None
        mul_290: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_33, sub_105);  div_33 = sub_105 = None
        mul_291: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1152, mul_162);  mul_162 = None
        sum_73: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 1]);  mul_291 = None
        sum_74: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1152, [0, 1]);  convert_element_type_1152 = None
        add_243: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_240, mul_290);  add_240 = mul_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1155: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_243, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_618: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1155, [4096, 2048]);  convert_element_type_1155 = None
        mm_115: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_618, permute_376);  permute_376 = None
        permute_377: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_618, [1, 0])
        mm_116: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_377, view_456);  permute_377 = view_456 = None
        sum_75: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_618, [0], True, dtype = torch.float32);  view_618 = None
        view_619: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_75, [2048]);  sum_75 = None
        convert_element_type_1160: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_619, torch.bfloat16);  view_619 = None
        view_620: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_115, [32, 128, 2048]);  mm_115 = None
        convert_element_type_1161: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_116, torch.float32);  mm_116 = None
        convert_element_type_1162: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1160, torch.float32);  convert_element_type_1160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_621: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_620, [32, 128, 16, 128]);  view_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_380: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_621, [0, 2, 1, 3]);  view_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_181: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_380, memory_format = torch.contiguous_format);  permute_380 = None
        view_622: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_181, [512, 128, 128]);  clone_181 = None
        bmm_60: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_381, view_622);  permute_381 = None
        bmm_61: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_622, permute_382);  view_622 = permute_382 = None
        view_623: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_60, [32, 16, 128, 128]);  bmm_60 = None
        view_624: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_61, [32, 16, 128, 128]);  bmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1167: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_624, torch.float32);  view_624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_292: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1167, div_20);  convert_element_type_1167 = None
        sum_76: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_292, [-1], True)
        neg_4: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_20);  div_20 = None
        fma_3: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_4, sum_76, mul_292);  neg_4 = sum_76 = mul_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1168: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_44: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_289, 2, 0, 128);  primals_289 = None
        slice_45: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_44, 3, 0, 128);  slice_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_32: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_45, convert_element_type_1168, full_default_32);  slice_45 = convert_element_type_1168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_625: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_32, [512, 128, 128]);  where_32 = None
        bmm_62: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_383, view_625);  permute_383 = None
        bmm_63: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_625, permute_384);  view_625 = permute_384 = None
        view_626: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_62, [32, 16, 128, 128]);  bmm_62 = None
        view_627: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_63, [32, 16, 128, 128]);  bmm_63 = None
        convert_element_type_1174: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_626, torch.float32);  view_626 = None
        permute_385: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1174, [0, 1, 3, 2]);  convert_element_type_1174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1175: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_385, torch.bfloat16);  permute_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_20: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_627, torch.bfloat16);  view_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_386: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_623, [0, 2, 1, 3]);  view_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_182: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
        view_628: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_182, [32, 128, 2048]);  clone_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_387: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1175, [0, 2, 1, 3]);  convert_element_type_1175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_629: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_387, [32, 128, 2048]);  permute_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_388: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_20, [0, 2, 1, 3]);  convert_element_type_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_183: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_388, memory_format = torch.contiguous_format);  permute_388 = None
        view_630: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_183, [32, 128, 2048]);  clone_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_631: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_628, [4096, 2048]);  view_628 = None
        permute_389: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_631, [1, 0])
        mm_117: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_389, view_440);  permute_389 = None
        mm_118: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_631, permute_391);  view_631 = permute_391 = None
        view_632: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_118, [32, 128, 2048]);  mm_118 = None
        convert_element_type_1181: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_632, torch.float32);  view_632 = None
        convert_element_type_1182: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_184: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_629, memory_format = torch.contiguous_format);  view_629 = None
        view_633: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_184, [4096, 2048]);  clone_184 = None
        permute_393: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_633, [1, 0])
        mm_119: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_393, view_440);  permute_393 = None
        mm_120: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_633, permute_395);  view_633 = permute_395 = None
        view_634: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_120, [32, 128, 2048]);  mm_120 = None
        convert_element_type_1187: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_634, torch.float32);  view_634 = None
        add_244: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1181, convert_element_type_1187);  convert_element_type_1181 = convert_element_type_1187 = None
        convert_element_type_1188: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_635: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_630, [4096, 2048]);  view_630 = None
        permute_397: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_635, [1, 0])
        mm_121: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_397, view_440);  permute_397 = view_440 = None
        mm_122: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_635, permute_399);  view_635 = permute_399 = None
        view_636: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [32, 128, 2048]);  mm_122 = None
        convert_element_type_1193: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_636, torch.float32);  view_636 = None
        add_245: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_244, convert_element_type_1193);  add_244 = convert_element_type_1193 = None
        convert_element_type_1194: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_294: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_245, primals_284);  primals_284 = None
        mul_295: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_294, 2048)
        sum_77: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_294, [2], True)
        mul_296: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_294, mul_160);  mul_294 = None
        sum_78: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_296, [2], True);  mul_296 = None
        mul_297: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, sum_78);  sum_78 = None
        sub_107: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_295, sum_77);  mul_295 = sum_77 = None
        sub_108: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_107, mul_297);  sub_107 = mul_297 = None
        mul_298: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_34, sub_108);  div_34 = sub_108 = None
        mul_299: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_245, mul_160);  mul_160 = None
        sum_79: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_299, [0, 1]);  mul_299 = None
        sum_80: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_245, [0, 1]);  add_245 = None
        add_246: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_243, mul_298);  add_243 = mul_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1195: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_246, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_637: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1195, [4096, 2048]);  convert_element_type_1195 = None
        mm_123: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_637, permute_401);  permute_401 = None
        permute_402: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_637, [1, 0])
        mm_124: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_402, view_438);  permute_402 = view_438 = None
        sum_81: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_637, [0], True, dtype = torch.float32);  view_637 = None
        view_638: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_81, [2048]);  sum_81 = None
        convert_element_type_1200: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_638, torch.bfloat16);  view_638 = None
        view_639: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_123, [32, 128, 8192]);  mm_123 = None
        convert_element_type_1201: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_639, torch.float32);  view_639 = None
        convert_element_type_1202: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_124, torch.float32);  mm_124 = None
        convert_element_type_1203: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1200, torch.float32);  convert_element_type_1200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_437: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [32, 128, 8192]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_156: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_437, 0.5)
        mul_300: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1201, mul_156);  mul_156 = None
        convert_element_type_773: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_437, torch.float32)
        pow_20: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_773, 3.0)
        mul_157: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_181: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_437, mul_157);  view_437 = mul_157 = None
        mul_158: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_181, 0.7978845608028654);  add_181 = None
        tanh_19: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
        add_182: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_19, 1.0)
        mul_301: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1201, add_182);  convert_element_type_1201 = add_182 = None
        convert_element_type_1204: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_301, torch.bfloat16);  mul_301 = None
        mul_302: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_19, tanh_19);  tanh_19 = None
        sub_109: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_302);  mul_302 = None
        mul_303: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_300, sub_109);  mul_300 = sub_109 = None
        mul_304: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_303, 0.7978845608028654);  mul_303 = None
        convert_element_type_1205: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_304, torch.bfloat16)
        mul_305: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_304, 0.044715);  mul_304 = None
        pow_29: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_773, 2.0);  convert_element_type_773 = None
        mul_306: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_29, 3.0);  pow_29 = None
        mul_307: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
        convert_element_type_1206: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_307, torch.bfloat16);  mul_307 = None
        add_247: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1205, convert_element_type_1206);  convert_element_type_1205 = convert_element_type_1206 = None
        mul_308: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1204, 0.5);  convert_element_type_1204 = None
        add_248: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_247, mul_308);  add_247 = mul_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_640: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_248, [4096, 8192]);  add_248 = None
        mm_125: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_640, permute_405);  permute_405 = None
        permute_406: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_640, [1, 0])
        mm_126: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_406, view_436);  permute_406 = view_436 = None
        sum_82: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_640, [0], True, dtype = torch.float32);  view_640 = None
        view_641: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_82, [8192]);  sum_82 = None
        convert_element_type_1211: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_641, torch.bfloat16);  view_641 = None
        view_642: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_125, [32, 128, 2048]);  mm_125 = None
        convert_element_type_1212: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_642, torch.float32);  view_642 = None
        convert_element_type_1213: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_126, torch.float32);  mm_126 = None
        convert_element_type_1214: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1211, torch.float32);  convert_element_type_1211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_310: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1212, primals_278);  primals_278 = None
        mul_311: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_310, 2048)
        sum_83: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_310, [2], True)
        mul_312: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_310, mul_154);  mul_310 = None
        sum_84: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_312, [2], True);  mul_312 = None
        mul_313: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, sum_84);  sum_84 = None
        sub_111: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_311, sum_83);  mul_311 = sum_83 = None
        sub_112: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_111, mul_313);  sub_111 = mul_313 = None
        mul_314: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_35, sub_112);  div_35 = sub_112 = None
        mul_315: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1212, mul_154);  mul_154 = None
        sum_85: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_315, [0, 1]);  mul_315 = None
        sum_86: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1212, [0, 1]);  convert_element_type_1212 = None
        add_249: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_246, mul_314);  add_246 = mul_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1215: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_249, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_643: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1215, [4096, 2048]);  convert_element_type_1215 = None
        mm_127: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_643, permute_409);  permute_409 = None
        permute_410: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_643, [1, 0])
        mm_128: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_410, view_434);  permute_410 = view_434 = None
        sum_87: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_643, [0], True, dtype = torch.float32);  view_643 = None
        view_644: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_87, [2048]);  sum_87 = None
        convert_element_type_1220: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_644, torch.bfloat16);  view_644 = None
        view_645: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_127, [32, 128, 2048]);  mm_127 = None
        convert_element_type_1221: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_128, torch.float32);  mm_128 = None
        convert_element_type_1222: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1220, torch.float32);  convert_element_type_1220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_646: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_645, [32, 128, 16, 128]);  view_645 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_413: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_646, [0, 2, 1, 3]);  view_646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_185: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_413, memory_format = torch.contiguous_format);  permute_413 = None
        view_647: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_185, [512, 128, 128]);  clone_185 = None
        bmm_64: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_414, view_647);  permute_414 = None
        bmm_65: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_647, permute_415);  view_647 = permute_415 = None
        view_648: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_64, [32, 16, 128, 128]);  bmm_64 = None
        view_649: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_65, [32, 16, 128, 128]);  bmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1227: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_649, torch.float32);  view_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_316: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1227, div_19);  convert_element_type_1227 = None
        sum_88: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_316, [-1], True)
        neg_5: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_19);  div_19 = None
        fma_4: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_5, sum_88, mul_316);  neg_5 = sum_88 = mul_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1228: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_42: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_275, 2, 0, 128);  primals_275 = None
        slice_43: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_42, 3, 0, 128);  slice_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_33: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_43, convert_element_type_1228, full_default_32);  slice_43 = convert_element_type_1228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_650: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_33, [512, 128, 128]);  where_33 = None
        bmm_66: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_416, view_650);  permute_416 = None
        bmm_67: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_650, permute_417);  view_650 = permute_417 = None
        view_651: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_66, [32, 16, 128, 128]);  bmm_66 = None
        view_652: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_67, [32, 16, 128, 128]);  bmm_67 = None
        convert_element_type_1234: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_651, torch.float32);  view_651 = None
        permute_418: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1234, [0, 1, 3, 2]);  convert_element_type_1234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1235: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_418, torch.bfloat16);  permute_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_19: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_652, torch.bfloat16);  view_652 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_419: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_186: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
        view_653: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_186, [32, 128, 2048]);  clone_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_420: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1235, [0, 2, 1, 3]);  convert_element_type_1235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_654: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_420, [32, 128, 2048]);  permute_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_421: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_19, [0, 2, 1, 3]);  convert_element_type_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_187: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_421, memory_format = torch.contiguous_format);  permute_421 = None
        view_655: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_187, [32, 128, 2048]);  clone_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_656: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_653, [4096, 2048]);  view_653 = None
        permute_422: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_656, [1, 0])
        mm_129: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_422, view_418);  permute_422 = None
        mm_130: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_656, permute_424);  view_656 = permute_424 = None
        view_657: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_130, [32, 128, 2048]);  mm_130 = None
        convert_element_type_1241: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_657, torch.float32);  view_657 = None
        convert_element_type_1242: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_188: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_654, memory_format = torch.contiguous_format);  view_654 = None
        view_658: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_188, [4096, 2048]);  clone_188 = None
        permute_426: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_658, [1, 0])
        mm_131: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_426, view_418);  permute_426 = None
        mm_132: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_658, permute_428);  view_658 = permute_428 = None
        view_659: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [32, 128, 2048]);  mm_132 = None
        convert_element_type_1247: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_659, torch.float32);  view_659 = None
        add_250: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1241, convert_element_type_1247);  convert_element_type_1241 = convert_element_type_1247 = None
        convert_element_type_1248: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_660: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_655, [4096, 2048]);  view_655 = None
        permute_430: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_660, [1, 0])
        mm_133: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_430, view_418);  permute_430 = view_418 = None
        mm_134: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_660, permute_432);  view_660 = permute_432 = None
        view_661: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [32, 128, 2048]);  mm_134 = None
        convert_element_type_1253: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_661, torch.float32);  view_661 = None
        add_251: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_250, convert_element_type_1253);  add_250 = convert_element_type_1253 = None
        convert_element_type_1254: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_318: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_251, primals_270);  primals_270 = None
        mul_319: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, 2048)
        sum_89: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True)
        mul_320: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, mul_152);  mul_318 = None
        sum_90: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True);  mul_320 = None
        mul_321: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, sum_90);  sum_90 = None
        sub_114: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_319, sum_89);  mul_319 = sum_89 = None
        sub_115: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_114, mul_321);  sub_114 = mul_321 = None
        mul_322: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_36, sub_115);  div_36 = sub_115 = None
        mul_323: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_251, mul_152);  mul_152 = None
        sum_91: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_323, [0, 1]);  mul_323 = None
        sum_92: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_251, [0, 1]);  add_251 = None
        add_252: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_249, mul_322);  add_249 = mul_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1255: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_252, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_662: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1255, [4096, 2048]);  convert_element_type_1255 = None
        mm_135: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_662, permute_434);  permute_434 = None
        permute_435: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_662, [1, 0])
        mm_136: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_435, view_416);  permute_435 = view_416 = None
        sum_93: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_662, [0], True, dtype = torch.float32);  view_662 = None
        view_663: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_93, [2048]);  sum_93 = None
        convert_element_type_1260: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_663, torch.bfloat16);  view_663 = None
        view_664: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_135, [32, 128, 8192]);  mm_135 = None
        convert_element_type_1261: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_664, torch.float32);  view_664 = None
        convert_element_type_1262: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_136, torch.float32);  mm_136 = None
        convert_element_type_1263: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1260, torch.float32);  convert_element_type_1260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_415: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [32, 128, 8192]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_148: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_415, 0.5)
        mul_324: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1261, mul_148);  mul_148 = None
        convert_element_type_734: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_415, torch.float32)
        pow_19: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_734, 3.0)
        mul_149: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_172: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_415, mul_149);  view_415 = mul_149 = None
        mul_150: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_172, 0.7978845608028654);  add_172 = None
        tanh_18: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_150);  mul_150 = None
        add_173: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_18, 1.0)
        mul_325: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1261, add_173);  convert_element_type_1261 = add_173 = None
        convert_element_type_1264: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_325, torch.bfloat16);  mul_325 = None
        mul_326: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_18, tanh_18);  tanh_18 = None
        sub_116: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_326);  mul_326 = None
        mul_327: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_324, sub_116);  mul_324 = sub_116 = None
        mul_328: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, 0.7978845608028654);  mul_327 = None
        convert_element_type_1265: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_328, torch.bfloat16)
        mul_329: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_328, 0.044715);  mul_328 = None
        pow_30: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_734, 2.0);  convert_element_type_734 = None
        mul_330: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_30, 3.0);  pow_30 = None
        mul_331: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_329, mul_330);  mul_329 = mul_330 = None
        convert_element_type_1266: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_331, torch.bfloat16);  mul_331 = None
        add_253: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1265, convert_element_type_1266);  convert_element_type_1265 = convert_element_type_1266 = None
        mul_332: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1264, 0.5);  convert_element_type_1264 = None
        add_254: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_253, mul_332);  add_253 = mul_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_665: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_254, [4096, 8192]);  add_254 = None
        mm_137: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_665, permute_438);  permute_438 = None
        permute_439: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_665, [1, 0])
        mm_138: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_439, view_414);  permute_439 = view_414 = None
        sum_94: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_665, [0], True, dtype = torch.float32);  view_665 = None
        view_666: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_94, [8192]);  sum_94 = None
        convert_element_type_1271: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_666, torch.bfloat16);  view_666 = None
        view_667: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_137, [32, 128, 2048]);  mm_137 = None
        convert_element_type_1272: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_667, torch.float32);  view_667 = None
        convert_element_type_1273: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_138, torch.float32);  mm_138 = None
        convert_element_type_1274: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1271, torch.float32);  convert_element_type_1271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_334: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1272, primals_264);  primals_264 = None
        mul_335: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_334, 2048)
        sum_95: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_334, [2], True)
        mul_336: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_334, mul_146);  mul_334 = None
        sum_96: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_336, [2], True);  mul_336 = None
        mul_337: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, sum_96);  sum_96 = None
        sub_118: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_335, sum_95);  mul_335 = sum_95 = None
        sub_119: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_118, mul_337);  sub_118 = mul_337 = None
        mul_338: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_37, sub_119);  div_37 = sub_119 = None
        mul_339: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1272, mul_146);  mul_146 = None
        sum_97: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_339, [0, 1]);  mul_339 = None
        sum_98: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1272, [0, 1]);  convert_element_type_1272 = None
        add_255: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_252, mul_338);  add_252 = mul_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1275: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_255, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_668: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1275, [4096, 2048]);  convert_element_type_1275 = None
        mm_139: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_668, permute_442);  permute_442 = None
        permute_443: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_668, [1, 0])
        mm_140: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_443, view_412);  permute_443 = view_412 = None
        sum_99: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_668, [0], True, dtype = torch.float32);  view_668 = None
        view_669: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_99, [2048]);  sum_99 = None
        convert_element_type_1280: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_669, torch.bfloat16);  view_669 = None
        view_670: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_139, [32, 128, 2048]);  mm_139 = None
        convert_element_type_1281: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_140, torch.float32);  mm_140 = None
        convert_element_type_1282: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1280, torch.float32);  convert_element_type_1280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_671: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_670, [32, 128, 16, 128]);  view_670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_446: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_671, [0, 2, 1, 3]);  view_671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_189: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_446, memory_format = torch.contiguous_format);  permute_446 = None
        view_672: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_189, [512, 128, 128]);  clone_189 = None
        bmm_68: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_447, view_672);  permute_447 = None
        bmm_69: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_672, permute_448);  view_672 = permute_448 = None
        view_673: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_68, [32, 16, 128, 128]);  bmm_68 = None
        view_674: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_69, [32, 16, 128, 128]);  bmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1287: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_674, torch.float32);  view_674 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_340: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1287, div_18);  convert_element_type_1287 = None
        sum_100: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_340, [-1], True)
        neg_6: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_18);  div_18 = None
        fma_5: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_6, sum_100, mul_340);  neg_6 = sum_100 = mul_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1288: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_40: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_261, 2, 0, 128);  primals_261 = None
        slice_41: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_40, 3, 0, 128);  slice_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_34: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_41, convert_element_type_1288, full_default_32);  slice_41 = convert_element_type_1288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_675: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_34, [512, 128, 128]);  where_34 = None
        bmm_70: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_449, view_675);  permute_449 = None
        bmm_71: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_675, permute_450);  view_675 = permute_450 = None
        view_676: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_70, [32, 16, 128, 128]);  bmm_70 = None
        view_677: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_71, [32, 16, 128, 128]);  bmm_71 = None
        convert_element_type_1294: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_676, torch.float32);  view_676 = None
        permute_451: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1294, [0, 1, 3, 2]);  convert_element_type_1294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1295: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_451, torch.bfloat16);  permute_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_18: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_677, torch.bfloat16);  view_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_452: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_190: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
        view_678: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_190, [32, 128, 2048]);  clone_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_453: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1295, [0, 2, 1, 3]);  convert_element_type_1295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_679: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_453, [32, 128, 2048]);  permute_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_454: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_18, [0, 2, 1, 3]);  convert_element_type_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_191: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_454, memory_format = torch.contiguous_format);  permute_454 = None
        view_680: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_191, [32, 128, 2048]);  clone_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_681: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_678, [4096, 2048]);  view_678 = None
        permute_455: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_681, [1, 0])
        mm_141: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_455, view_396);  permute_455 = None
        mm_142: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_681, permute_457);  view_681 = permute_457 = None
        view_682: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_142, [32, 128, 2048]);  mm_142 = None
        convert_element_type_1301: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_682, torch.float32);  view_682 = None
        convert_element_type_1302: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_192: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_679, memory_format = torch.contiguous_format);  view_679 = None
        view_683: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_192, [4096, 2048]);  clone_192 = None
        permute_459: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_683, [1, 0])
        mm_143: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_459, view_396);  permute_459 = None
        mm_144: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_683, permute_461);  view_683 = permute_461 = None
        view_684: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_144, [32, 128, 2048]);  mm_144 = None
        convert_element_type_1307: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_684, torch.float32);  view_684 = None
        add_256: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1301, convert_element_type_1307);  convert_element_type_1301 = convert_element_type_1307 = None
        convert_element_type_1308: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_685: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_680, [4096, 2048]);  view_680 = None
        permute_463: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_685, [1, 0])
        mm_145: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_463, view_396);  permute_463 = view_396 = None
        mm_146: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_685, permute_465);  view_685 = permute_465 = None
        view_686: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_146, [32, 128, 2048]);  mm_146 = None
        convert_element_type_1313: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_686, torch.float32);  view_686 = None
        add_257: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_256, convert_element_type_1313);  add_256 = convert_element_type_1313 = None
        convert_element_type_1314: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_342: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_257, primals_256);  primals_256 = None
        mul_343: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_342, 2048)
        sum_101: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True)
        mul_344: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_342, mul_144);  mul_342 = None
        sum_102: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_344, [2], True);  mul_344 = None
        mul_345: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, sum_102);  sum_102 = None
        sub_121: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_343, sum_101);  mul_343 = sum_101 = None
        sub_122: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_121, mul_345);  sub_121 = mul_345 = None
        mul_346: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_38, sub_122);  div_38 = sub_122 = None
        mul_347: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_257, mul_144);  mul_144 = None
        sum_103: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_347, [0, 1]);  mul_347 = None
        sum_104: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_257, [0, 1]);  add_257 = None
        add_258: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_255, mul_346);  add_255 = mul_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1315: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_258, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_687: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1315, [4096, 2048]);  convert_element_type_1315 = None
        mm_147: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_687, permute_467);  permute_467 = None
        permute_468: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_687, [1, 0])
        mm_148: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_468, view_394);  permute_468 = view_394 = None
        sum_105: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_687, [0], True, dtype = torch.float32);  view_687 = None
        view_688: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_105, [2048]);  sum_105 = None
        convert_element_type_1320: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_688, torch.bfloat16);  view_688 = None
        view_689: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_147, [32, 128, 8192]);  mm_147 = None
        convert_element_type_1321: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_689, torch.float32);  view_689 = None
        convert_element_type_1322: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_148, torch.float32);  mm_148 = None
        convert_element_type_1323: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1320, torch.float32);  convert_element_type_1320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_393: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [32, 128, 8192]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_140: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_393, 0.5)
        mul_348: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1321, mul_140);  mul_140 = None
        convert_element_type_695: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_393, torch.float32)
        pow_18: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_695, 3.0)
        mul_141: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_163: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_393, mul_141);  view_393 = mul_141 = None
        mul_142: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_163, 0.7978845608028654);  add_163 = None
        tanh_17: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
        add_164: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_17, 1.0)
        mul_349: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1321, add_164);  convert_element_type_1321 = add_164 = None
        convert_element_type_1324: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_349, torch.bfloat16);  mul_349 = None
        mul_350: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_17, tanh_17);  tanh_17 = None
        sub_123: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_350);  mul_350 = None
        mul_351: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_348, sub_123);  mul_348 = sub_123 = None
        mul_352: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_351, 0.7978845608028654);  mul_351 = None
        convert_element_type_1325: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_352, torch.bfloat16)
        mul_353: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_352, 0.044715);  mul_352 = None
        pow_31: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_695, 2.0);  convert_element_type_695 = None
        mul_354: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_31, 3.0);  pow_31 = None
        mul_355: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_353, mul_354);  mul_353 = mul_354 = None
        convert_element_type_1326: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_355, torch.bfloat16);  mul_355 = None
        add_259: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1325, convert_element_type_1326);  convert_element_type_1325 = convert_element_type_1326 = None
        mul_356: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1324, 0.5);  convert_element_type_1324 = None
        add_260: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_259, mul_356);  add_259 = mul_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_690: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_260, [4096, 8192]);  add_260 = None
        mm_149: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_690, permute_471);  permute_471 = None
        permute_472: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_690, [1, 0])
        mm_150: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_472, view_392);  permute_472 = view_392 = None
        sum_106: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_690, [0], True, dtype = torch.float32);  view_690 = None
        view_691: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_106, [8192]);  sum_106 = None
        convert_element_type_1331: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_691, torch.bfloat16);  view_691 = None
        view_692: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_149, [32, 128, 2048]);  mm_149 = None
        convert_element_type_1332: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_692, torch.float32);  view_692 = None
        convert_element_type_1333: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_150, torch.float32);  mm_150 = None
        convert_element_type_1334: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1331, torch.float32);  convert_element_type_1331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_358: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1332, primals_250);  primals_250 = None
        mul_359: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_358, 2048)
        sum_107: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True)
        mul_360: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_358, mul_138);  mul_358 = None
        sum_108: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_360, [2], True);  mul_360 = None
        mul_361: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, sum_108);  sum_108 = None
        sub_125: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_359, sum_107);  mul_359 = sum_107 = None
        sub_126: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_125, mul_361);  sub_125 = mul_361 = None
        mul_362: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_39, sub_126);  div_39 = sub_126 = None
        mul_363: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1332, mul_138);  mul_138 = None
        sum_109: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 1]);  mul_363 = None
        sum_110: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1332, [0, 1]);  convert_element_type_1332 = None
        add_261: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_258, mul_362);  add_258 = mul_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1335: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_261, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_693: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1335, [4096, 2048]);  convert_element_type_1335 = None
        mm_151: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_693, permute_475);  permute_475 = None
        permute_476: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_693, [1, 0])
        mm_152: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_476, view_390);  permute_476 = view_390 = None
        sum_111: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_693, [0], True, dtype = torch.float32);  view_693 = None
        view_694: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_111, [2048]);  sum_111 = None
        convert_element_type_1340: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_694, torch.bfloat16);  view_694 = None
        view_695: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_151, [32, 128, 2048]);  mm_151 = None
        convert_element_type_1341: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_152, torch.float32);  mm_152 = None
        convert_element_type_1342: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1340, torch.float32);  convert_element_type_1340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_696: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_695, [32, 128, 16, 128]);  view_695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_479: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_696, [0, 2, 1, 3]);  view_696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_193: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
        view_697: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_193, [512, 128, 128]);  clone_193 = None
        bmm_72: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_480, view_697);  permute_480 = None
        bmm_73: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_697, permute_481);  view_697 = permute_481 = None
        view_698: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_72, [32, 16, 128, 128]);  bmm_72 = None
        view_699: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_73, [32, 16, 128, 128]);  bmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1347: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_699, torch.float32);  view_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_364: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1347, div_17);  convert_element_type_1347 = None
        sum_112: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_364, [-1], True)
        neg_7: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_17);  div_17 = None
        fma_6: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_7, sum_112, mul_364);  neg_7 = sum_112 = mul_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1348: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_38: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_247, 2, 0, 128);  primals_247 = None
        slice_39: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_38, 3, 0, 128);  slice_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_35: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_39, convert_element_type_1348, full_default_32);  slice_39 = convert_element_type_1348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_700: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_35, [512, 128, 128]);  where_35 = None
        bmm_74: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_482, view_700);  permute_482 = None
        bmm_75: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_700, permute_483);  view_700 = permute_483 = None
        view_701: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_74, [32, 16, 128, 128]);  bmm_74 = None
        view_702: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_75, [32, 16, 128, 128]);  bmm_75 = None
        convert_element_type_1354: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_701, torch.float32);  view_701 = None
        permute_484: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1354, [0, 1, 3, 2]);  convert_element_type_1354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1355: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_484, torch.bfloat16);  permute_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_17: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_702, torch.bfloat16);  view_702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_485: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_698, [0, 2, 1, 3]);  view_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_194: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
        view_703: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_194, [32, 128, 2048]);  clone_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_486: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1355, [0, 2, 1, 3]);  convert_element_type_1355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_704: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_486, [32, 128, 2048]);  permute_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_487: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_17, [0, 2, 1, 3]);  convert_element_type_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_195: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_487, memory_format = torch.contiguous_format);  permute_487 = None
        view_705: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_195, [32, 128, 2048]);  clone_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_706: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_703, [4096, 2048]);  view_703 = None
        permute_488: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_706, [1, 0])
        mm_153: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_488, view_374);  permute_488 = None
        mm_154: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_706, permute_490);  view_706 = permute_490 = None
        view_707: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_154, [32, 128, 2048]);  mm_154 = None
        convert_element_type_1361: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_707, torch.float32);  view_707 = None
        convert_element_type_1362: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_153, torch.float32);  mm_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_196: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_704, memory_format = torch.contiguous_format);  view_704 = None
        view_708: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_196, [4096, 2048]);  clone_196 = None
        permute_492: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_708, [1, 0])
        mm_155: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_492, view_374);  permute_492 = None
        mm_156: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_708, permute_494);  view_708 = permute_494 = None
        view_709: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_156, [32, 128, 2048]);  mm_156 = None
        convert_element_type_1367: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_709, torch.float32);  view_709 = None
        add_262: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1361, convert_element_type_1367);  convert_element_type_1361 = convert_element_type_1367 = None
        convert_element_type_1368: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_155, torch.float32);  mm_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_710: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_705, [4096, 2048]);  view_705 = None
        permute_496: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_710, [1, 0])
        mm_157: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_496, view_374);  permute_496 = view_374 = None
        mm_158: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_710, permute_498);  view_710 = permute_498 = None
        view_711: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_158, [32, 128, 2048]);  mm_158 = None
        convert_element_type_1373: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_711, torch.float32);  view_711 = None
        add_263: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_262, convert_element_type_1373);  add_262 = convert_element_type_1373 = None
        convert_element_type_1374: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_157, torch.float32);  mm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_366: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_263, primals_242);  primals_242 = None
        mul_367: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_366, 2048)
        sum_113: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_366, [2], True)
        mul_368: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_366, mul_136);  mul_366 = None
        sum_114: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_368, [2], True);  mul_368 = None
        mul_369: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, sum_114);  sum_114 = None
        sub_128: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_367, sum_113);  mul_367 = sum_113 = None
        sub_129: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_128, mul_369);  sub_128 = mul_369 = None
        mul_370: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_40, sub_129);  div_40 = sub_129 = None
        mul_371: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_263, mul_136);  mul_136 = None
        sum_115: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_371, [0, 1]);  mul_371 = None
        sum_116: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None
        add_264: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_261, mul_370);  add_261 = mul_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1375: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_264, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_712: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1375, [4096, 2048]);  convert_element_type_1375 = None
        mm_159: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_712, permute_500);  permute_500 = None
        permute_501: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_712, [1, 0])
        mm_160: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_501, view_372);  permute_501 = view_372 = None
        sum_117: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_712, [0], True, dtype = torch.float32);  view_712 = None
        view_713: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_117, [2048]);  sum_117 = None
        convert_element_type_1380: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_713, torch.bfloat16);  view_713 = None
        view_714: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_159, [32, 128, 8192]);  mm_159 = None
        convert_element_type_1381: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_714, torch.float32);  view_714 = None
        convert_element_type_1382: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_160, torch.float32);  mm_160 = None
        convert_element_type_1383: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1380, torch.float32);  convert_element_type_1380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_371: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [32, 128, 8192]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_132: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_371, 0.5)
        mul_372: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1381, mul_132);  mul_132 = None
        convert_element_type_656: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_371, torch.float32)
        pow_17: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_656, 3.0)
        mul_133: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_154: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_371, mul_133);  view_371 = mul_133 = None
        mul_134: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_154, 0.7978845608028654);  add_154 = None
        tanh_16: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_134);  mul_134 = None
        add_155: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_16, 1.0)
        mul_373: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1381, add_155);  convert_element_type_1381 = add_155 = None
        convert_element_type_1384: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_373, torch.bfloat16);  mul_373 = None
        mul_374: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_16, tanh_16);  tanh_16 = None
        sub_130: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_374);  mul_374 = None
        mul_375: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_372, sub_130);  mul_372 = sub_130 = None
        mul_376: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_375, 0.7978845608028654);  mul_375 = None
        convert_element_type_1385: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_376, torch.bfloat16)
        mul_377: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_376, 0.044715);  mul_376 = None
        pow_32: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_656, 2.0);  convert_element_type_656 = None
        mul_378: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_32, 3.0);  pow_32 = None
        mul_379: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_377, mul_378);  mul_377 = mul_378 = None
        convert_element_type_1386: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_379, torch.bfloat16);  mul_379 = None
        add_265: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1385, convert_element_type_1386);  convert_element_type_1385 = convert_element_type_1386 = None
        mul_380: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1384, 0.5);  convert_element_type_1384 = None
        add_266: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_265, mul_380);  add_265 = mul_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_715: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_266, [4096, 8192]);  add_266 = None
        mm_161: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_715, permute_504);  permute_504 = None
        permute_505: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_715, [1, 0])
        mm_162: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_505, view_370);  permute_505 = view_370 = None
        sum_118: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_715, [0], True, dtype = torch.float32);  view_715 = None
        view_716: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_118, [8192]);  sum_118 = None
        convert_element_type_1391: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_716, torch.bfloat16);  view_716 = None
        view_717: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_161, [32, 128, 2048]);  mm_161 = None
        convert_element_type_1392: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_717, torch.float32);  view_717 = None
        convert_element_type_1393: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_162, torch.float32);  mm_162 = None
        convert_element_type_1394: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1391, torch.float32);  convert_element_type_1391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_382: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1392, primals_236);  primals_236 = None
        mul_383: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_382, 2048)
        sum_119: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_382, [2], True)
        mul_384: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_382, mul_130);  mul_382 = None
        sum_120: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_384, [2], True);  mul_384 = None
        mul_385: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, sum_120);  sum_120 = None
        sub_132: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_383, sum_119);  mul_383 = sum_119 = None
        sub_133: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_132, mul_385);  sub_132 = mul_385 = None
        mul_386: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_41, sub_133);  div_41 = sub_133 = None
        mul_387: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1392, mul_130);  mul_130 = None
        sum_121: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_387, [0, 1]);  mul_387 = None
        sum_122: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1392, [0, 1]);  convert_element_type_1392 = None
        add_267: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_264, mul_386);  add_264 = mul_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1395: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_267, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_718: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1395, [4096, 2048]);  convert_element_type_1395 = None
        mm_163: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_718, permute_508);  permute_508 = None
        permute_509: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_718, [1, 0])
        mm_164: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_509, view_368);  permute_509 = view_368 = None
        sum_123: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_718, [0], True, dtype = torch.float32);  view_718 = None
        view_719: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_123, [2048]);  sum_123 = None
        convert_element_type_1400: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_719, torch.bfloat16);  view_719 = None
        view_720: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_163, [32, 128, 2048]);  mm_163 = None
        convert_element_type_1401: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_164, torch.float32);  mm_164 = None
        convert_element_type_1402: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1400, torch.float32);  convert_element_type_1400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_721: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_720, [32, 128, 16, 128]);  view_720 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_512: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_721, [0, 2, 1, 3]);  view_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_197: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_512, memory_format = torch.contiguous_format);  permute_512 = None
        view_722: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_197, [512, 128, 128]);  clone_197 = None
        bmm_76: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_513, view_722);  permute_513 = None
        bmm_77: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_722, permute_514);  view_722 = permute_514 = None
        view_723: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_76, [32, 16, 128, 128]);  bmm_76 = None
        view_724: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_77, [32, 16, 128, 128]);  bmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1407: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_724, torch.float32);  view_724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_388: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1407, div_16);  convert_element_type_1407 = None
        sum_124: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_388, [-1], True)
        neg_8: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_16);  div_16 = None
        fma_7: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_8, sum_124, mul_388);  neg_8 = sum_124 = mul_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1408: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_36: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_233, 2, 0, 128);  primals_233 = None
        slice_37: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_36, 3, 0, 128);  slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_36: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_37, convert_element_type_1408, full_default_32);  slice_37 = convert_element_type_1408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_725: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_36, [512, 128, 128]);  where_36 = None
        bmm_78: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_515, view_725);  permute_515 = None
        bmm_79: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_725, permute_516);  view_725 = permute_516 = None
        view_726: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_78, [32, 16, 128, 128]);  bmm_78 = None
        view_727: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_79, [32, 16, 128, 128]);  bmm_79 = None
        convert_element_type_1414: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_726, torch.float32);  view_726 = None
        permute_517: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1414, [0, 1, 3, 2]);  convert_element_type_1414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1415: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_517, torch.bfloat16);  permute_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_16: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_727, torch.bfloat16);  view_727 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_518: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_723, [0, 2, 1, 3]);  view_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_198: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
        view_728: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_198, [32, 128, 2048]);  clone_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_519: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1415, [0, 2, 1, 3]);  convert_element_type_1415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_729: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_519, [32, 128, 2048]);  permute_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_520: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_16, [0, 2, 1, 3]);  convert_element_type_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_199: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_520, memory_format = torch.contiguous_format);  permute_520 = None
        view_730: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_199, [32, 128, 2048]);  clone_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_731: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_728, [4096, 2048]);  view_728 = None
        permute_521: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_731, [1, 0])
        mm_165: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_521, view_352);  permute_521 = None
        mm_166: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_731, permute_523);  view_731 = permute_523 = None
        view_732: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_166, [32, 128, 2048]);  mm_166 = None
        convert_element_type_1421: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_732, torch.float32);  view_732 = None
        convert_element_type_1422: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_165, torch.float32);  mm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_200: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_729, memory_format = torch.contiguous_format);  view_729 = None
        view_733: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_200, [4096, 2048]);  clone_200 = None
        permute_525: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_733, [1, 0])
        mm_167: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_525, view_352);  permute_525 = None
        mm_168: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_733, permute_527);  view_733 = permute_527 = None
        view_734: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_168, [32, 128, 2048]);  mm_168 = None
        convert_element_type_1427: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_734, torch.float32);  view_734 = None
        add_268: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1421, convert_element_type_1427);  convert_element_type_1421 = convert_element_type_1427 = None
        convert_element_type_1428: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_167, torch.float32);  mm_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_735: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_730, [4096, 2048]);  view_730 = None
        permute_529: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_735, [1, 0])
        mm_169: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_529, view_352);  permute_529 = view_352 = None
        mm_170: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_735, permute_531);  view_735 = permute_531 = None
        view_736: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_170, [32, 128, 2048]);  mm_170 = None
        convert_element_type_1433: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_736, torch.float32);  view_736 = None
        add_269: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_268, convert_element_type_1433);  add_268 = convert_element_type_1433 = None
        convert_element_type_1434: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_169, torch.float32);  mm_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_390: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_269, primals_228);  primals_228 = None
        mul_391: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_390, 2048)
        sum_125: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_390, [2], True)
        mul_392: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_390, mul_128);  mul_390 = None
        sum_126: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_392, [2], True);  mul_392 = None
        mul_393: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, sum_126);  sum_126 = None
        sub_135: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_391, sum_125);  mul_391 = sum_125 = None
        sub_136: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_135, mul_393);  sub_135 = mul_393 = None
        mul_394: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_42, sub_136);  div_42 = sub_136 = None
        mul_395: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_269, mul_128);  mul_128 = None
        sum_127: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_395, [0, 1]);  mul_395 = None
        sum_128: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_269, [0, 1]);  add_269 = None
        add_270: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_267, mul_394);  add_267 = mul_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1435: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_270, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_737: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1435, [4096, 2048]);  convert_element_type_1435 = None
        mm_171: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_737, permute_533);  permute_533 = None
        permute_534: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_737, [1, 0])
        mm_172: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_534, view_350);  permute_534 = view_350 = None
        sum_129: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_737, [0], True, dtype = torch.float32);  view_737 = None
        view_738: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_129, [2048]);  sum_129 = None
        convert_element_type_1440: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_738, torch.bfloat16);  view_738 = None
        view_739: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_171, [32, 128, 8192]);  mm_171 = None
        convert_element_type_1441: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_739, torch.float32);  view_739 = None
        convert_element_type_1442: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_172, torch.float32);  mm_172 = None
        convert_element_type_1443: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1440, torch.float32);  convert_element_type_1440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_349: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [32, 128, 8192]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_124: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_349, 0.5)
        mul_396: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1441, mul_124);  mul_124 = None
        convert_element_type_617: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_349, torch.float32)
        pow_16: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_617, 3.0)
        mul_125: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_145: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_349, mul_125);  view_349 = mul_125 = None
        mul_126: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_145, 0.7978845608028654);  add_145 = None
        tanh_15: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_126);  mul_126 = None
        add_146: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
        mul_397: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1441, add_146);  convert_element_type_1441 = add_146 = None
        convert_element_type_1444: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_397, torch.bfloat16);  mul_397 = None
        mul_398: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_15, tanh_15);  tanh_15 = None
        sub_137: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_398);  mul_398 = None
        mul_399: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_396, sub_137);  mul_396 = sub_137 = None
        mul_400: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_399, 0.7978845608028654);  mul_399 = None
        convert_element_type_1445: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_400, torch.bfloat16)
        mul_401: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_400, 0.044715);  mul_400 = None
        pow_33: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_617, 2.0);  convert_element_type_617 = None
        mul_402: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_33, 3.0);  pow_33 = None
        mul_403: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_401, mul_402);  mul_401 = mul_402 = None
        convert_element_type_1446: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_403, torch.bfloat16);  mul_403 = None
        add_271: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1445, convert_element_type_1446);  convert_element_type_1445 = convert_element_type_1446 = None
        mul_404: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1444, 0.5);  convert_element_type_1444 = None
        add_272: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_271, mul_404);  add_271 = mul_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_740: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_272, [4096, 8192]);  add_272 = None
        mm_173: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_740, permute_537);  permute_537 = None
        permute_538: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_740, [1, 0])
        mm_174: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_538, view_348);  permute_538 = view_348 = None
        sum_130: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_740, [0], True, dtype = torch.float32);  view_740 = None
        view_741: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_130, [8192]);  sum_130 = None
        convert_element_type_1451: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_741, torch.bfloat16);  view_741 = None
        view_742: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_173, [32, 128, 2048]);  mm_173 = None
        convert_element_type_1452: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_742, torch.float32);  view_742 = None
        convert_element_type_1453: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_174, torch.float32);  mm_174 = None
        convert_element_type_1454: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1451, torch.float32);  convert_element_type_1451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_406: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1452, primals_222);  primals_222 = None
        mul_407: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_406, 2048)
        sum_131: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_406, [2], True)
        mul_408: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_406, mul_122);  mul_406 = None
        sum_132: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_408, [2], True);  mul_408 = None
        mul_409: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, sum_132);  sum_132 = None
        sub_139: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_407, sum_131);  mul_407 = sum_131 = None
        sub_140: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_139, mul_409);  sub_139 = mul_409 = None
        mul_410: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_43, sub_140);  div_43 = sub_140 = None
        mul_411: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1452, mul_122);  mul_122 = None
        sum_133: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_411, [0, 1]);  mul_411 = None
        sum_134: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1452, [0, 1]);  convert_element_type_1452 = None
        add_273: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_270, mul_410);  add_270 = mul_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1455: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_273, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_743: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1455, [4096, 2048]);  convert_element_type_1455 = None
        mm_175: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_743, permute_541);  permute_541 = None
        permute_542: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_743, [1, 0])
        mm_176: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_542, view_346);  permute_542 = view_346 = None
        sum_135: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_743, [0], True, dtype = torch.float32);  view_743 = None
        view_744: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_135, [2048]);  sum_135 = None
        convert_element_type_1460: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_744, torch.bfloat16);  view_744 = None
        view_745: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_175, [32, 128, 2048]);  mm_175 = None
        convert_element_type_1461: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_176, torch.float32);  mm_176 = None
        convert_element_type_1462: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1460, torch.float32);  convert_element_type_1460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_746: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_745, [32, 128, 16, 128]);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_545: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_746, [0, 2, 1, 3]);  view_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_201: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_545, memory_format = torch.contiguous_format);  permute_545 = None
        view_747: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_201, [512, 128, 128]);  clone_201 = None
        bmm_80: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_546, view_747);  permute_546 = None
        bmm_81: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_747, permute_547);  view_747 = permute_547 = None
        view_748: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_80, [32, 16, 128, 128]);  bmm_80 = None
        view_749: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_81, [32, 16, 128, 128]);  bmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1467: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_749, torch.float32);  view_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_412: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1467, div_15);  convert_element_type_1467 = None
        sum_136: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_412, [-1], True)
        neg_9: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_15);  div_15 = None
        fma_8: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_9, sum_136, mul_412);  neg_9 = sum_136 = mul_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1468: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_34: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_219, 2, 0, 128);  primals_219 = None
        slice_35: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_34, 3, 0, 128);  slice_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_37: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_35, convert_element_type_1468, full_default_32);  slice_35 = convert_element_type_1468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_750: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_37, [512, 128, 128]);  where_37 = None
        bmm_82: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_548, view_750);  permute_548 = None
        bmm_83: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_750, permute_549);  view_750 = permute_549 = None
        view_751: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_82, [32, 16, 128, 128]);  bmm_82 = None
        view_752: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_83, [32, 16, 128, 128]);  bmm_83 = None
        convert_element_type_1474: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_751, torch.float32);  view_751 = None
        permute_550: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1474, [0, 1, 3, 2]);  convert_element_type_1474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1475: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_550, torch.bfloat16);  permute_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_15: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_752, torch.bfloat16);  view_752 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_551: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_748, [0, 2, 1, 3]);  view_748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_202: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
        view_753: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_202, [32, 128, 2048]);  clone_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_552: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1475, [0, 2, 1, 3]);  convert_element_type_1475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_754: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_552, [32, 128, 2048]);  permute_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_553: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_15, [0, 2, 1, 3]);  convert_element_type_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_203: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_553, memory_format = torch.contiguous_format);  permute_553 = None
        view_755: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_203, [32, 128, 2048]);  clone_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_756: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_753, [4096, 2048]);  view_753 = None
        permute_554: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_756, [1, 0])
        mm_177: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_554, view_330);  permute_554 = None
        mm_178: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_756, permute_556);  view_756 = permute_556 = None
        view_757: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_178, [32, 128, 2048]);  mm_178 = None
        convert_element_type_1481: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_757, torch.float32);  view_757 = None
        convert_element_type_1482: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_177, torch.float32);  mm_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_204: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_754, memory_format = torch.contiguous_format);  view_754 = None
        view_758: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_204, [4096, 2048]);  clone_204 = None
        permute_558: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_758, [1, 0])
        mm_179: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_558, view_330);  permute_558 = None
        mm_180: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_758, permute_560);  view_758 = permute_560 = None
        view_759: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_180, [32, 128, 2048]);  mm_180 = None
        convert_element_type_1487: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_759, torch.float32);  view_759 = None
        add_274: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1481, convert_element_type_1487);  convert_element_type_1481 = convert_element_type_1487 = None
        convert_element_type_1488: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_179, torch.float32);  mm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_760: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_755, [4096, 2048]);  view_755 = None
        permute_562: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_760, [1, 0])
        mm_181: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_562, view_330);  permute_562 = view_330 = None
        mm_182: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_760, permute_564);  view_760 = permute_564 = None
        view_761: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_182, [32, 128, 2048]);  mm_182 = None
        convert_element_type_1493: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_761, torch.float32);  view_761 = None
        add_275: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_274, convert_element_type_1493);  add_274 = convert_element_type_1493 = None
        convert_element_type_1494: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_181, torch.float32);  mm_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_414: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_275, primals_214);  primals_214 = None
        mul_415: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_414, 2048)
        sum_137: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_414, [2], True)
        mul_416: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_414, mul_120);  mul_414 = None
        sum_138: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_416, [2], True);  mul_416 = None
        mul_417: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, sum_138);  sum_138 = None
        sub_142: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_415, sum_137);  mul_415 = sum_137 = None
        sub_143: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_142, mul_417);  sub_142 = mul_417 = None
        mul_418: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_44, sub_143);  div_44 = sub_143 = None
        mul_419: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_275, mul_120);  mul_120 = None
        sum_139: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_419, [0, 1]);  mul_419 = None
        sum_140: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_275, [0, 1]);  add_275 = None
        add_276: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_273, mul_418);  add_273 = mul_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1495: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_276, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_762: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1495, [4096, 2048]);  convert_element_type_1495 = None
        mm_183: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_762, permute_566);  permute_566 = None
        permute_567: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_762, [1, 0])
        mm_184: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_567, view_328);  permute_567 = view_328 = None
        sum_141: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_762, [0], True, dtype = torch.float32);  view_762 = None
        view_763: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_141, [2048]);  sum_141 = None
        convert_element_type_1500: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_763, torch.bfloat16);  view_763 = None
        view_764: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_183, [32, 128, 8192]);  mm_183 = None
        convert_element_type_1501: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_764, torch.float32);  view_764 = None
        convert_element_type_1502: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_184, torch.float32);  mm_184 = None
        convert_element_type_1503: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1500, torch.float32);  convert_element_type_1500 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_327: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [32, 128, 8192]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_327, 0.5)
        mul_420: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1501, mul_116);  mul_116 = None
        convert_element_type_578: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_327, torch.float32)
        pow_15: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_578, 3.0)
        mul_117: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_136: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_327, mul_117);  view_327 = mul_117 = None
        mul_118: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_136, 0.7978845608028654);  add_136 = None
        tanh_14: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_137: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
        mul_421: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1501, add_137);  convert_element_type_1501 = add_137 = None
        convert_element_type_1504: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_421, torch.bfloat16);  mul_421 = None
        mul_422: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_14, tanh_14);  tanh_14 = None
        sub_144: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_422);  mul_422 = None
        mul_423: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_420, sub_144);  mul_420 = sub_144 = None
        mul_424: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_423, 0.7978845608028654);  mul_423 = None
        convert_element_type_1505: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_424, torch.bfloat16)
        mul_425: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_424, 0.044715);  mul_424 = None
        pow_34: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_578, 2.0);  convert_element_type_578 = None
        mul_426: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_34, 3.0);  pow_34 = None
        mul_427: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_425, mul_426);  mul_425 = mul_426 = None
        convert_element_type_1506: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_427, torch.bfloat16);  mul_427 = None
        add_277: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1505, convert_element_type_1506);  convert_element_type_1505 = convert_element_type_1506 = None
        mul_428: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1504, 0.5);  convert_element_type_1504 = None
        add_278: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_277, mul_428);  add_277 = mul_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_765: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_278, [4096, 8192]);  add_278 = None
        mm_185: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_765, permute_570);  permute_570 = None
        permute_571: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_765, [1, 0])
        mm_186: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_571, view_326);  permute_571 = view_326 = None
        sum_142: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_765, [0], True, dtype = torch.float32);  view_765 = None
        view_766: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_142, [8192]);  sum_142 = None
        convert_element_type_1511: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_766, torch.bfloat16);  view_766 = None
        view_767: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_185, [32, 128, 2048]);  mm_185 = None
        convert_element_type_1512: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_767, torch.float32);  view_767 = None
        convert_element_type_1513: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_186, torch.float32);  mm_186 = None
        convert_element_type_1514: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1511, torch.float32);  convert_element_type_1511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_430: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1512, primals_208);  primals_208 = None
        mul_431: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_430, 2048)
        sum_143: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_430, [2], True)
        mul_432: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_430, mul_114);  mul_430 = None
        sum_144: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_432, [2], True);  mul_432 = None
        mul_433: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_114, sum_144);  sum_144 = None
        sub_146: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_431, sum_143);  mul_431 = sum_143 = None
        sub_147: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_146, mul_433);  sub_146 = mul_433 = None
        mul_434: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_45, sub_147);  div_45 = sub_147 = None
        mul_435: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1512, mul_114);  mul_114 = None
        sum_145: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_435, [0, 1]);  mul_435 = None
        sum_146: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1512, [0, 1]);  convert_element_type_1512 = None
        add_279: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_276, mul_434);  add_276 = mul_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1515: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_279, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_768: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1515, [4096, 2048]);  convert_element_type_1515 = None
        mm_187: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_768, permute_574);  permute_574 = None
        permute_575: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_768, [1, 0])
        mm_188: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_575, view_324);  permute_575 = view_324 = None
        sum_147: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_768, [0], True, dtype = torch.float32);  view_768 = None
        view_769: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [2048]);  sum_147 = None
        convert_element_type_1520: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_769, torch.bfloat16);  view_769 = None
        view_770: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_187, [32, 128, 2048]);  mm_187 = None
        convert_element_type_1521: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_188, torch.float32);  mm_188 = None
        convert_element_type_1522: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1520, torch.float32);  convert_element_type_1520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_771: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_770, [32, 128, 16, 128]);  view_770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_578: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_771, [0, 2, 1, 3]);  view_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_205: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_578, memory_format = torch.contiguous_format);  permute_578 = None
        view_772: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_205, [512, 128, 128]);  clone_205 = None
        bmm_84: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_579, view_772);  permute_579 = None
        bmm_85: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_772, permute_580);  view_772 = permute_580 = None
        view_773: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_84, [32, 16, 128, 128]);  bmm_84 = None
        view_774: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_85, [32, 16, 128, 128]);  bmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1527: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_774, torch.float32);  view_774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_436: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1527, div_14);  convert_element_type_1527 = None
        sum_148: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_436, [-1], True)
        neg_10: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_14);  div_14 = None
        fma_9: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_10, sum_148, mul_436);  neg_10 = sum_148 = mul_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1528: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_32: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_205, 2, 0, 128);  primals_205 = None
        slice_33: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_32, 3, 0, 128);  slice_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_38: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_33, convert_element_type_1528, full_default_32);  slice_33 = convert_element_type_1528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_775: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_38, [512, 128, 128]);  where_38 = None
        bmm_86: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_581, view_775);  permute_581 = None
        bmm_87: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_775, permute_582);  view_775 = permute_582 = None
        view_776: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_86, [32, 16, 128, 128]);  bmm_86 = None
        view_777: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_87, [32, 16, 128, 128]);  bmm_87 = None
        convert_element_type_1534: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_776, torch.float32);  view_776 = None
        permute_583: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1534, [0, 1, 3, 2]);  convert_element_type_1534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1535: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_583, torch.bfloat16);  permute_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_14: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_777, torch.bfloat16);  view_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_584: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_773, [0, 2, 1, 3]);  view_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_206: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_584, memory_format = torch.contiguous_format);  permute_584 = None
        view_778: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_206, [32, 128, 2048]);  clone_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_585: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1535, [0, 2, 1, 3]);  convert_element_type_1535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_779: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_585, [32, 128, 2048]);  permute_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_586: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_14, [0, 2, 1, 3]);  convert_element_type_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_207: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_586, memory_format = torch.contiguous_format);  permute_586 = None
        view_780: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_207, [32, 128, 2048]);  clone_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_781: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_778, [4096, 2048]);  view_778 = None
        permute_587: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_781, [1, 0])
        mm_189: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_587, view_308);  permute_587 = None
        mm_190: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_781, permute_589);  view_781 = permute_589 = None
        view_782: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_190, [32, 128, 2048]);  mm_190 = None
        convert_element_type_1541: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_782, torch.float32);  view_782 = None
        convert_element_type_1542: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_189, torch.float32);  mm_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_208: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_779, memory_format = torch.contiguous_format);  view_779 = None
        view_783: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_208, [4096, 2048]);  clone_208 = None
        permute_591: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_783, [1, 0])
        mm_191: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_591, view_308);  permute_591 = None
        mm_192: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_783, permute_593);  view_783 = permute_593 = None
        view_784: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_192, [32, 128, 2048]);  mm_192 = None
        convert_element_type_1547: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_784, torch.float32);  view_784 = None
        add_280: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1541, convert_element_type_1547);  convert_element_type_1541 = convert_element_type_1547 = None
        convert_element_type_1548: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_191, torch.float32);  mm_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_785: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_780, [4096, 2048]);  view_780 = None
        permute_595: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_785, [1, 0])
        mm_193: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_595, view_308);  permute_595 = view_308 = None
        mm_194: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_785, permute_597);  view_785 = permute_597 = None
        view_786: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_194, [32, 128, 2048]);  mm_194 = None
        convert_element_type_1553: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_786, torch.float32);  view_786 = None
        add_281: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_280, convert_element_type_1553);  add_280 = convert_element_type_1553 = None
        convert_element_type_1554: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_193, torch.float32);  mm_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_438: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_281, primals_200);  primals_200 = None
        mul_439: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_438, 2048)
        sum_149: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_438, [2], True)
        mul_440: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_438, mul_112);  mul_438 = None
        sum_150: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_440, [2], True);  mul_440 = None
        mul_441: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, sum_150);  sum_150 = None
        sub_149: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_439, sum_149);  mul_439 = sum_149 = None
        sub_150: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_149, mul_441);  sub_149 = mul_441 = None
        mul_442: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_46, sub_150);  div_46 = sub_150 = None
        mul_443: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_281, mul_112);  mul_112 = None
        sum_151: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_443, [0, 1]);  mul_443 = None
        sum_152: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_281, [0, 1]);  add_281 = None
        add_282: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_279, mul_442);  add_279 = mul_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1555: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_282, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_787: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1555, [4096, 2048]);  convert_element_type_1555 = None
        mm_195: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_787, permute_599);  permute_599 = None
        permute_600: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_787, [1, 0])
        mm_196: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_600, view_306);  permute_600 = view_306 = None
        sum_153: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_787, [0], True, dtype = torch.float32);  view_787 = None
        view_788: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_153, [2048]);  sum_153 = None
        convert_element_type_1560: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_788, torch.bfloat16);  view_788 = None
        view_789: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_195, [32, 128, 8192]);  mm_195 = None
        convert_element_type_1561: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_789, torch.float32);  view_789 = None
        convert_element_type_1562: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_196, torch.float32);  mm_196 = None
        convert_element_type_1563: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1560, torch.float32);  convert_element_type_1560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_305: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [32, 128, 8192]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_108: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_305, 0.5)
        mul_444: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1561, mul_108);  mul_108 = None
        convert_element_type_539: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_305, torch.float32)
        pow_14: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_539, 3.0)
        mul_109: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_127: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_305, mul_109);  view_305 = mul_109 = None
        mul_110: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_127, 0.7978845608028654);  add_127 = None
        tanh_13: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_110);  mul_110 = None
        add_128: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
        mul_445: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1561, add_128);  convert_element_type_1561 = add_128 = None
        convert_element_type_1564: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_445, torch.bfloat16);  mul_445 = None
        mul_446: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
        sub_151: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_446);  mul_446 = None
        mul_447: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_444, sub_151);  mul_444 = sub_151 = None
        mul_448: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_447, 0.7978845608028654);  mul_447 = None
        convert_element_type_1565: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_448, torch.bfloat16)
        mul_449: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_448, 0.044715);  mul_448 = None
        pow_35: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_539, 2.0);  convert_element_type_539 = None
        mul_450: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_35, 3.0);  pow_35 = None
        mul_451: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
        convert_element_type_1566: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_451, torch.bfloat16);  mul_451 = None
        add_283: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1565, convert_element_type_1566);  convert_element_type_1565 = convert_element_type_1566 = None
        mul_452: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1564, 0.5);  convert_element_type_1564 = None
        add_284: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_283, mul_452);  add_283 = mul_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_790: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_284, [4096, 8192]);  add_284 = None
        mm_197: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_790, permute_603);  permute_603 = None
        permute_604: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_790, [1, 0])
        mm_198: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_604, view_304);  permute_604 = view_304 = None
        sum_154: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_790, [0], True, dtype = torch.float32);  view_790 = None
        view_791: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_154, [8192]);  sum_154 = None
        convert_element_type_1571: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_791, torch.bfloat16);  view_791 = None
        view_792: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_197, [32, 128, 2048]);  mm_197 = None
        convert_element_type_1572: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_792, torch.float32);  view_792 = None
        convert_element_type_1573: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_198, torch.float32);  mm_198 = None
        convert_element_type_1574: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1571, torch.float32);  convert_element_type_1571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_454: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1572, primals_194);  primals_194 = None
        mul_455: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_454, 2048)
        sum_155: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_454, [2], True)
        mul_456: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_454, mul_106);  mul_454 = None
        sum_156: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_456, [2], True);  mul_456 = None
        mul_457: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_106, sum_156);  sum_156 = None
        sub_153: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_455, sum_155);  mul_455 = sum_155 = None
        sub_154: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_153, mul_457);  sub_153 = mul_457 = None
        mul_458: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_47, sub_154);  div_47 = sub_154 = None
        mul_459: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1572, mul_106);  mul_106 = None
        sum_157: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_459, [0, 1]);  mul_459 = None
        sum_158: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1572, [0, 1]);  convert_element_type_1572 = None
        add_285: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_282, mul_458);  add_282 = mul_458 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1575: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_285, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_793: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1575, [4096, 2048]);  convert_element_type_1575 = None
        mm_199: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_793, permute_607);  permute_607 = None
        permute_608: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_793, [1, 0])
        mm_200: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_608, view_302);  permute_608 = view_302 = None
        sum_159: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_793, [0], True, dtype = torch.float32);  view_793 = None
        view_794: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_159, [2048]);  sum_159 = None
        convert_element_type_1580: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_794, torch.bfloat16);  view_794 = None
        view_795: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_199, [32, 128, 2048]);  mm_199 = None
        convert_element_type_1581: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_200, torch.float32);  mm_200 = None
        convert_element_type_1582: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1580, torch.float32);  convert_element_type_1580 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_796: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_795, [32, 128, 16, 128]);  view_795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_611: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_796, [0, 2, 1, 3]);  view_796 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_209: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_611, memory_format = torch.contiguous_format);  permute_611 = None
        view_797: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_209, [512, 128, 128]);  clone_209 = None
        bmm_88: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_612, view_797);  permute_612 = None
        bmm_89: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_797, permute_613);  view_797 = permute_613 = None
        view_798: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_88, [32, 16, 128, 128]);  bmm_88 = None
        view_799: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_89, [32, 16, 128, 128]);  bmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1587: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_799, torch.float32);  view_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_460: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1587, div_13);  convert_element_type_1587 = None
        sum_160: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_460, [-1], True)
        neg_11: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_10: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_11, sum_160, mul_460);  neg_11 = sum_160 = mul_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1588: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_30: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_191, 2, 0, 128);  primals_191 = None
        slice_31: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_30, 3, 0, 128);  slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_39: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_31, convert_element_type_1588, full_default_32);  slice_31 = convert_element_type_1588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_800: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_39, [512, 128, 128]);  where_39 = None
        bmm_90: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_614, view_800);  permute_614 = None
        bmm_91: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_800, permute_615);  view_800 = permute_615 = None
        view_801: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_90, [32, 16, 128, 128]);  bmm_90 = None
        view_802: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_91, [32, 16, 128, 128]);  bmm_91 = None
        convert_element_type_1594: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_801, torch.float32);  view_801 = None
        permute_616: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1594, [0, 1, 3, 2]);  convert_element_type_1594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1595: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_616, torch.bfloat16);  permute_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_13: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_802, torch.bfloat16);  view_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_617: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_798, [0, 2, 1, 3]);  view_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_210: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_617, memory_format = torch.contiguous_format);  permute_617 = None
        view_803: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_210, [32, 128, 2048]);  clone_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_618: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1595, [0, 2, 1, 3]);  convert_element_type_1595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_804: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_618, [32, 128, 2048]);  permute_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_619: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_13, [0, 2, 1, 3]);  convert_element_type_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_211: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_619, memory_format = torch.contiguous_format);  permute_619 = None
        view_805: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_211, [32, 128, 2048]);  clone_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_806: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_803, [4096, 2048]);  view_803 = None
        permute_620: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_806, [1, 0])
        mm_201: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_620, view_286);  permute_620 = None
        mm_202: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_806, permute_622);  view_806 = permute_622 = None
        view_807: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_202, [32, 128, 2048]);  mm_202 = None
        convert_element_type_1601: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_807, torch.float32);  view_807 = None
        convert_element_type_1602: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_201, torch.float32);  mm_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_212: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_804, memory_format = torch.contiguous_format);  view_804 = None
        view_808: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_212, [4096, 2048]);  clone_212 = None
        permute_624: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_808, [1, 0])
        mm_203: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_624, view_286);  permute_624 = None
        mm_204: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_808, permute_626);  view_808 = permute_626 = None
        view_809: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_204, [32, 128, 2048]);  mm_204 = None
        convert_element_type_1607: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_809, torch.float32);  view_809 = None
        add_286: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1601, convert_element_type_1607);  convert_element_type_1601 = convert_element_type_1607 = None
        convert_element_type_1608: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_203, torch.float32);  mm_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_810: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_805, [4096, 2048]);  view_805 = None
        permute_628: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_810, [1, 0])
        mm_205: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_628, view_286);  permute_628 = view_286 = None
        mm_206: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_810, permute_630);  view_810 = permute_630 = None
        view_811: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_206, [32, 128, 2048]);  mm_206 = None
        convert_element_type_1613: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_811, torch.float32);  view_811 = None
        add_287: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_286, convert_element_type_1613);  add_286 = convert_element_type_1613 = None
        convert_element_type_1614: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_205, torch.float32);  mm_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_462: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_287, primals_186);  primals_186 = None
        mul_463: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_462, 2048)
        sum_161: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_462, [2], True)
        mul_464: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_462, mul_104);  mul_462 = None
        sum_162: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True);  mul_464 = None
        mul_465: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, sum_162);  sum_162 = None
        sub_156: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_463, sum_161);  mul_463 = sum_161 = None
        sub_157: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_156, mul_465);  sub_156 = mul_465 = None
        mul_466: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_48, sub_157);  div_48 = sub_157 = None
        mul_467: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_287, mul_104);  mul_104 = None
        sum_163: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 1]);  mul_467 = None
        sum_164: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_287, [0, 1]);  add_287 = None
        add_288: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_285, mul_466);  add_285 = mul_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1615: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_288, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_812: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1615, [4096, 2048]);  convert_element_type_1615 = None
        mm_207: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_812, permute_632);  permute_632 = None
        permute_633: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_812, [1, 0])
        mm_208: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_633, view_284);  permute_633 = view_284 = None
        sum_165: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_812, [0], True, dtype = torch.float32);  view_812 = None
        view_813: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_165, [2048]);  sum_165 = None
        convert_element_type_1620: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_813, torch.bfloat16);  view_813 = None
        view_814: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_207, [32, 128, 8192]);  mm_207 = None
        convert_element_type_1621: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_814, torch.float32);  view_814 = None
        convert_element_type_1622: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_208, torch.float32);  mm_208 = None
        convert_element_type_1623: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1620, torch.float32);  convert_element_type_1620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_283: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [32, 128, 8192]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_100: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_283, 0.5)
        mul_468: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1621, mul_100);  mul_100 = None
        convert_element_type_500: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_283, torch.float32)
        pow_13: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_500, 3.0)
        mul_101: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_118: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_283, mul_101);  view_283 = mul_101 = None
        mul_102: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_118, 0.7978845608028654);  add_118 = None
        tanh_12: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_102);  mul_102 = None
        add_119: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
        mul_469: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1621, add_119);  convert_element_type_1621 = add_119 = None
        convert_element_type_1624: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_469, torch.bfloat16);  mul_469 = None
        mul_470: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
        sub_158: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_470);  mul_470 = None
        mul_471: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_468, sub_158);  mul_468 = sub_158 = None
        mul_472: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_471, 0.7978845608028654);  mul_471 = None
        convert_element_type_1625: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_472, torch.bfloat16)
        mul_473: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_472, 0.044715);  mul_472 = None
        pow_36: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_500, 2.0);  convert_element_type_500 = None
        mul_474: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_36, 3.0);  pow_36 = None
        mul_475: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
        convert_element_type_1626: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_475, torch.bfloat16);  mul_475 = None
        add_289: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1625, convert_element_type_1626);  convert_element_type_1625 = convert_element_type_1626 = None
        mul_476: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1624, 0.5);  convert_element_type_1624 = None
        add_290: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_289, mul_476);  add_289 = mul_476 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_815: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_290, [4096, 8192]);  add_290 = None
        mm_209: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_815, permute_636);  permute_636 = None
        permute_637: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_815, [1, 0])
        mm_210: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_637, view_282);  permute_637 = view_282 = None
        sum_166: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_815, [0], True, dtype = torch.float32);  view_815 = None
        view_816: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_166, [8192]);  sum_166 = None
        convert_element_type_1631: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_816, torch.bfloat16);  view_816 = None
        view_817: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_209, [32, 128, 2048]);  mm_209 = None
        convert_element_type_1632: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_817, torch.float32);  view_817 = None
        convert_element_type_1633: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_210, torch.float32);  mm_210 = None
        convert_element_type_1634: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1631, torch.float32);  convert_element_type_1631 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_478: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1632, primals_180);  primals_180 = None
        mul_479: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_478, 2048)
        sum_167: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_478, [2], True)
        mul_480: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_478, mul_98);  mul_478 = None
        sum_168: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_480, [2], True);  mul_480 = None
        mul_481: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, sum_168);  sum_168 = None
        sub_160: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_479, sum_167);  mul_479 = sum_167 = None
        sub_161: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_160, mul_481);  sub_160 = mul_481 = None
        mul_482: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_49, sub_161);  div_49 = sub_161 = None
        mul_483: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1632, mul_98);  mul_98 = None
        sum_169: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_483, [0, 1]);  mul_483 = None
        sum_170: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1632, [0, 1]);  convert_element_type_1632 = None
        add_291: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_288, mul_482);  add_288 = mul_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1635: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_291, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_818: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1635, [4096, 2048]);  convert_element_type_1635 = None
        mm_211: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_818, permute_640);  permute_640 = None
        permute_641: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_818, [1, 0])
        mm_212: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_641, view_280);  permute_641 = view_280 = None
        sum_171: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_818, [0], True, dtype = torch.float32);  view_818 = None
        view_819: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_171, [2048]);  sum_171 = None
        convert_element_type_1640: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_819, torch.bfloat16);  view_819 = None
        view_820: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_211, [32, 128, 2048]);  mm_211 = None
        convert_element_type_1641: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_212, torch.float32);  mm_212 = None
        convert_element_type_1642: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1640, torch.float32);  convert_element_type_1640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_821: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_820, [32, 128, 16, 128]);  view_820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_644: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_821, [0, 2, 1, 3]);  view_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_213: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_644, memory_format = torch.contiguous_format);  permute_644 = None
        view_822: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_213, [512, 128, 128]);  clone_213 = None
        bmm_92: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_645, view_822);  permute_645 = None
        bmm_93: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_822, permute_646);  view_822 = permute_646 = None
        view_823: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_92, [32, 16, 128, 128]);  bmm_92 = None
        view_824: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_93, [32, 16, 128, 128]);  bmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1647: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_824, torch.float32);  view_824 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_484: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1647, div_12);  convert_element_type_1647 = None
        sum_172: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_484, [-1], True)
        neg_12: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_12);  div_12 = None
        fma_11: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_12, sum_172, mul_484);  neg_12 = sum_172 = mul_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1648: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_28: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_177, 2, 0, 128);  primals_177 = None
        slice_29: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_28, 3, 0, 128);  slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_40: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_29, convert_element_type_1648, full_default_32);  slice_29 = convert_element_type_1648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_825: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_40, [512, 128, 128]);  where_40 = None
        bmm_94: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_647, view_825);  permute_647 = None
        bmm_95: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_825, permute_648);  view_825 = permute_648 = None
        view_826: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_94, [32, 16, 128, 128]);  bmm_94 = None
        view_827: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_95, [32, 16, 128, 128]);  bmm_95 = None
        convert_element_type_1654: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_826, torch.float32);  view_826 = None
        permute_649: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1654, [0, 1, 3, 2]);  convert_element_type_1654 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1655: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_649, torch.bfloat16);  permute_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_12: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_827, torch.bfloat16);  view_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_650: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_823, [0, 2, 1, 3]);  view_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_214: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_650, memory_format = torch.contiguous_format);  permute_650 = None
        view_828: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_214, [32, 128, 2048]);  clone_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_651: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1655, [0, 2, 1, 3]);  convert_element_type_1655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_829: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_651, [32, 128, 2048]);  permute_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_652: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_12, [0, 2, 1, 3]);  convert_element_type_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_215: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_652, memory_format = torch.contiguous_format);  permute_652 = None
        view_830: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_215, [32, 128, 2048]);  clone_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_831: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_828, [4096, 2048]);  view_828 = None
        permute_653: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_831, [1, 0])
        mm_213: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_653, view_264);  permute_653 = None
        mm_214: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_831, permute_655);  view_831 = permute_655 = None
        view_832: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_214, [32, 128, 2048]);  mm_214 = None
        convert_element_type_1661: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_832, torch.float32);  view_832 = None
        convert_element_type_1662: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_213, torch.float32);  mm_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_216: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_829, memory_format = torch.contiguous_format);  view_829 = None
        view_833: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_216, [4096, 2048]);  clone_216 = None
        permute_657: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_833, [1, 0])
        mm_215: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_657, view_264);  permute_657 = None
        mm_216: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_833, permute_659);  view_833 = permute_659 = None
        view_834: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_216, [32, 128, 2048]);  mm_216 = None
        convert_element_type_1667: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_834, torch.float32);  view_834 = None
        add_292: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1661, convert_element_type_1667);  convert_element_type_1661 = convert_element_type_1667 = None
        convert_element_type_1668: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_215, torch.float32);  mm_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_835: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_830, [4096, 2048]);  view_830 = None
        permute_661: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_835, [1, 0])
        mm_217: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_661, view_264);  permute_661 = view_264 = None
        mm_218: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_835, permute_663);  view_835 = permute_663 = None
        view_836: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_218, [32, 128, 2048]);  mm_218 = None
        convert_element_type_1673: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_836, torch.float32);  view_836 = None
        add_293: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_292, convert_element_type_1673);  add_292 = convert_element_type_1673 = None
        convert_element_type_1674: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_217, torch.float32);  mm_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_486: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_293, primals_172);  primals_172 = None
        mul_487: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_486, 2048)
        sum_173: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_486, [2], True)
        mul_488: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_486, mul_96);  mul_486 = None
        sum_174: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_488, [2], True);  mul_488 = None
        mul_489: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, sum_174);  sum_174 = None
        sub_163: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_487, sum_173);  mul_487 = sum_173 = None
        sub_164: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_163, mul_489);  sub_163 = mul_489 = None
        mul_490: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_50, sub_164);  div_50 = sub_164 = None
        mul_491: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_293, mul_96);  mul_96 = None
        sum_175: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 1]);  mul_491 = None
        sum_176: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_293, [0, 1]);  add_293 = None
        add_294: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_291, mul_490);  add_291 = mul_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1675: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_294, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_837: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1675, [4096, 2048]);  convert_element_type_1675 = None
        mm_219: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_837, permute_665);  permute_665 = None
        permute_666: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_837, [1, 0])
        mm_220: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_666, view_262);  permute_666 = view_262 = None
        sum_177: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_837, [0], True, dtype = torch.float32);  view_837 = None
        view_838: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_177, [2048]);  sum_177 = None
        convert_element_type_1680: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_838, torch.bfloat16);  view_838 = None
        view_839: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_219, [32, 128, 8192]);  mm_219 = None
        convert_element_type_1681: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_839, torch.float32);  view_839 = None
        convert_element_type_1682: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_220, torch.float32);  mm_220 = None
        convert_element_type_1683: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1680, torch.float32);  convert_element_type_1680 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_261: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [32, 128, 8192]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_492: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1681, mul_92);  mul_92 = None
        convert_element_type_461: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_261, torch.float32)
        pow_12: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_461, 3.0)
        mul_93: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_109: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_261, mul_93);  view_261 = mul_93 = None
        mul_94: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_109, 0.7978845608028654);  add_109 = None
        tanh_11: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_110: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_493: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1681, add_110);  convert_element_type_1681 = add_110 = None
        convert_element_type_1684: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_493, torch.bfloat16);  mul_493 = None
        mul_494: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_165: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_494);  mul_494 = None
        mul_495: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_492, sub_165);  mul_492 = sub_165 = None
        mul_496: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_495, 0.7978845608028654);  mul_495 = None
        convert_element_type_1685: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_496, torch.bfloat16)
        mul_497: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_496, 0.044715);  mul_496 = None
        pow_37: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_461, 2.0);  convert_element_type_461 = None
        mul_498: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_37, 3.0);  pow_37 = None
        mul_499: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_497, mul_498);  mul_497 = mul_498 = None
        convert_element_type_1686: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_499, torch.bfloat16);  mul_499 = None
        add_295: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1685, convert_element_type_1686);  convert_element_type_1685 = convert_element_type_1686 = None
        mul_500: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1684, 0.5);  convert_element_type_1684 = None
        add_296: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_295, mul_500);  add_295 = mul_500 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_840: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_296, [4096, 8192]);  add_296 = None
        mm_221: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_840, permute_669);  permute_669 = None
        permute_670: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_840, [1, 0])
        mm_222: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_670, view_260);  permute_670 = view_260 = None
        sum_178: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_840, [0], True, dtype = torch.float32);  view_840 = None
        view_841: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_178, [8192]);  sum_178 = None
        convert_element_type_1691: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_841, torch.bfloat16);  view_841 = None
        view_842: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_221, [32, 128, 2048]);  mm_221 = None
        convert_element_type_1692: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_842, torch.float32);  view_842 = None
        convert_element_type_1693: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_222, torch.float32);  mm_222 = None
        convert_element_type_1694: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1691, torch.float32);  convert_element_type_1691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_502: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1692, primals_166);  primals_166 = None
        mul_503: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_502, 2048)
        sum_179: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_502, [2], True)
        mul_504: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_502, mul_90);  mul_502 = None
        sum_180: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_504, [2], True);  mul_504 = None
        mul_505: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, sum_180);  sum_180 = None
        sub_167: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_503, sum_179);  mul_503 = sum_179 = None
        sub_168: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_167, mul_505);  sub_167 = mul_505 = None
        mul_506: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_51, sub_168);  div_51 = sub_168 = None
        mul_507: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1692, mul_90);  mul_90 = None
        sum_181: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_507, [0, 1]);  mul_507 = None
        sum_182: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1692, [0, 1]);  convert_element_type_1692 = None
        add_297: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_294, mul_506);  add_294 = mul_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1695: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_297, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_843: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1695, [4096, 2048]);  convert_element_type_1695 = None
        mm_223: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_843, permute_673);  permute_673 = None
        permute_674: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_843, [1, 0])
        mm_224: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_674, view_258);  permute_674 = view_258 = None
        sum_183: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_843, [0], True, dtype = torch.float32);  view_843 = None
        view_844: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_183, [2048]);  sum_183 = None
        convert_element_type_1700: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_844, torch.bfloat16);  view_844 = None
        view_845: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_223, [32, 128, 2048]);  mm_223 = None
        convert_element_type_1701: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_224, torch.float32);  mm_224 = None
        convert_element_type_1702: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1700, torch.float32);  convert_element_type_1700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_846: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_845, [32, 128, 16, 128]);  view_845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_677: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_846, [0, 2, 1, 3]);  view_846 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_217: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_677, memory_format = torch.contiguous_format);  permute_677 = None
        view_847: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_217, [512, 128, 128]);  clone_217 = None
        bmm_96: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_678, view_847);  permute_678 = None
        bmm_97: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_847, permute_679);  view_847 = permute_679 = None
        view_848: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_96, [32, 16, 128, 128]);  bmm_96 = None
        view_849: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_97, [32, 16, 128, 128]);  bmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1707: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_849, torch.float32);  view_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_508: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1707, div_11);  convert_element_type_1707 = None
        sum_184: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_508, [-1], True)
        neg_13: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_11);  div_11 = None
        fma_12: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_13, sum_184, mul_508);  neg_13 = sum_184 = mul_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1708: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_12, torch.bfloat16);  fma_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_26: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_163, 2, 0, 128);  primals_163 = None
        slice_27: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_26, 3, 0, 128);  slice_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_41: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_27, convert_element_type_1708, full_default_32);  slice_27 = convert_element_type_1708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_850: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_41, [512, 128, 128]);  where_41 = None
        bmm_98: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_680, view_850);  permute_680 = None
        bmm_99: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_850, permute_681);  view_850 = permute_681 = None
        view_851: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_98, [32, 16, 128, 128]);  bmm_98 = None
        view_852: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_99, [32, 16, 128, 128]);  bmm_99 = None
        convert_element_type_1714: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_851, torch.float32);  view_851 = None
        permute_682: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1714, [0, 1, 3, 2]);  convert_element_type_1714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1715: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_682, torch.bfloat16);  permute_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_11: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_852, torch.bfloat16);  view_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_683: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_848, [0, 2, 1, 3]);  view_848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_218: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_683, memory_format = torch.contiguous_format);  permute_683 = None
        view_853: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_218, [32, 128, 2048]);  clone_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_684: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1715, [0, 2, 1, 3]);  convert_element_type_1715 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_854: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_684, [32, 128, 2048]);  permute_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_685: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_11, [0, 2, 1, 3]);  convert_element_type_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_219: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_685, memory_format = torch.contiguous_format);  permute_685 = None
        view_855: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_219, [32, 128, 2048]);  clone_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_856: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_853, [4096, 2048]);  view_853 = None
        permute_686: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_856, [1, 0])
        mm_225: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_686, view_242);  permute_686 = None
        mm_226: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_856, permute_688);  view_856 = permute_688 = None
        view_857: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_226, [32, 128, 2048]);  mm_226 = None
        convert_element_type_1721: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_857, torch.float32);  view_857 = None
        convert_element_type_1722: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_225, torch.float32);  mm_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_220: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_854, memory_format = torch.contiguous_format);  view_854 = None
        view_858: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_220, [4096, 2048]);  clone_220 = None
        permute_690: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_858, [1, 0])
        mm_227: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_690, view_242);  permute_690 = None
        mm_228: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_858, permute_692);  view_858 = permute_692 = None
        view_859: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_228, [32, 128, 2048]);  mm_228 = None
        convert_element_type_1727: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_859, torch.float32);  view_859 = None
        add_298: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1721, convert_element_type_1727);  convert_element_type_1721 = convert_element_type_1727 = None
        convert_element_type_1728: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_227, torch.float32);  mm_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_860: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_855, [4096, 2048]);  view_855 = None
        permute_694: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_860, [1, 0])
        mm_229: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_694, view_242);  permute_694 = view_242 = None
        mm_230: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_860, permute_696);  view_860 = permute_696 = None
        view_861: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_230, [32, 128, 2048]);  mm_230 = None
        convert_element_type_1733: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_861, torch.float32);  view_861 = None
        add_299: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_298, convert_element_type_1733);  add_298 = convert_element_type_1733 = None
        convert_element_type_1734: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_229, torch.float32);  mm_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_510: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_299, primals_158);  primals_158 = None
        mul_511: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_510, 2048)
        sum_185: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_510, [2], True)
        mul_512: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_510, mul_88);  mul_510 = None
        sum_186: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_512, [2], True);  mul_512 = None
        mul_513: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, sum_186);  sum_186 = None
        sub_170: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_511, sum_185);  mul_511 = sum_185 = None
        sub_171: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_170, mul_513);  sub_170 = mul_513 = None
        mul_514: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_52, sub_171);  div_52 = sub_171 = None
        mul_515: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_299, mul_88);  mul_88 = None
        sum_187: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_515, [0, 1]);  mul_515 = None
        sum_188: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_299, [0, 1]);  add_299 = None
        add_300: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_297, mul_514);  add_297 = mul_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1735: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_300, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_862: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1735, [4096, 2048]);  convert_element_type_1735 = None
        mm_231: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_862, permute_698);  permute_698 = None
        permute_699: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_862, [1, 0])
        mm_232: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_699, view_240);  permute_699 = view_240 = None
        sum_189: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_862, [0], True, dtype = torch.float32);  view_862 = None
        view_863: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_189, [2048]);  sum_189 = None
        convert_element_type_1740: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_863, torch.bfloat16);  view_863 = None
        view_864: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_231, [32, 128, 8192]);  mm_231 = None
        convert_element_type_1741: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_864, torch.float32);  view_864 = None
        convert_element_type_1742: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_232, torch.float32);  mm_232 = None
        convert_element_type_1743: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1740, torch.float32);  convert_element_type_1740 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_239: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [32, 128, 8192]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_84: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_516: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1741, mul_84);  mul_84 = None
        convert_element_type_422: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32)
        pow_11: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_422, 3.0)
        mul_85: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_100: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_239, mul_85);  view_239 = mul_85 = None
        mul_86: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_10: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_101: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_517: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1741, add_101);  convert_element_type_1741 = add_101 = None
        convert_element_type_1744: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_517, torch.bfloat16);  mul_517 = None
        mul_518: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_172: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_518);  mul_518 = None
        mul_519: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_516, sub_172);  mul_516 = sub_172 = None
        mul_520: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_519, 0.7978845608028654);  mul_519 = None
        convert_element_type_1745: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_520, torch.bfloat16)
        mul_521: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_520, 0.044715);  mul_520 = None
        pow_38: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_422, 2.0);  convert_element_type_422 = None
        mul_522: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_38, 3.0);  pow_38 = None
        mul_523: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
        convert_element_type_1746: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_523, torch.bfloat16);  mul_523 = None
        add_301: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1745, convert_element_type_1746);  convert_element_type_1745 = convert_element_type_1746 = None
        mul_524: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1744, 0.5);  convert_element_type_1744 = None
        add_302: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_301, mul_524);  add_301 = mul_524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_865: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_302, [4096, 8192]);  add_302 = None
        mm_233: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_865, permute_702);  permute_702 = None
        permute_703: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_865, [1, 0])
        mm_234: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_703, view_238);  permute_703 = view_238 = None
        sum_190: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_865, [0], True, dtype = torch.float32);  view_865 = None
        view_866: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_190, [8192]);  sum_190 = None
        convert_element_type_1751: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_866, torch.bfloat16);  view_866 = None
        view_867: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_233, [32, 128, 2048]);  mm_233 = None
        convert_element_type_1752: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_867, torch.float32);  view_867 = None
        convert_element_type_1753: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_234, torch.float32);  mm_234 = None
        convert_element_type_1754: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1751, torch.float32);  convert_element_type_1751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_526: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1752, primals_152);  primals_152 = None
        mul_527: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_526, 2048)
        sum_191: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_526, [2], True)
        mul_528: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_526, mul_82);  mul_526 = None
        sum_192: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_528, [2], True);  mul_528 = None
        mul_529: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, sum_192);  sum_192 = None
        sub_174: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_527, sum_191);  mul_527 = sum_191 = None
        sub_175: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_174, mul_529);  sub_174 = mul_529 = None
        mul_530: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_53, sub_175);  div_53 = sub_175 = None
        mul_531: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1752, mul_82);  mul_82 = None
        sum_193: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_531, [0, 1]);  mul_531 = None
        sum_194: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1752, [0, 1]);  convert_element_type_1752 = None
        add_303: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_300, mul_530);  add_300 = mul_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1755: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_303, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_868: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1755, [4096, 2048]);  convert_element_type_1755 = None
        mm_235: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_868, permute_706);  permute_706 = None
        permute_707: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_868, [1, 0])
        mm_236: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_707, view_236);  permute_707 = view_236 = None
        sum_195: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_868, [0], True, dtype = torch.float32);  view_868 = None
        view_869: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_195, [2048]);  sum_195 = None
        convert_element_type_1760: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_869, torch.bfloat16);  view_869 = None
        view_870: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_235, [32, 128, 2048]);  mm_235 = None
        convert_element_type_1761: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_236, torch.float32);  mm_236 = None
        convert_element_type_1762: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1760, torch.float32);  convert_element_type_1760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_871: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_870, [32, 128, 16, 128]);  view_870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_710: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_871, [0, 2, 1, 3]);  view_871 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_221: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_710, memory_format = torch.contiguous_format);  permute_710 = None
        view_872: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_221, [512, 128, 128]);  clone_221 = None
        bmm_100: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_711, view_872);  permute_711 = None
        bmm_101: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_872, permute_712);  view_872 = permute_712 = None
        view_873: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_100, [32, 16, 128, 128]);  bmm_100 = None
        view_874: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_101, [32, 16, 128, 128]);  bmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1767: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_874, torch.float32);  view_874 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_532: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1767, div_10);  convert_element_type_1767 = None
        sum_196: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_532, [-1], True)
        neg_14: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_13: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_14, sum_196, mul_532);  neg_14 = sum_196 = mul_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1768: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_13, torch.bfloat16);  fma_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_24: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_149, 2, 0, 128);  primals_149 = None
        slice_25: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_24, 3, 0, 128);  slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_42: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_25, convert_element_type_1768, full_default_32);  slice_25 = convert_element_type_1768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_875: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_42, [512, 128, 128]);  where_42 = None
        bmm_102: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_713, view_875);  permute_713 = None
        bmm_103: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_875, permute_714);  view_875 = permute_714 = None
        view_876: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_102, [32, 16, 128, 128]);  bmm_102 = None
        view_877: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_103, [32, 16, 128, 128]);  bmm_103 = None
        convert_element_type_1774: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_876, torch.float32);  view_876 = None
        permute_715: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1774, [0, 1, 3, 2]);  convert_element_type_1774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1775: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_715, torch.bfloat16);  permute_715 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_10: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_877, torch.bfloat16);  view_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_716: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_873, [0, 2, 1, 3]);  view_873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_222: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_716, memory_format = torch.contiguous_format);  permute_716 = None
        view_878: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_222, [32, 128, 2048]);  clone_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_717: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1775, [0, 2, 1, 3]);  convert_element_type_1775 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_879: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_717, [32, 128, 2048]);  permute_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_718: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_10, [0, 2, 1, 3]);  convert_element_type_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_223: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_718, memory_format = torch.contiguous_format);  permute_718 = None
        view_880: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_223, [32, 128, 2048]);  clone_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_881: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_878, [4096, 2048]);  view_878 = None
        permute_719: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_881, [1, 0])
        mm_237: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_719, view_220);  permute_719 = None
        mm_238: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_881, permute_721);  view_881 = permute_721 = None
        view_882: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_238, [32, 128, 2048]);  mm_238 = None
        convert_element_type_1781: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_882, torch.float32);  view_882 = None
        convert_element_type_1782: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_237, torch.float32);  mm_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_224: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_879, memory_format = torch.contiguous_format);  view_879 = None
        view_883: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_224, [4096, 2048]);  clone_224 = None
        permute_723: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_883, [1, 0])
        mm_239: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_723, view_220);  permute_723 = None
        mm_240: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_883, permute_725);  view_883 = permute_725 = None
        view_884: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_240, [32, 128, 2048]);  mm_240 = None
        convert_element_type_1787: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_884, torch.float32);  view_884 = None
        add_304: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1781, convert_element_type_1787);  convert_element_type_1781 = convert_element_type_1787 = None
        convert_element_type_1788: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_239, torch.float32);  mm_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_885: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_880, [4096, 2048]);  view_880 = None
        permute_727: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_885, [1, 0])
        mm_241: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_727, view_220);  permute_727 = view_220 = None
        mm_242: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_885, permute_729);  view_885 = permute_729 = None
        view_886: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_242, [32, 128, 2048]);  mm_242 = None
        convert_element_type_1793: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_886, torch.float32);  view_886 = None
        add_305: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_304, convert_element_type_1793);  add_304 = convert_element_type_1793 = None
        convert_element_type_1794: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_241, torch.float32);  mm_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_534: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_305, primals_144);  primals_144 = None
        mul_535: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_534, 2048)
        sum_197: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_534, [2], True)
        mul_536: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_534, mul_80);  mul_534 = None
        sum_198: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_536, [2], True);  mul_536 = None
        mul_537: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, sum_198);  sum_198 = None
        sub_177: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_535, sum_197);  mul_535 = sum_197 = None
        sub_178: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_177, mul_537);  sub_177 = mul_537 = None
        mul_538: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_54, sub_178);  div_54 = sub_178 = None
        mul_539: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_305, mul_80);  mul_80 = None
        sum_199: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_539, [0, 1]);  mul_539 = None
        sum_200: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_305, [0, 1]);  add_305 = None
        add_306: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_303, mul_538);  add_303 = mul_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1795: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_306, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_887: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1795, [4096, 2048]);  convert_element_type_1795 = None
        mm_243: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_887, permute_731);  permute_731 = None
        permute_732: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_887, [1, 0])
        mm_244: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_732, view_218);  permute_732 = view_218 = None
        sum_201: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_887, [0], True, dtype = torch.float32);  view_887 = None
        view_888: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_201, [2048]);  sum_201 = None
        convert_element_type_1800: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_888, torch.bfloat16);  view_888 = None
        view_889: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_243, [32, 128, 8192]);  mm_243 = None
        convert_element_type_1801: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_889, torch.float32);  view_889 = None
        convert_element_type_1802: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_244, torch.float32);  mm_244 = None
        convert_element_type_1803: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1800, torch.float32);  convert_element_type_1800 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_217: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [32, 128, 8192]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_540: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1801, mul_76);  mul_76 = None
        convert_element_type_383: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_217, torch.float32)
        pow_10: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_383, 3.0)
        mul_77: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_91: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_217, mul_77);  view_217 = mul_77 = None
        mul_78: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_91, 0.7978845608028654);  add_91 = None
        tanh_9: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_92: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_541: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1801, add_92);  convert_element_type_1801 = add_92 = None
        convert_element_type_1804: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_541, torch.bfloat16);  mul_541 = None
        mul_542: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_179: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_542);  mul_542 = None
        mul_543: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_540, sub_179);  mul_540 = sub_179 = None
        mul_544: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_543, 0.7978845608028654);  mul_543 = None
        convert_element_type_1805: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_544, torch.bfloat16)
        mul_545: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_544, 0.044715);  mul_544 = None
        pow_39: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_383, 2.0);  convert_element_type_383 = None
        mul_546: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_39, 3.0);  pow_39 = None
        mul_547: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_545, mul_546);  mul_545 = mul_546 = None
        convert_element_type_1806: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_547, torch.bfloat16);  mul_547 = None
        add_307: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1805, convert_element_type_1806);  convert_element_type_1805 = convert_element_type_1806 = None
        mul_548: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1804, 0.5);  convert_element_type_1804 = None
        add_308: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_307, mul_548);  add_307 = mul_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_890: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_308, [4096, 8192]);  add_308 = None
        mm_245: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_890, permute_735);  permute_735 = None
        permute_736: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_890, [1, 0])
        mm_246: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_736, view_216);  permute_736 = view_216 = None
        sum_202: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_890, [0], True, dtype = torch.float32);  view_890 = None
        view_891: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_202, [8192]);  sum_202 = None
        convert_element_type_1811: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_891, torch.bfloat16);  view_891 = None
        view_892: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_245, [32, 128, 2048]);  mm_245 = None
        convert_element_type_1812: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_892, torch.float32);  view_892 = None
        convert_element_type_1813: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_246, torch.float32);  mm_246 = None
        convert_element_type_1814: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1811, torch.float32);  convert_element_type_1811 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_550: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1812, primals_138);  primals_138 = None
        mul_551: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_550, 2048)
        sum_203: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_550, [2], True)
        mul_552: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_550, mul_74);  mul_550 = None
        sum_204: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_552, [2], True);  mul_552 = None
        mul_553: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, sum_204);  sum_204 = None
        sub_181: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_551, sum_203);  mul_551 = sum_203 = None
        sub_182: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_181, mul_553);  sub_181 = mul_553 = None
        mul_554: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_55, sub_182);  div_55 = sub_182 = None
        mul_555: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1812, mul_74);  mul_74 = None
        sum_205: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_555, [0, 1]);  mul_555 = None
        sum_206: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1812, [0, 1]);  convert_element_type_1812 = None
        add_309: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_306, mul_554);  add_306 = mul_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1815: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_309, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_893: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1815, [4096, 2048]);  convert_element_type_1815 = None
        mm_247: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_893, permute_739);  permute_739 = None
        permute_740: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_893, [1, 0])
        mm_248: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_740, view_214);  permute_740 = view_214 = None
        sum_207: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_893, [0], True, dtype = torch.float32);  view_893 = None
        view_894: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_207, [2048]);  sum_207 = None
        convert_element_type_1820: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_894, torch.bfloat16);  view_894 = None
        view_895: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_247, [32, 128, 2048]);  mm_247 = None
        convert_element_type_1821: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_248, torch.float32);  mm_248 = None
        convert_element_type_1822: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1820, torch.float32);  convert_element_type_1820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_896: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_895, [32, 128, 16, 128]);  view_895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_743: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_896, [0, 2, 1, 3]);  view_896 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_225: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_743, memory_format = torch.contiguous_format);  permute_743 = None
        view_897: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_225, [512, 128, 128]);  clone_225 = None
        bmm_104: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_744, view_897);  permute_744 = None
        bmm_105: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_897, permute_745);  view_897 = permute_745 = None
        view_898: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_104, [32, 16, 128, 128]);  bmm_104 = None
        view_899: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_105, [32, 16, 128, 128]);  bmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1827: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_899, torch.float32);  view_899 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_556: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1827, div_9);  convert_element_type_1827 = None
        sum_208: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_556, [-1], True)
        neg_15: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_9);  div_9 = None
        fma_14: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_15, sum_208, mul_556);  neg_15 = sum_208 = mul_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1828: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_14, torch.bfloat16);  fma_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_22: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_135, 2, 0, 128);  primals_135 = None
        slice_23: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_22, 3, 0, 128);  slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_43: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_23, convert_element_type_1828, full_default_32);  slice_23 = convert_element_type_1828 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_900: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_43, [512, 128, 128]);  where_43 = None
        bmm_106: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_746, view_900);  permute_746 = None
        bmm_107: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_900, permute_747);  view_900 = permute_747 = None
        view_901: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_106, [32, 16, 128, 128]);  bmm_106 = None
        view_902: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_107, [32, 16, 128, 128]);  bmm_107 = None
        convert_element_type_1834: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_901, torch.float32);  view_901 = None
        permute_748: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1834, [0, 1, 3, 2]);  convert_element_type_1834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1835: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_748, torch.bfloat16);  permute_748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_9: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_902, torch.bfloat16);  view_902 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_749: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_898, [0, 2, 1, 3]);  view_898 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_226: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_749, memory_format = torch.contiguous_format);  permute_749 = None
        view_903: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_226, [32, 128, 2048]);  clone_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_750: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1835, [0, 2, 1, 3]);  convert_element_type_1835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_904: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_750, [32, 128, 2048]);  permute_750 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_751: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_9, [0, 2, 1, 3]);  convert_element_type_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_227: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_751, memory_format = torch.contiguous_format);  permute_751 = None
        view_905: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_227, [32, 128, 2048]);  clone_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_906: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_903, [4096, 2048]);  view_903 = None
        permute_752: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_906, [1, 0])
        mm_249: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_752, view_198);  permute_752 = None
        mm_250: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_906, permute_754);  view_906 = permute_754 = None
        view_907: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_250, [32, 128, 2048]);  mm_250 = None
        convert_element_type_1841: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_907, torch.float32);  view_907 = None
        convert_element_type_1842: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_249, torch.float32);  mm_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_228: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_904, memory_format = torch.contiguous_format);  view_904 = None
        view_908: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_228, [4096, 2048]);  clone_228 = None
        permute_756: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_908, [1, 0])
        mm_251: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_756, view_198);  permute_756 = None
        mm_252: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_908, permute_758);  view_908 = permute_758 = None
        view_909: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_252, [32, 128, 2048]);  mm_252 = None
        convert_element_type_1847: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_909, torch.float32);  view_909 = None
        add_310: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1841, convert_element_type_1847);  convert_element_type_1841 = convert_element_type_1847 = None
        convert_element_type_1848: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_251, torch.float32);  mm_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_910: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_905, [4096, 2048]);  view_905 = None
        permute_760: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_910, [1, 0])
        mm_253: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_760, view_198);  permute_760 = view_198 = None
        mm_254: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_910, permute_762);  view_910 = permute_762 = None
        view_911: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_254, [32, 128, 2048]);  mm_254 = None
        convert_element_type_1853: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_911, torch.float32);  view_911 = None
        add_311: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_310, convert_element_type_1853);  add_310 = convert_element_type_1853 = None
        convert_element_type_1854: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_253, torch.float32);  mm_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_558: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_311, primals_130);  primals_130 = None
        mul_559: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_558, 2048)
        sum_209: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_558, [2], True)
        mul_560: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_558, mul_72);  mul_558 = None
        sum_210: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_560, [2], True);  mul_560 = None
        mul_561: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, sum_210);  sum_210 = None
        sub_184: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_559, sum_209);  mul_559 = sum_209 = None
        sub_185: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_184, mul_561);  sub_184 = mul_561 = None
        mul_562: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_56, sub_185);  div_56 = sub_185 = None
        mul_563: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_311, mul_72);  mul_72 = None
        sum_211: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_563, [0, 1]);  mul_563 = None
        sum_212: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_311, [0, 1]);  add_311 = None
        add_312: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_309, mul_562);  add_309 = mul_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1855: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_312, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_912: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1855, [4096, 2048]);  convert_element_type_1855 = None
        mm_255: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_912, permute_764);  permute_764 = None
        permute_765: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_912, [1, 0])
        mm_256: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_765, view_196);  permute_765 = view_196 = None
        sum_213: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_912, [0], True, dtype = torch.float32);  view_912 = None
        view_913: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_213, [2048]);  sum_213 = None
        convert_element_type_1860: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_913, torch.bfloat16);  view_913 = None
        view_914: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_255, [32, 128, 8192]);  mm_255 = None
        convert_element_type_1861: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_914, torch.float32);  view_914 = None
        convert_element_type_1862: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_256, torch.float32);  mm_256 = None
        convert_element_type_1863: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1860, torch.float32);  convert_element_type_1860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_195: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [32, 128, 8192]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_564: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1861, mul_68);  mul_68 = None
        convert_element_type_344: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_195, torch.float32)
        pow_9: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_344, 3.0)
        mul_69: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_82: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_195, mul_69);  view_195 = mul_69 = None
        mul_70: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_82, 0.7978845608028654);  add_82 = None
        tanh_8: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_83: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_565: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1861, add_83);  convert_element_type_1861 = add_83 = None
        convert_element_type_1864: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_565, torch.bfloat16);  mul_565 = None
        mul_566: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_186: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_566);  mul_566 = None
        mul_567: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_564, sub_186);  mul_564 = sub_186 = None
        mul_568: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_567, 0.7978845608028654);  mul_567 = None
        convert_element_type_1865: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_568, torch.bfloat16)
        mul_569: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_568, 0.044715);  mul_568 = None
        pow_40: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_344, 2.0);  convert_element_type_344 = None
        mul_570: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_40, 3.0);  pow_40 = None
        mul_571: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_569, mul_570);  mul_569 = mul_570 = None
        convert_element_type_1866: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_571, torch.bfloat16);  mul_571 = None
        add_313: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1865, convert_element_type_1866);  convert_element_type_1865 = convert_element_type_1866 = None
        mul_572: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1864, 0.5);  convert_element_type_1864 = None
        add_314: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_313, mul_572);  add_313 = mul_572 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_915: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_314, [4096, 8192]);  add_314 = None
        mm_257: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_915, permute_768);  permute_768 = None
        permute_769: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_915, [1, 0])
        mm_258: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_769, view_194);  permute_769 = view_194 = None
        sum_214: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_915, [0], True, dtype = torch.float32);  view_915 = None
        view_916: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_214, [8192]);  sum_214 = None
        convert_element_type_1871: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_916, torch.bfloat16);  view_916 = None
        view_917: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_257, [32, 128, 2048]);  mm_257 = None
        convert_element_type_1872: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_917, torch.float32);  view_917 = None
        convert_element_type_1873: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_258, torch.float32);  mm_258 = None
        convert_element_type_1874: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1871, torch.float32);  convert_element_type_1871 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_574: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1872, primals_124);  primals_124 = None
        mul_575: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_574, 2048)
        sum_215: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_574, [2], True)
        mul_576: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_574, mul_66);  mul_574 = None
        sum_216: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_576, [2], True);  mul_576 = None
        mul_577: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, sum_216);  sum_216 = None
        sub_188: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_575, sum_215);  mul_575 = sum_215 = None
        sub_189: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_188, mul_577);  sub_188 = mul_577 = None
        mul_578: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_57, sub_189);  div_57 = sub_189 = None
        mul_579: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1872, mul_66);  mul_66 = None
        sum_217: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_579, [0, 1]);  mul_579 = None
        sum_218: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1872, [0, 1]);  convert_element_type_1872 = None
        add_315: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_312, mul_578);  add_312 = mul_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1875: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_315, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_918: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1875, [4096, 2048]);  convert_element_type_1875 = None
        mm_259: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_918, permute_772);  permute_772 = None
        permute_773: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_918, [1, 0])
        mm_260: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_773, view_192);  permute_773 = view_192 = None
        sum_219: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_918, [0], True, dtype = torch.float32);  view_918 = None
        view_919: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_219, [2048]);  sum_219 = None
        convert_element_type_1880: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_919, torch.bfloat16);  view_919 = None
        view_920: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_259, [32, 128, 2048]);  mm_259 = None
        convert_element_type_1881: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_260, torch.float32);  mm_260 = None
        convert_element_type_1882: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1880, torch.float32);  convert_element_type_1880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_921: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_920, [32, 128, 16, 128]);  view_920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_776: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_921, [0, 2, 1, 3]);  view_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_229: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_776, memory_format = torch.contiguous_format);  permute_776 = None
        view_922: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_229, [512, 128, 128]);  clone_229 = None
        bmm_108: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_777, view_922);  permute_777 = None
        bmm_109: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_922, permute_778);  view_922 = permute_778 = None
        view_923: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_108, [32, 16, 128, 128]);  bmm_108 = None
        view_924: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_109, [32, 16, 128, 128]);  bmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1887: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_924, torch.float32);  view_924 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_580: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1887, div_8);  convert_element_type_1887 = None
        sum_220: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_580, [-1], True)
        neg_16: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_8);  div_8 = None
        fma_15: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_16, sum_220, mul_580);  neg_16 = sum_220 = mul_580 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1888: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_15, torch.bfloat16);  fma_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_20: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_121, 2, 0, 128);  primals_121 = None
        slice_21: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_20, 3, 0, 128);  slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_44: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_21, convert_element_type_1888, full_default_32);  slice_21 = convert_element_type_1888 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_925: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_44, [512, 128, 128]);  where_44 = None
        bmm_110: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_779, view_925);  permute_779 = None
        bmm_111: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_925, permute_780);  view_925 = permute_780 = None
        view_926: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_110, [32, 16, 128, 128]);  bmm_110 = None
        view_927: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_111, [32, 16, 128, 128]);  bmm_111 = None
        convert_element_type_1894: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_926, torch.float32);  view_926 = None
        permute_781: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1894, [0, 1, 3, 2]);  convert_element_type_1894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1895: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_781, torch.bfloat16);  permute_781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_8: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_927, torch.bfloat16);  view_927 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_782: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_923, [0, 2, 1, 3]);  view_923 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_230: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_782, memory_format = torch.contiguous_format);  permute_782 = None
        view_928: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_230, [32, 128, 2048]);  clone_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_783: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1895, [0, 2, 1, 3]);  convert_element_type_1895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_929: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_783, [32, 128, 2048]);  permute_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_784: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_8, [0, 2, 1, 3]);  convert_element_type_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_231: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_784, memory_format = torch.contiguous_format);  permute_784 = None
        view_930: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_231, [32, 128, 2048]);  clone_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_931: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_928, [4096, 2048]);  view_928 = None
        permute_785: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_931, [1, 0])
        mm_261: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_785, view_176);  permute_785 = None
        mm_262: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_931, permute_787);  view_931 = permute_787 = None
        view_932: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_262, [32, 128, 2048]);  mm_262 = None
        convert_element_type_1901: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_932, torch.float32);  view_932 = None
        convert_element_type_1902: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_261, torch.float32);  mm_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_232: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_929, memory_format = torch.contiguous_format);  view_929 = None
        view_933: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_232, [4096, 2048]);  clone_232 = None
        permute_789: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_933, [1, 0])
        mm_263: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_789, view_176);  permute_789 = None
        mm_264: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_933, permute_791);  view_933 = permute_791 = None
        view_934: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_264, [32, 128, 2048]);  mm_264 = None
        convert_element_type_1907: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_934, torch.float32);  view_934 = None
        add_316: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1901, convert_element_type_1907);  convert_element_type_1901 = convert_element_type_1907 = None
        convert_element_type_1908: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_263, torch.float32);  mm_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_935: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_930, [4096, 2048]);  view_930 = None
        permute_793: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_935, [1, 0])
        mm_265: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_793, view_176);  permute_793 = view_176 = None
        mm_266: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_935, permute_795);  view_935 = permute_795 = None
        view_936: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_266, [32, 128, 2048]);  mm_266 = None
        convert_element_type_1913: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_936, torch.float32);  view_936 = None
        add_317: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_316, convert_element_type_1913);  add_316 = convert_element_type_1913 = None
        convert_element_type_1914: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_265, torch.float32);  mm_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_582: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_317, primals_116);  primals_116 = None
        mul_583: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_582, 2048)
        sum_221: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_582, [2], True)
        mul_584: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_582, mul_64);  mul_582 = None
        sum_222: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_584, [2], True);  mul_584 = None
        mul_585: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, sum_222);  sum_222 = None
        sub_191: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_583, sum_221);  mul_583 = sum_221 = None
        sub_192: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_191, mul_585);  sub_191 = mul_585 = None
        mul_586: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_58, sub_192);  div_58 = sub_192 = None
        mul_587: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_317, mul_64);  mul_64 = None
        sum_223: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_587, [0, 1]);  mul_587 = None
        sum_224: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_317, [0, 1]);  add_317 = None
        add_318: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_315, mul_586);  add_315 = mul_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1915: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_318, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_937: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1915, [4096, 2048]);  convert_element_type_1915 = None
        mm_267: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_937, permute_797);  permute_797 = None
        permute_798: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_937, [1, 0])
        mm_268: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_798, view_174);  permute_798 = view_174 = None
        sum_225: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_937, [0], True, dtype = torch.float32);  view_937 = None
        view_938: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_225, [2048]);  sum_225 = None
        convert_element_type_1920: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_938, torch.bfloat16);  view_938 = None
        view_939: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_267, [32, 128, 8192]);  mm_267 = None
        convert_element_type_1921: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_939, torch.float32);  view_939 = None
        convert_element_type_1922: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_268, torch.float32);  mm_268 = None
        convert_element_type_1923: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1920, torch.float32);  convert_element_type_1920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_173: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [32, 128, 8192]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_588: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1921, mul_60);  mul_60 = None
        convert_element_type_305: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32)
        pow_8: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_305, 3.0)
        mul_61: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_73: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_173, mul_61);  view_173 = mul_61 = None
        mul_62: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_7: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_74: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_589: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1921, add_74);  convert_element_type_1921 = add_74 = None
        convert_element_type_1924: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_589, torch.bfloat16);  mul_589 = None
        mul_590: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_193: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_590);  mul_590 = None
        mul_591: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_588, sub_193);  mul_588 = sub_193 = None
        mul_592: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_591, 0.7978845608028654);  mul_591 = None
        convert_element_type_1925: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_592, torch.bfloat16)
        mul_593: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_592, 0.044715);  mul_592 = None
        pow_41: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_305, 2.0);  convert_element_type_305 = None
        mul_594: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_41, 3.0);  pow_41 = None
        mul_595: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_593, mul_594);  mul_593 = mul_594 = None
        convert_element_type_1926: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_595, torch.bfloat16);  mul_595 = None
        add_319: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1925, convert_element_type_1926);  convert_element_type_1925 = convert_element_type_1926 = None
        mul_596: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1924, 0.5);  convert_element_type_1924 = None
        add_320: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_319, mul_596);  add_319 = mul_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_940: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_320, [4096, 8192]);  add_320 = None
        mm_269: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_940, permute_801);  permute_801 = None
        permute_802: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_940, [1, 0])
        mm_270: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_802, view_172);  permute_802 = view_172 = None
        sum_226: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_940, [0], True, dtype = torch.float32);  view_940 = None
        view_941: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_226, [8192]);  sum_226 = None
        convert_element_type_1931: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_941, torch.bfloat16);  view_941 = None
        view_942: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_269, [32, 128, 2048]);  mm_269 = None
        convert_element_type_1932: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_942, torch.float32);  view_942 = None
        convert_element_type_1933: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_270, torch.float32);  mm_270 = None
        convert_element_type_1934: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1931, torch.float32);  convert_element_type_1931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_598: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1932, primals_110);  primals_110 = None
        mul_599: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_598, 2048)
        sum_227: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_598, [2], True)
        mul_600: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_598, mul_58);  mul_598 = None
        sum_228: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_600, [2], True);  mul_600 = None
        mul_601: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, sum_228);  sum_228 = None
        sub_195: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_599, sum_227);  mul_599 = sum_227 = None
        sub_196: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_195, mul_601);  sub_195 = mul_601 = None
        mul_602: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_59, sub_196);  div_59 = sub_196 = None
        mul_603: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1932, mul_58);  mul_58 = None
        sum_229: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_603, [0, 1]);  mul_603 = None
        sum_230: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1932, [0, 1]);  convert_element_type_1932 = None
        add_321: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_318, mul_602);  add_318 = mul_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1935: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_321, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_943: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1935, [4096, 2048]);  convert_element_type_1935 = None
        mm_271: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_943, permute_805);  permute_805 = None
        permute_806: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_943, [1, 0])
        mm_272: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_806, view_170);  permute_806 = view_170 = None
        sum_231: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_943, [0], True, dtype = torch.float32);  view_943 = None
        view_944: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_231, [2048]);  sum_231 = None
        convert_element_type_1940: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_944, torch.bfloat16);  view_944 = None
        view_945: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_271, [32, 128, 2048]);  mm_271 = None
        convert_element_type_1941: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_272, torch.float32);  mm_272 = None
        convert_element_type_1942: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1940, torch.float32);  convert_element_type_1940 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_946: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_945, [32, 128, 16, 128]);  view_945 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_809: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_946, [0, 2, 1, 3]);  view_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_233: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_809, memory_format = torch.contiguous_format);  permute_809 = None
        view_947: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_233, [512, 128, 128]);  clone_233 = None
        bmm_112: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_810, view_947);  permute_810 = None
        bmm_113: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_947, permute_811);  view_947 = permute_811 = None
        view_948: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_112, [32, 16, 128, 128]);  bmm_112 = None
        view_949: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_113, [32, 16, 128, 128]);  bmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_1947: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_949, torch.float32);  view_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_604: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1947, div_7);  convert_element_type_1947 = None
        sum_232: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_604, [-1], True)
        neg_17: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_16: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_17, sum_232, mul_604);  neg_17 = sum_232 = mul_604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1948: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_16, torch.bfloat16);  fma_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_18: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_107, 2, 0, 128);  primals_107 = None
        slice_19: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_18, 3, 0, 128);  slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_45: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_19, convert_element_type_1948, full_default_32);  slice_19 = convert_element_type_1948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_950: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_45, [512, 128, 128]);  where_45 = None
        bmm_114: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_812, view_950);  permute_812 = None
        bmm_115: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_950, permute_813);  view_950 = permute_813 = None
        view_951: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_114, [32, 16, 128, 128]);  bmm_114 = None
        view_952: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_115, [32, 16, 128, 128]);  bmm_115 = None
        convert_element_type_1954: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_951, torch.float32);  view_951 = None
        permute_814: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1954, [0, 1, 3, 2]);  convert_element_type_1954 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_1955: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_814, torch.bfloat16);  permute_814 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_7: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_952, torch.bfloat16);  view_952 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_815: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_948, [0, 2, 1, 3]);  view_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_234: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_815, memory_format = torch.contiguous_format);  permute_815 = None
        view_953: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_234, [32, 128, 2048]);  clone_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_816: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1955, [0, 2, 1, 3]);  convert_element_type_1955 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_954: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_816, [32, 128, 2048]);  permute_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_817: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_7, [0, 2, 1, 3]);  convert_element_type_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_235: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_817, memory_format = torch.contiguous_format);  permute_817 = None
        view_955: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_235, [32, 128, 2048]);  clone_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_956: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_953, [4096, 2048]);  view_953 = None
        permute_818: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_956, [1, 0])
        mm_273: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_818, view_154);  permute_818 = None
        mm_274: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_956, permute_820);  view_956 = permute_820 = None
        view_957: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_274, [32, 128, 2048]);  mm_274 = None
        convert_element_type_1961: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_957, torch.float32);  view_957 = None
        convert_element_type_1962: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_273, torch.float32);  mm_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_236: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_954, memory_format = torch.contiguous_format);  view_954 = None
        view_958: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_236, [4096, 2048]);  clone_236 = None
        permute_822: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_958, [1, 0])
        mm_275: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_822, view_154);  permute_822 = None
        mm_276: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_958, permute_824);  view_958 = permute_824 = None
        view_959: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_276, [32, 128, 2048]);  mm_276 = None
        convert_element_type_1967: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_959, torch.float32);  view_959 = None
        add_322: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1961, convert_element_type_1967);  convert_element_type_1961 = convert_element_type_1967 = None
        convert_element_type_1968: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_275, torch.float32);  mm_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_960: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_955, [4096, 2048]);  view_955 = None
        permute_826: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_960, [1, 0])
        mm_277: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_826, view_154);  permute_826 = view_154 = None
        mm_278: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_960, permute_828);  view_960 = permute_828 = None
        view_961: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_278, [32, 128, 2048]);  mm_278 = None
        convert_element_type_1973: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_961, torch.float32);  view_961 = None
        add_323: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_322, convert_element_type_1973);  add_322 = convert_element_type_1973 = None
        convert_element_type_1974: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_277, torch.float32);  mm_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_606: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_323, primals_102);  primals_102 = None
        mul_607: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_606, 2048)
        sum_233: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_606, [2], True)
        mul_608: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_606, mul_56);  mul_606 = None
        sum_234: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_608, [2], True);  mul_608 = None
        mul_609: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, sum_234);  sum_234 = None
        sub_198: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_607, sum_233);  mul_607 = sum_233 = None
        sub_199: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_198, mul_609);  sub_198 = mul_609 = None
        mul_610: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_60, sub_199);  div_60 = sub_199 = None
        mul_611: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_323, mul_56);  mul_56 = None
        sum_235: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_611, [0, 1]);  mul_611 = None
        sum_236: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_323, [0, 1]);  add_323 = None
        add_324: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_321, mul_610);  add_321 = mul_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_1975: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_324, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_962: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1975, [4096, 2048]);  convert_element_type_1975 = None
        mm_279: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_962, permute_830);  permute_830 = None
        permute_831: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_962, [1, 0])
        mm_280: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_831, view_152);  permute_831 = view_152 = None
        sum_237: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_962, [0], True, dtype = torch.float32);  view_962 = None
        view_963: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_237, [2048]);  sum_237 = None
        convert_element_type_1980: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_963, torch.bfloat16);  view_963 = None
        view_964: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_279, [32, 128, 8192]);  mm_279 = None
        convert_element_type_1981: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_964, torch.float32);  view_964 = None
        convert_element_type_1982: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_280, torch.float32);  mm_280 = None
        convert_element_type_1983: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1980, torch.float32);  convert_element_type_1980 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_151: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [32, 128, 8192]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_52: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_151, 0.5)
        mul_612: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1981, mul_52);  mul_52 = None
        convert_element_type_266: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_151, torch.float32)
        pow_7: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_266, 3.0)
        mul_53: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_64: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_151, mul_53);  view_151 = mul_53 = None
        mul_54: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_64, 0.7978845608028654);  add_64 = None
        tanh_6: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_65: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_613: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1981, add_65);  convert_element_type_1981 = add_65 = None
        convert_element_type_1984: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_613, torch.bfloat16);  mul_613 = None
        mul_614: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_200: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_614);  mul_614 = None
        mul_615: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_612, sub_200);  mul_612 = sub_200 = None
        mul_616: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_615, 0.7978845608028654);  mul_615 = None
        convert_element_type_1985: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_616, torch.bfloat16)
        mul_617: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_616, 0.044715);  mul_616 = None
        pow_42: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_266, 2.0);  convert_element_type_266 = None
        mul_618: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_42, 3.0);  pow_42 = None
        mul_619: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
        convert_element_type_1986: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_619, torch.bfloat16);  mul_619 = None
        add_325: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1985, convert_element_type_1986);  convert_element_type_1985 = convert_element_type_1986 = None
        mul_620: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1984, 0.5);  convert_element_type_1984 = None
        add_326: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_325, mul_620);  add_325 = mul_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_965: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_326, [4096, 8192]);  add_326 = None
        mm_281: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_965, permute_834);  permute_834 = None
        permute_835: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_965, [1, 0])
        mm_282: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_835, view_150);  permute_835 = view_150 = None
        sum_238: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_965, [0], True, dtype = torch.float32);  view_965 = None
        view_966: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_238, [8192]);  sum_238 = None
        convert_element_type_1991: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_966, torch.bfloat16);  view_966 = None
        view_967: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_281, [32, 128, 2048]);  mm_281 = None
        convert_element_type_1992: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_967, torch.float32);  view_967 = None
        convert_element_type_1993: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_282, torch.float32);  mm_282 = None
        convert_element_type_1994: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1991, torch.float32);  convert_element_type_1991 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_622: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1992, primals_96);  primals_96 = None
        mul_623: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_622, 2048)
        sum_239: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_622, [2], True)
        mul_624: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_622, mul_50);  mul_622 = None
        sum_240: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_624, [2], True);  mul_624 = None
        mul_625: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, sum_240);  sum_240 = None
        sub_202: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_623, sum_239);  mul_623 = sum_239 = None
        sub_203: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_202, mul_625);  sub_202 = mul_625 = None
        mul_626: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_61, sub_203);  div_61 = sub_203 = None
        mul_627: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1992, mul_50);  mul_50 = None
        sum_241: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_627, [0, 1]);  mul_627 = None
        sum_242: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1992, [0, 1]);  convert_element_type_1992 = None
        add_327: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_324, mul_626);  add_324 = mul_626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_1995: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_327, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_968: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1995, [4096, 2048]);  convert_element_type_1995 = None
        mm_283: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_968, permute_838);  permute_838 = None
        permute_839: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_968, [1, 0])
        mm_284: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_839, view_148);  permute_839 = view_148 = None
        sum_243: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_968, [0], True, dtype = torch.float32);  view_968 = None
        view_969: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_243, [2048]);  sum_243 = None
        convert_element_type_2000: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_969, torch.bfloat16);  view_969 = None
        view_970: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_283, [32, 128, 2048]);  mm_283 = None
        convert_element_type_2001: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_284, torch.float32);  mm_284 = None
        convert_element_type_2002: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2000, torch.float32);  convert_element_type_2000 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_971: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_970, [32, 128, 16, 128]);  view_970 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_842: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_971, [0, 2, 1, 3]);  view_971 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_237: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_842, memory_format = torch.contiguous_format);  permute_842 = None
        view_972: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_237, [512, 128, 128]);  clone_237 = None
        bmm_116: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_843, view_972);  permute_843 = None
        bmm_117: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_972, permute_844);  view_972 = permute_844 = None
        view_973: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_116, [32, 16, 128, 128]);  bmm_116 = None
        view_974: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_117, [32, 16, 128, 128]);  bmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2007: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_974, torch.float32);  view_974 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_628: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2007, div_6);  convert_element_type_2007 = None
        sum_244: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_628, [-1], True)
        neg_18: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_6);  div_6 = None
        fma_17: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_18, sum_244, mul_628);  neg_18 = sum_244 = mul_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2008: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_17, torch.bfloat16);  fma_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_16: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_93, 2, 0, 128);  primals_93 = None
        slice_17: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_16, 3, 0, 128);  slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_46: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_17, convert_element_type_2008, full_default_32);  slice_17 = convert_element_type_2008 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_975: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_46, [512, 128, 128]);  where_46 = None
        bmm_118: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_845, view_975);  permute_845 = None
        bmm_119: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_975, permute_846);  view_975 = permute_846 = None
        view_976: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_118, [32, 16, 128, 128]);  bmm_118 = None
        view_977: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_119, [32, 16, 128, 128]);  bmm_119 = None
        convert_element_type_2014: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_976, torch.float32);  view_976 = None
        permute_847: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2014, [0, 1, 3, 2]);  convert_element_type_2014 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2015: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_847, torch.bfloat16);  permute_847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_6: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_977, torch.bfloat16);  view_977 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_848: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_973, [0, 2, 1, 3]);  view_973 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_238: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_848, memory_format = torch.contiguous_format);  permute_848 = None
        view_978: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_238, [32, 128, 2048]);  clone_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_849: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2015, [0, 2, 1, 3]);  convert_element_type_2015 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_979: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_849, [32, 128, 2048]);  permute_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_850: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_6, [0, 2, 1, 3]);  convert_element_type_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_239: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_850, memory_format = torch.contiguous_format);  permute_850 = None
        view_980: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_239, [32, 128, 2048]);  clone_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_981: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_978, [4096, 2048]);  view_978 = None
        permute_851: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_981, [1, 0])
        mm_285: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_851, view_132);  permute_851 = None
        mm_286: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_981, permute_853);  view_981 = permute_853 = None
        view_982: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_286, [32, 128, 2048]);  mm_286 = None
        convert_element_type_2021: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_982, torch.float32);  view_982 = None
        convert_element_type_2022: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_285, torch.float32);  mm_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_240: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_979, memory_format = torch.contiguous_format);  view_979 = None
        view_983: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_240, [4096, 2048]);  clone_240 = None
        permute_855: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_983, [1, 0])
        mm_287: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_855, view_132);  permute_855 = None
        mm_288: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_983, permute_857);  view_983 = permute_857 = None
        view_984: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_288, [32, 128, 2048]);  mm_288 = None
        convert_element_type_2027: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_984, torch.float32);  view_984 = None
        add_328: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2021, convert_element_type_2027);  convert_element_type_2021 = convert_element_type_2027 = None
        convert_element_type_2028: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_287, torch.float32);  mm_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_985: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_980, [4096, 2048]);  view_980 = None
        permute_859: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_985, [1, 0])
        mm_289: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_859, view_132);  permute_859 = view_132 = None
        mm_290: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_985, permute_861);  view_985 = permute_861 = None
        view_986: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_290, [32, 128, 2048]);  mm_290 = None
        convert_element_type_2033: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_986, torch.float32);  view_986 = None
        add_329: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_328, convert_element_type_2033);  add_328 = convert_element_type_2033 = None
        convert_element_type_2034: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_289, torch.float32);  mm_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_630: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_329, primals_88);  primals_88 = None
        mul_631: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_630, 2048)
        sum_245: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_630, [2], True)
        mul_632: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_630, mul_48);  mul_630 = None
        sum_246: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_632, [2], True);  mul_632 = None
        mul_633: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, sum_246);  sum_246 = None
        sub_205: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_631, sum_245);  mul_631 = sum_245 = None
        sub_206: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_205, mul_633);  sub_205 = mul_633 = None
        mul_634: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_62, sub_206);  div_62 = sub_206 = None
        mul_635: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_329, mul_48);  mul_48 = None
        sum_247: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_635, [0, 1]);  mul_635 = None
        sum_248: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_329, [0, 1]);  add_329 = None
        add_330: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_327, mul_634);  add_327 = mul_634 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_2035: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_330, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_987: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2035, [4096, 2048]);  convert_element_type_2035 = None
        mm_291: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_987, permute_863);  permute_863 = None
        permute_864: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_987, [1, 0])
        mm_292: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_864, view_130);  permute_864 = view_130 = None
        sum_249: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_987, [0], True, dtype = torch.float32);  view_987 = None
        view_988: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_249, [2048]);  sum_249 = None
        convert_element_type_2040: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_988, torch.bfloat16);  view_988 = None
        view_989: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_291, [32, 128, 8192]);  mm_291 = None
        convert_element_type_2041: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_989, torch.float32);  view_989 = None
        convert_element_type_2042: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_292, torch.float32);  mm_292 = None
        convert_element_type_2043: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2040, torch.float32);  convert_element_type_2040 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_129: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [32, 128, 8192]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_636: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2041, mul_44);  mul_44 = None
        convert_element_type_227: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_129, torch.float32)
        pow_6: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_227, 3.0)
        mul_45: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_55: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_129, mul_45);  view_129 = mul_45 = None
        mul_46: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_5: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_56: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_637: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2041, add_56);  convert_element_type_2041 = add_56 = None
        convert_element_type_2044: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_637, torch.bfloat16);  mul_637 = None
        mul_638: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_207: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_638);  mul_638 = None
        mul_639: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_636, sub_207);  mul_636 = sub_207 = None
        mul_640: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_639, 0.7978845608028654);  mul_639 = None
        convert_element_type_2045: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_640, torch.bfloat16)
        mul_641: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_640, 0.044715);  mul_640 = None
        pow_43: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_227, 2.0);  convert_element_type_227 = None
        mul_642: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_43, 3.0);  pow_43 = None
        mul_643: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_641, mul_642);  mul_641 = mul_642 = None
        convert_element_type_2046: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_643, torch.bfloat16);  mul_643 = None
        add_331: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2045, convert_element_type_2046);  convert_element_type_2045 = convert_element_type_2046 = None
        mul_644: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2044, 0.5);  convert_element_type_2044 = None
        add_332: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_331, mul_644);  add_331 = mul_644 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_990: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_332, [4096, 8192]);  add_332 = None
        mm_293: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_990, permute_867);  permute_867 = None
        permute_868: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_990, [1, 0])
        mm_294: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_868, view_128);  permute_868 = view_128 = None
        sum_250: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_990, [0], True, dtype = torch.float32);  view_990 = None
        view_991: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_250, [8192]);  sum_250 = None
        convert_element_type_2051: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_991, torch.bfloat16);  view_991 = None
        view_992: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_293, [32, 128, 2048]);  mm_293 = None
        convert_element_type_2052: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_992, torch.float32);  view_992 = None
        convert_element_type_2053: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_294, torch.float32);  mm_294 = None
        convert_element_type_2054: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2051, torch.float32);  convert_element_type_2051 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_646: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2052, primals_82);  primals_82 = None
        mul_647: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_646, 2048)
        sum_251: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_646, [2], True)
        mul_648: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_646, mul_42);  mul_646 = None
        sum_252: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_648, [2], True);  mul_648 = None
        mul_649: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, sum_252);  sum_252 = None
        sub_209: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_647, sum_251);  mul_647 = sum_251 = None
        sub_210: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_209, mul_649);  sub_209 = mul_649 = None
        mul_650: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_63, sub_210);  div_63 = sub_210 = None
        mul_651: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2052, mul_42);  mul_42 = None
        sum_253: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_651, [0, 1]);  mul_651 = None
        sum_254: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2052, [0, 1]);  convert_element_type_2052 = None
        add_333: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_330, mul_650);  add_330 = mul_650 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_2055: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_333, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_993: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2055, [4096, 2048]);  convert_element_type_2055 = None
        mm_295: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_993, permute_871);  permute_871 = None
        permute_872: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_993, [1, 0])
        mm_296: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_872, view_126);  permute_872 = view_126 = None
        sum_255: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_993, [0], True, dtype = torch.float32);  view_993 = None
        view_994: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_255, [2048]);  sum_255 = None
        convert_element_type_2060: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_994, torch.bfloat16);  view_994 = None
        view_995: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_295, [32, 128, 2048]);  mm_295 = None
        convert_element_type_2061: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_296, torch.float32);  mm_296 = None
        convert_element_type_2062: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2060, torch.float32);  convert_element_type_2060 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_996: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_995, [32, 128, 16, 128]);  view_995 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_875: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_996, [0, 2, 1, 3]);  view_996 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_241: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_875, memory_format = torch.contiguous_format);  permute_875 = None
        view_997: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_241, [512, 128, 128]);  clone_241 = None
        bmm_120: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_876, view_997);  permute_876 = None
        bmm_121: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_997, permute_877);  view_997 = permute_877 = None
        view_998: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_120, [32, 16, 128, 128]);  bmm_120 = None
        view_999: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_121, [32, 16, 128, 128]);  bmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2067: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_999, torch.float32);  view_999 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_652: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2067, div_5);  convert_element_type_2067 = None
        sum_256: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_652, [-1], True)
        neg_19: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_18: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_19, sum_256, mul_652);  neg_19 = sum_256 = mul_652 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2068: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_18, torch.bfloat16);  fma_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_14: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_79, 2, 0, 128);  primals_79 = None
        slice_15: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_14, 3, 0, 128);  slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_47: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_15, convert_element_type_2068, full_default_32);  slice_15 = convert_element_type_2068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1000: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_47, [512, 128, 128]);  where_47 = None
        bmm_122: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_878, view_1000);  permute_878 = None
        bmm_123: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1000, permute_879);  view_1000 = permute_879 = None
        view_1001: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_122, [32, 16, 128, 128]);  bmm_122 = None
        view_1002: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_123, [32, 16, 128, 128]);  bmm_123 = None
        convert_element_type_2074: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1001, torch.float32);  view_1001 = None
        permute_880: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2074, [0, 1, 3, 2]);  convert_element_type_2074 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2075: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_880, torch.bfloat16);  permute_880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_5: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1002, torch.bfloat16);  view_1002 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_881: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_998, [0, 2, 1, 3]);  view_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_242: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_881, memory_format = torch.contiguous_format);  permute_881 = None
        view_1003: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_242, [32, 128, 2048]);  clone_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_882: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2075, [0, 2, 1, 3]);  convert_element_type_2075 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1004: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_882, [32, 128, 2048]);  permute_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_883: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_5, [0, 2, 1, 3]);  convert_element_type_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_243: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_883, memory_format = torch.contiguous_format);  permute_883 = None
        view_1005: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_243, [32, 128, 2048]);  clone_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1006: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1003, [4096, 2048]);  view_1003 = None
        permute_884: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1006, [1, 0])
        mm_297: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_884, view_110);  permute_884 = None
        mm_298: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1006, permute_886);  view_1006 = permute_886 = None
        view_1007: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_298, [32, 128, 2048]);  mm_298 = None
        convert_element_type_2081: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1007, torch.float32);  view_1007 = None
        convert_element_type_2082: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_297, torch.float32);  mm_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_244: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_1004, memory_format = torch.contiguous_format);  view_1004 = None
        view_1008: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_244, [4096, 2048]);  clone_244 = None
        permute_888: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1008, [1, 0])
        mm_299: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_888, view_110);  permute_888 = None
        mm_300: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1008, permute_890);  view_1008 = permute_890 = None
        view_1009: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_300, [32, 128, 2048]);  mm_300 = None
        convert_element_type_2087: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1009, torch.float32);  view_1009 = None
        add_334: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2081, convert_element_type_2087);  convert_element_type_2081 = convert_element_type_2087 = None
        convert_element_type_2088: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_299, torch.float32);  mm_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1010: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1005, [4096, 2048]);  view_1005 = None
        permute_892: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1010, [1, 0])
        mm_301: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_892, view_110);  permute_892 = view_110 = None
        mm_302: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1010, permute_894);  view_1010 = permute_894 = None
        view_1011: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_302, [32, 128, 2048]);  mm_302 = None
        convert_element_type_2093: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1011, torch.float32);  view_1011 = None
        add_335: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_334, convert_element_type_2093);  add_334 = convert_element_type_2093 = None
        convert_element_type_2094: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_301, torch.float32);  mm_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_654: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_335, primals_74);  primals_74 = None
        mul_655: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_654, 2048)
        sum_257: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_654, [2], True)
        mul_656: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_654, mul_40);  mul_654 = None
        sum_258: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_656, [2], True);  mul_656 = None
        mul_657: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, sum_258);  sum_258 = None
        sub_212: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_655, sum_257);  mul_655 = sum_257 = None
        sub_213: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_212, mul_657);  sub_212 = mul_657 = None
        mul_658: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_64, sub_213);  div_64 = sub_213 = None
        mul_659: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_335, mul_40);  mul_40 = None
        sum_259: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_659, [0, 1]);  mul_659 = None
        sum_260: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_335, [0, 1]);  add_335 = None
        add_336: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_333, mul_658);  add_333 = mul_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_2095: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_336, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_1012: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2095, [4096, 2048]);  convert_element_type_2095 = None
        mm_303: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_1012, permute_896);  permute_896 = None
        permute_897: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1012, [1, 0])
        mm_304: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_897, view_108);  permute_897 = view_108 = None
        sum_261: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1012, [0], True, dtype = torch.float32);  view_1012 = None
        view_1013: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_261, [2048]);  sum_261 = None
        convert_element_type_2100: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1013, torch.bfloat16);  view_1013 = None
        view_1014: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_303, [32, 128, 8192]);  mm_303 = None
        convert_element_type_2101: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1014, torch.float32);  view_1014 = None
        convert_element_type_2102: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_304, torch.float32);  mm_304 = None
        convert_element_type_2103: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2100, torch.float32);  convert_element_type_2100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_107: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [32, 128, 8192]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        mul_660: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2101, mul_36);  mul_36 = None
        convert_element_type_188: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.float32)
        pow_5: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_188, 3.0)
        mul_37: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_46: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_107, mul_37);  view_107 = mul_37 = None
        mul_38: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_4: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_47: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_661: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2101, add_47);  convert_element_type_2101 = add_47 = None
        convert_element_type_2104: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_661, torch.bfloat16);  mul_661 = None
        mul_662: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_214: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_662);  mul_662 = None
        mul_663: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_660, sub_214);  mul_660 = sub_214 = None
        mul_664: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_663, 0.7978845608028654);  mul_663 = None
        convert_element_type_2105: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_664, torch.bfloat16)
        mul_665: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_664, 0.044715);  mul_664 = None
        pow_44: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_188, 2.0);  convert_element_type_188 = None
        mul_666: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_44, 3.0);  pow_44 = None
        mul_667: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_665, mul_666);  mul_665 = mul_666 = None
        convert_element_type_2106: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_667, torch.bfloat16);  mul_667 = None
        add_337: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2105, convert_element_type_2106);  convert_element_type_2105 = convert_element_type_2106 = None
        mul_668: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2104, 0.5);  convert_element_type_2104 = None
        add_338: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_337, mul_668);  add_337 = mul_668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_1015: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_338, [4096, 8192]);  add_338 = None
        mm_305: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1015, permute_900);  permute_900 = None
        permute_901: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_1015, [1, 0])
        mm_306: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_901, view_106);  permute_901 = view_106 = None
        sum_262: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1015, [0], True, dtype = torch.float32);  view_1015 = None
        view_1016: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_262, [8192]);  sum_262 = None
        convert_element_type_2111: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1016, torch.bfloat16);  view_1016 = None
        view_1017: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_305, [32, 128, 2048]);  mm_305 = None
        convert_element_type_2112: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1017, torch.float32);  view_1017 = None
        convert_element_type_2113: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_306, torch.float32);  mm_306 = None
        convert_element_type_2114: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2111, torch.float32);  convert_element_type_2111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_670: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2112, primals_68);  primals_68 = None
        mul_671: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_670, 2048)
        sum_263: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_670, [2], True)
        mul_672: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_670, mul_34);  mul_670 = None
        sum_264: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_672, [2], True);  mul_672 = None
        mul_673: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, sum_264);  sum_264 = None
        sub_216: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_671, sum_263);  mul_671 = sum_263 = None
        sub_217: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_216, mul_673);  sub_216 = mul_673 = None
        mul_674: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_65, sub_217);  div_65 = sub_217 = None
        mul_675: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2112, mul_34);  mul_34 = None
        sum_265: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_675, [0, 1]);  mul_675 = None
        sum_266: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2112, [0, 1]);  convert_element_type_2112 = None
        add_339: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_336, mul_674);  add_336 = mul_674 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_2115: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_339, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_1018: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2115, [4096, 2048]);  convert_element_type_2115 = None
        mm_307: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1018, permute_904);  permute_904 = None
        permute_905: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1018, [1, 0])
        mm_308: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_905, view_104);  permute_905 = view_104 = None
        sum_267: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1018, [0], True, dtype = torch.float32);  view_1018 = None
        view_1019: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_267, [2048]);  sum_267 = None
        convert_element_type_2120: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1019, torch.bfloat16);  view_1019 = None
        view_1020: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_307, [32, 128, 2048]);  mm_307 = None
        convert_element_type_2121: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_308, torch.float32);  mm_308 = None
        convert_element_type_2122: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2120, torch.float32);  convert_element_type_2120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_1021: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1020, [32, 128, 16, 128]);  view_1020 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_908: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_1021, [0, 2, 1, 3]);  view_1021 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_245: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_908, memory_format = torch.contiguous_format);  permute_908 = None
        view_1022: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_245, [512, 128, 128]);  clone_245 = None
        bmm_124: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_909, view_1022);  permute_909 = None
        bmm_125: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1022, permute_910);  view_1022 = permute_910 = None
        view_1023: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_124, [32, 16, 128, 128]);  bmm_124 = None
        view_1024: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_125, [32, 16, 128, 128]);  bmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2127: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1024, torch.float32);  view_1024 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_676: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2127, div_4);  convert_element_type_2127 = None
        sum_268: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_676, [-1], True)
        neg_20: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_19: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_20, sum_268, mul_676);  neg_20 = sum_268 = mul_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2128: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_19, torch.bfloat16);  fma_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_12: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_65, 2, 0, 128);  primals_65 = None
        slice_13: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_12, 3, 0, 128);  slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_48: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_13, convert_element_type_2128, full_default_32);  slice_13 = convert_element_type_2128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1025: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_48, [512, 128, 128]);  where_48 = None
        bmm_126: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_911, view_1025);  permute_911 = None
        bmm_127: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1025, permute_912);  view_1025 = permute_912 = None
        view_1026: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_126, [32, 16, 128, 128]);  bmm_126 = None
        view_1027: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_127, [32, 16, 128, 128]);  bmm_127 = None
        convert_element_type_2134: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1026, torch.float32);  view_1026 = None
        permute_913: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2134, [0, 1, 3, 2]);  convert_element_type_2134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2135: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_913, torch.bfloat16);  permute_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_4: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1027, torch.bfloat16);  view_1027 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_914: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_1023, [0, 2, 1, 3]);  view_1023 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_246: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_914, memory_format = torch.contiguous_format);  permute_914 = None
        view_1028: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_246, [32, 128, 2048]);  clone_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_915: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2135, [0, 2, 1, 3]);  convert_element_type_2135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1029: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_915, [32, 128, 2048]);  permute_915 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_916: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_4, [0, 2, 1, 3]);  convert_element_type_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_247: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_916, memory_format = torch.contiguous_format);  permute_916 = None
        view_1030: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_247, [32, 128, 2048]);  clone_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1031: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1028, [4096, 2048]);  view_1028 = None
        permute_917: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1031, [1, 0])
        mm_309: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_917, view_88);  permute_917 = None
        mm_310: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1031, permute_919);  view_1031 = permute_919 = None
        view_1032: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_310, [32, 128, 2048]);  mm_310 = None
        convert_element_type_2141: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1032, torch.float32);  view_1032 = None
        convert_element_type_2142: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_309, torch.float32);  mm_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_248: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_1029, memory_format = torch.contiguous_format);  view_1029 = None
        view_1033: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_248, [4096, 2048]);  clone_248 = None
        permute_921: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1033, [1, 0])
        mm_311: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_921, view_88);  permute_921 = None
        mm_312: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1033, permute_923);  view_1033 = permute_923 = None
        view_1034: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_312, [32, 128, 2048]);  mm_312 = None
        convert_element_type_2147: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1034, torch.float32);  view_1034 = None
        add_340: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2141, convert_element_type_2147);  convert_element_type_2141 = convert_element_type_2147 = None
        convert_element_type_2148: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_311, torch.float32);  mm_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1035: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1030, [4096, 2048]);  view_1030 = None
        permute_925: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1035, [1, 0])
        mm_313: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_925, view_88);  permute_925 = view_88 = None
        mm_314: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1035, permute_927);  view_1035 = permute_927 = None
        view_1036: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_314, [32, 128, 2048]);  mm_314 = None
        convert_element_type_2153: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1036, torch.float32);  view_1036 = None
        add_341: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_340, convert_element_type_2153);  add_340 = convert_element_type_2153 = None
        convert_element_type_2154: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_313, torch.float32);  mm_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_678: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_341, primals_60);  primals_60 = None
        mul_679: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_678, 2048)
        sum_269: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_678, [2], True)
        mul_680: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_678, mul_32);  mul_678 = None
        sum_270: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_680, [2], True);  mul_680 = None
        mul_681: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, sum_270);  sum_270 = None
        sub_219: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_679, sum_269);  mul_679 = sum_269 = None
        sub_220: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_219, mul_681);  sub_219 = mul_681 = None
        mul_682: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_66, sub_220);  div_66 = sub_220 = None
        mul_683: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_341, mul_32);  mul_32 = None
        sum_271: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_683, [0, 1]);  mul_683 = None
        sum_272: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_341, [0, 1]);  add_341 = None
        add_342: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_339, mul_682);  add_339 = mul_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_2155: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_342, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_1037: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2155, [4096, 2048]);  convert_element_type_2155 = None
        mm_315: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_1037, permute_929);  permute_929 = None
        permute_930: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1037, [1, 0])
        mm_316: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_930, view_86);  permute_930 = view_86 = None
        sum_273: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1037, [0], True, dtype = torch.float32);  view_1037 = None
        view_1038: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_273, [2048]);  sum_273 = None
        convert_element_type_2160: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1038, torch.bfloat16);  view_1038 = None
        view_1039: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_315, [32, 128, 8192]);  mm_315 = None
        convert_element_type_2161: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1039, torch.float32);  view_1039 = None
        convert_element_type_2162: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_316, torch.float32);  mm_316 = None
        convert_element_type_2163: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2160, torch.float32);  convert_element_type_2160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_85: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [32, 128, 8192]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_85, 0.5)
        mul_684: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2161, mul_28);  mul_28 = None
        convert_element_type_149: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_85, torch.float32)
        pow_4: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_149, 3.0)
        mul_29: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_37: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_85, mul_29);  view_85 = mul_29 = None
        mul_30: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_37, 0.7978845608028654);  add_37 = None
        tanh_3: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_38: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_685: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2161, add_38);  convert_element_type_2161 = add_38 = None
        convert_element_type_2164: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_685, torch.bfloat16);  mul_685 = None
        mul_686: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_221: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_686);  mul_686 = None
        mul_687: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_684, sub_221);  mul_684 = sub_221 = None
        mul_688: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_687, 0.7978845608028654);  mul_687 = None
        convert_element_type_2165: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_688, torch.bfloat16)
        mul_689: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_688, 0.044715);  mul_688 = None
        pow_45: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_149, 2.0);  convert_element_type_149 = None
        mul_690: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_45, 3.0);  pow_45 = None
        mul_691: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_689, mul_690);  mul_689 = mul_690 = None
        convert_element_type_2166: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_691, torch.bfloat16);  mul_691 = None
        add_343: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2165, convert_element_type_2166);  convert_element_type_2165 = convert_element_type_2166 = None
        mul_692: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2164, 0.5);  convert_element_type_2164 = None
        add_344: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_343, mul_692);  add_343 = mul_692 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_1040: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_344, [4096, 8192]);  add_344 = None
        mm_317: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1040, permute_933);  permute_933 = None
        permute_934: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_1040, [1, 0])
        mm_318: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_934, view_84);  permute_934 = view_84 = None
        sum_274: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1040, [0], True, dtype = torch.float32);  view_1040 = None
        view_1041: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_274, [8192]);  sum_274 = None
        convert_element_type_2171: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1041, torch.bfloat16);  view_1041 = None
        view_1042: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_317, [32, 128, 2048]);  mm_317 = None
        convert_element_type_2172: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1042, torch.float32);  view_1042 = None
        convert_element_type_2173: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_318, torch.float32);  mm_318 = None
        convert_element_type_2174: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2171, torch.float32);  convert_element_type_2171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_694: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2172, primals_54);  primals_54 = None
        mul_695: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_694, 2048)
        sum_275: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_694, [2], True)
        mul_696: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_694, mul_26);  mul_694 = None
        sum_276: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_696, [2], True);  mul_696 = None
        mul_697: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, sum_276);  sum_276 = None
        sub_223: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_695, sum_275);  mul_695 = sum_275 = None
        sub_224: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_223, mul_697);  sub_223 = mul_697 = None
        mul_698: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_67, sub_224);  div_67 = sub_224 = None
        mul_699: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2172, mul_26);  mul_26 = None
        sum_277: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_699, [0, 1]);  mul_699 = None
        sum_278: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2172, [0, 1]);  convert_element_type_2172 = None
        add_345: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_342, mul_698);  add_342 = mul_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_2175: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_345, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_1043: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2175, [4096, 2048]);  convert_element_type_2175 = None
        mm_319: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1043, permute_937);  permute_937 = None
        permute_938: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1043, [1, 0])
        mm_320: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_938, view_82);  permute_938 = view_82 = None
        sum_279: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1043, [0], True, dtype = torch.float32);  view_1043 = None
        view_1044: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_279, [2048]);  sum_279 = None
        convert_element_type_2180: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1044, torch.bfloat16);  view_1044 = None
        view_1045: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_319, [32, 128, 2048]);  mm_319 = None
        convert_element_type_2181: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_320, torch.float32);  mm_320 = None
        convert_element_type_2182: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2180, torch.float32);  convert_element_type_2180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_1046: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1045, [32, 128, 16, 128]);  view_1045 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_941: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_1046, [0, 2, 1, 3]);  view_1046 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_249: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_941, memory_format = torch.contiguous_format);  permute_941 = None
        view_1047: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_249, [512, 128, 128]);  clone_249 = None
        bmm_128: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_942, view_1047);  permute_942 = None
        bmm_129: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1047, permute_943);  view_1047 = permute_943 = None
        view_1048: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_128, [32, 16, 128, 128]);  bmm_128 = None
        view_1049: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_129, [32, 16, 128, 128]);  bmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2187: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1049, torch.float32);  view_1049 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_700: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2187, div_3);  convert_element_type_2187 = None
        sum_280: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_700, [-1], True)
        neg_21: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_20: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_21, sum_280, mul_700);  neg_21 = sum_280 = mul_700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2188: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_20, torch.bfloat16);  fma_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_10: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_51, 2, 0, 128);  primals_51 = None
        slice_11: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_10, 3, 0, 128);  slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_49: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_11, convert_element_type_2188, full_default_32);  slice_11 = convert_element_type_2188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1050: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_49, [512, 128, 128]);  where_49 = None
        bmm_130: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_944, view_1050);  permute_944 = None
        bmm_131: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1050, permute_945);  view_1050 = permute_945 = None
        view_1051: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_130, [32, 16, 128, 128]);  bmm_130 = None
        view_1052: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_131, [32, 16, 128, 128]);  bmm_131 = None
        convert_element_type_2194: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1051, torch.float32);  view_1051 = None
        permute_946: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2194, [0, 1, 3, 2]);  convert_element_type_2194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2195: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_946, torch.bfloat16);  permute_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_3: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1052, torch.bfloat16);  view_1052 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_947: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_1048, [0, 2, 1, 3]);  view_1048 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_250: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_947, memory_format = torch.contiguous_format);  permute_947 = None
        view_1053: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_250, [32, 128, 2048]);  clone_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_948: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2195, [0, 2, 1, 3]);  convert_element_type_2195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1054: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_948, [32, 128, 2048]);  permute_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_949: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_3, [0, 2, 1, 3]);  convert_element_type_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_251: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_949, memory_format = torch.contiguous_format);  permute_949 = None
        view_1055: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_251, [32, 128, 2048]);  clone_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1056: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1053, [4096, 2048]);  view_1053 = None
        permute_950: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1056, [1, 0])
        mm_321: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_950, view_66);  permute_950 = None
        mm_322: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1056, permute_952);  view_1056 = permute_952 = None
        view_1057: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_322, [32, 128, 2048]);  mm_322 = None
        convert_element_type_2201: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1057, torch.float32);  view_1057 = None
        convert_element_type_2202: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_321, torch.float32);  mm_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_252: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_1054, memory_format = torch.contiguous_format);  view_1054 = None
        view_1058: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_252, [4096, 2048]);  clone_252 = None
        permute_954: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1058, [1, 0])
        mm_323: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_954, view_66);  permute_954 = None
        mm_324: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1058, permute_956);  view_1058 = permute_956 = None
        view_1059: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_324, [32, 128, 2048]);  mm_324 = None
        convert_element_type_2207: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1059, torch.float32);  view_1059 = None
        add_346: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2201, convert_element_type_2207);  convert_element_type_2201 = convert_element_type_2207 = None
        convert_element_type_2208: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_323, torch.float32);  mm_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1060: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1055, [4096, 2048]);  view_1055 = None
        permute_958: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1060, [1, 0])
        mm_325: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_958, view_66);  permute_958 = view_66 = None
        mm_326: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1060, permute_960);  view_1060 = permute_960 = None
        view_1061: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_326, [32, 128, 2048]);  mm_326 = None
        convert_element_type_2213: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1061, torch.float32);  view_1061 = None
        add_347: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_346, convert_element_type_2213);  add_346 = convert_element_type_2213 = None
        convert_element_type_2214: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_325, torch.float32);  mm_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_702: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_347, primals_46);  primals_46 = None
        mul_703: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_702, 2048)
        sum_281: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_702, [2], True)
        mul_704: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_702, mul_24);  mul_702 = None
        sum_282: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_704, [2], True);  mul_704 = None
        mul_705: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, sum_282);  sum_282 = None
        sub_226: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_703, sum_281);  mul_703 = sum_281 = None
        sub_227: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_226, mul_705);  sub_226 = mul_705 = None
        mul_706: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_68, sub_227);  div_68 = sub_227 = None
        mul_707: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_347, mul_24);  mul_24 = None
        sum_283: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 1]);  mul_707 = None
        sum_284: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_347, [0, 1]);  add_347 = None
        add_348: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_345, mul_706);  add_345 = mul_706 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_2215: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_348, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_1062: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2215, [4096, 2048]);  convert_element_type_2215 = None
        mm_327: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_1062, permute_962);  permute_962 = None
        permute_963: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1062, [1, 0])
        mm_328: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_963, view_64);  permute_963 = view_64 = None
        sum_285: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1062, [0], True, dtype = torch.float32);  view_1062 = None
        view_1063: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_285, [2048]);  sum_285 = None
        convert_element_type_2220: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1063, torch.bfloat16);  view_1063 = None
        view_1064: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_327, [32, 128, 8192]);  mm_327 = None
        convert_element_type_2221: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1064, torch.float32);  view_1064 = None
        convert_element_type_2222: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_328, torch.float32);  mm_328 = None
        convert_element_type_2223: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2220, torch.float32);  convert_element_type_2220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_63: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [32, 128, 8192]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_708: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2221, mul_20);  mul_20 = None
        convert_element_type_110: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32)
        pow_3: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_110, 3.0)
        mul_21: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_28: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_63, mul_21);  view_63 = mul_21 = None
        mul_22: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_28, 0.7978845608028654);  add_28 = None
        tanh_2: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_29: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_709: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2221, add_29);  convert_element_type_2221 = add_29 = None
        convert_element_type_2224: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_709, torch.bfloat16);  mul_709 = None
        mul_710: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_228: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_710);  mul_710 = None
        mul_711: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_708, sub_228);  mul_708 = sub_228 = None
        mul_712: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_711, 0.7978845608028654);  mul_711 = None
        convert_element_type_2225: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_712, torch.bfloat16)
        mul_713: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_712, 0.044715);  mul_712 = None
        pow_46: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_110, 2.0);  convert_element_type_110 = None
        mul_714: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_46, 3.0);  pow_46 = None
        mul_715: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_713, mul_714);  mul_713 = mul_714 = None
        convert_element_type_2226: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_715, torch.bfloat16);  mul_715 = None
        add_349: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2225, convert_element_type_2226);  convert_element_type_2225 = convert_element_type_2226 = None
        mul_716: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2224, 0.5);  convert_element_type_2224 = None
        add_350: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_349, mul_716);  add_349 = mul_716 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_1065: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_350, [4096, 8192]);  add_350 = None
        mm_329: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1065, permute_966);  permute_966 = None
        permute_967: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_1065, [1, 0])
        mm_330: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_967, view_62);  permute_967 = view_62 = None
        sum_286: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1065, [0], True, dtype = torch.float32);  view_1065 = None
        view_1066: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_286, [8192]);  sum_286 = None
        convert_element_type_2231: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1066, torch.bfloat16);  view_1066 = None
        view_1067: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_329, [32, 128, 2048]);  mm_329 = None
        convert_element_type_2232: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1067, torch.float32);  view_1067 = None
        convert_element_type_2233: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_330, torch.float32);  mm_330 = None
        convert_element_type_2234: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2231, torch.float32);  convert_element_type_2231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_718: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2232, primals_40);  primals_40 = None
        mul_719: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_718, 2048)
        sum_287: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_718, [2], True)
        mul_720: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_718, mul_18);  mul_718 = None
        sum_288: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_720, [2], True);  mul_720 = None
        mul_721: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, sum_288);  sum_288 = None
        sub_230: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_719, sum_287);  mul_719 = sum_287 = None
        sub_231: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_230, mul_721);  sub_230 = mul_721 = None
        mul_722: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_69, sub_231);  div_69 = sub_231 = None
        mul_723: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2232, mul_18);  mul_18 = None
        sum_289: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_723, [0, 1]);  mul_723 = None
        sum_290: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2232, [0, 1]);  convert_element_type_2232 = None
        add_351: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_348, mul_722);  add_348 = mul_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_2235: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_351, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_1068: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2235, [4096, 2048]);  convert_element_type_2235 = None
        mm_331: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1068, permute_970);  permute_970 = None
        permute_971: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1068, [1, 0])
        mm_332: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_971, view_60);  permute_971 = view_60 = None
        sum_291: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1068, [0], True, dtype = torch.float32);  view_1068 = None
        view_1069: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_291, [2048]);  sum_291 = None
        convert_element_type_2240: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1069, torch.bfloat16);  view_1069 = None
        view_1070: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_331, [32, 128, 2048]);  mm_331 = None
        convert_element_type_2241: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_332, torch.float32);  mm_332 = None
        convert_element_type_2242: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2240, torch.float32);  convert_element_type_2240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_1071: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1070, [32, 128, 16, 128]);  view_1070 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_974: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_1071, [0, 2, 1, 3]);  view_1071 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_253: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_974, memory_format = torch.contiguous_format);  permute_974 = None
        view_1072: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_253, [512, 128, 128]);  clone_253 = None
        bmm_132: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_975, view_1072);  permute_975 = None
        bmm_133: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1072, permute_976);  view_1072 = permute_976 = None
        view_1073: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_132, [32, 16, 128, 128]);  bmm_132 = None
        view_1074: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_133, [32, 16, 128, 128]);  bmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2247: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1074, torch.float32);  view_1074 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_724: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2247, div_2);  convert_element_type_2247 = None
        sum_292: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_724, [-1], True)
        neg_22: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_2);  div_2 = None
        fma_21: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_22, sum_292, mul_724);  neg_22 = sum_292 = mul_724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2248: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_21, torch.bfloat16);  fma_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_8: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_37, 2, 0, 128);  primals_37 = None
        slice_9: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_8, 3, 0, 128);  slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_50: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_9, convert_element_type_2248, full_default_32);  slice_9 = convert_element_type_2248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1075: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_50, [512, 128, 128]);  where_50 = None
        bmm_134: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_977, view_1075);  permute_977 = None
        bmm_135: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1075, permute_978);  view_1075 = permute_978 = None
        view_1076: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_134, [32, 16, 128, 128]);  bmm_134 = None
        view_1077: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_135, [32, 16, 128, 128]);  bmm_135 = None
        convert_element_type_2254: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1076, torch.float32);  view_1076 = None
        permute_979: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2254, [0, 1, 3, 2]);  convert_element_type_2254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2255: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_979, torch.bfloat16);  permute_979 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_2: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1077, torch.bfloat16);  view_1077 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_980: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_1073, [0, 2, 1, 3]);  view_1073 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_254: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_980, memory_format = torch.contiguous_format);  permute_980 = None
        view_1078: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_254, [32, 128, 2048]);  clone_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_981: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2255, [0, 2, 1, 3]);  convert_element_type_2255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1079: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_981, [32, 128, 2048]);  permute_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_982: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_2, [0, 2, 1, 3]);  convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_255: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_982, memory_format = torch.contiguous_format);  permute_982 = None
        view_1080: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_255, [32, 128, 2048]);  clone_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1081: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1078, [4096, 2048]);  view_1078 = None
        permute_983: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1081, [1, 0])
        mm_333: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_983, view_44);  permute_983 = None
        mm_334: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1081, permute_985);  view_1081 = permute_985 = None
        view_1082: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_334, [32, 128, 2048]);  mm_334 = None
        convert_element_type_2261: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1082, torch.float32);  view_1082 = None
        convert_element_type_2262: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_333, torch.float32);  mm_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_256: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_1079, memory_format = torch.contiguous_format);  view_1079 = None
        view_1083: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_256, [4096, 2048]);  clone_256 = None
        permute_987: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1083, [1, 0])
        mm_335: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_987, view_44);  permute_987 = None
        mm_336: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1083, permute_989);  view_1083 = permute_989 = None
        view_1084: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_336, [32, 128, 2048]);  mm_336 = None
        convert_element_type_2267: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1084, torch.float32);  view_1084 = None
        add_352: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2261, convert_element_type_2267);  convert_element_type_2261 = convert_element_type_2267 = None
        convert_element_type_2268: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_335, torch.float32);  mm_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1085: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1080, [4096, 2048]);  view_1080 = None
        permute_991: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1085, [1, 0])
        mm_337: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_991, view_44);  permute_991 = view_44 = None
        mm_338: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1085, permute_993);  view_1085 = permute_993 = None
        view_1086: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_338, [32, 128, 2048]);  mm_338 = None
        convert_element_type_2273: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1086, torch.float32);  view_1086 = None
        add_353: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_352, convert_element_type_2273);  add_352 = convert_element_type_2273 = None
        convert_element_type_2274: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_337, torch.float32);  mm_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_726: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_353, primals_32);  primals_32 = None
        mul_727: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_726, 2048)
        sum_293: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_726, [2], True)
        mul_728: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_726, mul_16);  mul_726 = None
        sum_294: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_728, [2], True);  mul_728 = None
        mul_729: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, sum_294);  sum_294 = None
        sub_233: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_727, sum_293);  mul_727 = sum_293 = None
        sub_234: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_233, mul_729);  sub_233 = mul_729 = None
        mul_730: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_70, sub_234);  div_70 = sub_234 = None
        mul_731: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_353, mul_16);  mul_16 = None
        sum_295: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_731, [0, 1]);  mul_731 = None
        sum_296: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_353, [0, 1]);  add_353 = None
        add_354: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_351, mul_730);  add_351 = mul_730 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_2275: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_354, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_1087: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2275, [4096, 2048]);  convert_element_type_2275 = None
        mm_339: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_1087, permute_995);  permute_995 = None
        permute_996: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1087, [1, 0])
        mm_340: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_996, view_42);  permute_996 = view_42 = None
        sum_297: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1087, [0], True, dtype = torch.float32);  view_1087 = None
        view_1088: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_297, [2048]);  sum_297 = None
        convert_element_type_2280: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1088, torch.bfloat16);  view_1088 = None
        view_1089: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_339, [32, 128, 8192]);  mm_339 = None
        convert_element_type_2281: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1089, torch.float32);  view_1089 = None
        convert_element_type_2282: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_340, torch.float32);  mm_340 = None
        convert_element_type_2283: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2280, torch.float32);  convert_element_type_2280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_41: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [32, 128, 8192]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_732: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2281, mul_12);  mul_12 = None
        convert_element_type_71: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32)
        pow_2: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_71, 3.0)
        mul_13: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_19: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_41, mul_13);  view_41 = mul_13 = None
        mul_14: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_19, 0.7978845608028654);  add_19 = None
        tanh_1: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_20: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_733: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2281, add_20);  convert_element_type_2281 = add_20 = None
        convert_element_type_2284: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_733, torch.bfloat16);  mul_733 = None
        mul_734: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_235: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_734);  mul_734 = None
        mul_735: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_732, sub_235);  mul_732 = sub_235 = None
        mul_736: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_735, 0.7978845608028654);  mul_735 = None
        convert_element_type_2285: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_736, torch.bfloat16)
        mul_737: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_736, 0.044715);  mul_736 = None
        pow_47: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_71, 2.0);  convert_element_type_71 = None
        mul_738: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_47, 3.0);  pow_47 = None
        mul_739: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_737, mul_738);  mul_737 = mul_738 = None
        convert_element_type_2286: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_739, torch.bfloat16);  mul_739 = None
        add_355: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2285, convert_element_type_2286);  convert_element_type_2285 = convert_element_type_2286 = None
        mul_740: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2284, 0.5);  convert_element_type_2284 = None
        add_356: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_355, mul_740);  add_355 = mul_740 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_1090: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_356, [4096, 8192]);  add_356 = None
        mm_341: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1090, permute_999);  permute_999 = None
        permute_1000: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_1090, [1, 0])
        mm_342: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1000, view_40);  permute_1000 = view_40 = None
        sum_298: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1090, [0], True, dtype = torch.float32);  view_1090 = None
        view_1091: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_298, [8192]);  sum_298 = None
        convert_element_type_2291: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1091, torch.bfloat16);  view_1091 = None
        view_1092: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_341, [32, 128, 2048]);  mm_341 = None
        convert_element_type_2292: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1092, torch.float32);  view_1092 = None
        convert_element_type_2293: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_342, torch.float32);  mm_342 = None
        convert_element_type_2294: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2291, torch.float32);  convert_element_type_2291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_742: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2292, primals_26);  primals_26 = None
        mul_743: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_742, 2048)
        sum_299: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_742, [2], True)
        mul_744: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_742, mul_10);  mul_742 = None
        sum_300: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_744, [2], True);  mul_744 = None
        mul_745: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, sum_300);  sum_300 = None
        sub_237: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_743, sum_299);  mul_743 = sum_299 = None
        sub_238: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_237, mul_745);  sub_237 = mul_745 = None
        mul_746: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_71, sub_238);  div_71 = sub_238 = None
        mul_747: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2292, mul_10);  mul_10 = None
        sum_301: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_747, [0, 1]);  mul_747 = None
        sum_302: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2292, [0, 1]);  convert_element_type_2292 = None
        add_357: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_354, mul_746);  add_354 = mul_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_2295: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_357, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_1093: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2295, [4096, 2048]);  convert_element_type_2295 = None
        mm_343: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1093, permute_1003);  permute_1003 = None
        permute_1004: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1093, [1, 0])
        mm_344: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1004, view_38);  permute_1004 = view_38 = None
        sum_303: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1093, [0], True, dtype = torch.float32);  view_1093 = None
        view_1094: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_303, [2048]);  sum_303 = None
        convert_element_type_2300: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1094, torch.bfloat16);  view_1094 = None
        view_1095: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_343, [32, 128, 2048]);  mm_343 = None
        convert_element_type_2301: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_344, torch.float32);  mm_344 = None
        convert_element_type_2302: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2300, torch.float32);  convert_element_type_2300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_1096: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1095, [32, 128, 16, 128]);  view_1095 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1007: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_1096, [0, 2, 1, 3]);  view_1096 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_257: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_1007, memory_format = torch.contiguous_format);  permute_1007 = None
        view_1097: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_257, [512, 128, 128]);  clone_257 = None
        bmm_136: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1008, view_1097);  permute_1008 = None
        bmm_137: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1097, permute_1009);  view_1097 = permute_1009 = None
        view_1098: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_136, [32, 16, 128, 128]);  bmm_136 = None
        view_1099: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_137, [32, 16, 128, 128]);  bmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2307: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1099, torch.float32);  view_1099 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_748: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2307, div_1);  convert_element_type_2307 = None
        sum_304: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_748, [-1], True)
        neg_23: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_22: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_23, sum_304, mul_748);  neg_23 = sum_304 = mul_748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2308: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_22, torch.bfloat16);  fma_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_6: "b8[1, 1, 128, 2048][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(primals_23, 2, 0, 128);  primals_23 = None
        slice_7: "b8[1, 1, 128, 128][4194304, 4194304, 2048, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_6, 3, 0, 128);  slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_51: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_7, convert_element_type_2308, full_default_32);  slice_7 = convert_element_type_2308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1100: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_51, [512, 128, 128]);  where_51 = None
        bmm_138: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1010, view_1100);  permute_1010 = None
        bmm_139: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1100, permute_1011);  view_1100 = permute_1011 = None
        view_1101: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_138, [32, 16, 128, 128]);  bmm_138 = None
        view_1102: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_139, [32, 16, 128, 128]);  bmm_139 = None
        convert_element_type_2314: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1101, torch.float32);  view_1101 = None
        permute_1012: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2314, [0, 1, 3, 2]);  convert_element_type_2314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2315: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1012, torch.bfloat16);  permute_1012 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default_1: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1102, torch.bfloat16);  view_1102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1013: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_1098, [0, 2, 1, 3]);  view_1098 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_258: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_1013, memory_format = torch.contiguous_format);  permute_1013 = None
        view_1103: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_258, [32, 128, 2048]);  clone_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1014: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2315, [0, 2, 1, 3]);  convert_element_type_2315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1104: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_1014, [32, 128, 2048]);  permute_1014 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1015: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default_1, [0, 2, 1, 3]);  convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_259: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_1015, memory_format = torch.contiguous_format);  permute_1015 = None
        view_1105: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_259, [32, 128, 2048]);  clone_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1106: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1103, [4096, 2048]);  view_1103 = None
        permute_1016: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1106, [1, 0])
        mm_345: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1016, view_22);  permute_1016 = None
        mm_346: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1106, permute_1018);  view_1106 = permute_1018 = None
        view_1107: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_346, [32, 128, 2048]);  mm_346 = None
        convert_element_type_2321: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1107, torch.float32);  view_1107 = None
        convert_element_type_2322: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_345, torch.float32);  mm_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_260: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_1104, memory_format = torch.contiguous_format);  view_1104 = None
        view_1108: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_260, [4096, 2048]);  clone_260 = None
        permute_1020: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1108, [1, 0])
        mm_347: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1020, view_22);  permute_1020 = None
        mm_348: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1108, permute_1022);  view_1108 = permute_1022 = None
        view_1109: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_348, [32, 128, 2048]);  mm_348 = None
        convert_element_type_2327: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1109, torch.float32);  view_1109 = None
        add_358: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2321, convert_element_type_2327);  convert_element_type_2321 = convert_element_type_2327 = None
        convert_element_type_2328: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_347, torch.float32);  mm_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1110: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1105, [4096, 2048]);  view_1105 = None
        permute_1024: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1110, [1, 0])
        mm_349: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1024, view_22);  permute_1024 = view_22 = None
        mm_350: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1110, permute_1026);  view_1110 = permute_1026 = None
        view_1111: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_350, [32, 128, 2048]);  mm_350 = None
        convert_element_type_2333: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1111, torch.float32);  view_1111 = None
        add_359: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_358, convert_element_type_2333);  add_358 = convert_element_type_2333 = None
        convert_element_type_2334: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_349, torch.float32);  mm_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_750: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_359, primals_18);  primals_18 = None
        mul_751: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_750, 2048)
        sum_305: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_750, [2], True)
        mul_752: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_750, mul_8);  mul_750 = None
        sum_306: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_752, [2], True);  mul_752 = None
        mul_753: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, sum_306);  sum_306 = None
        sub_240: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_751, sum_305);  mul_751 = sum_305 = None
        sub_241: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_240, mul_753);  sub_240 = mul_753 = None
        mul_754: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_72, sub_241);  div_72 = sub_241 = None
        mul_755: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_359, mul_8);  mul_8 = None
        sum_307: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_755, [0, 1]);  mul_755 = None
        sum_308: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_359, [0, 1]);  add_359 = None
        add_360: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_357, mul_754);  add_357 = mul_754 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:348 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_2335: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_360, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_1112: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2335, [4096, 2048]);  convert_element_type_2335 = None
        mm_351: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(view_1112, permute_1028);  permute_1028 = None
        permute_1029: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1112, [1, 0])
        mm_352: "bf16[2048, 8192][8192, 1]cuda:0" = torch.ops.aten.mm.default(permute_1029, view_20);  permute_1029 = view_20 = None
        sum_309: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1112, [0], True, dtype = torch.float32);  view_1112 = None
        view_1113: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_309, [2048]);  sum_309 = None
        convert_element_type_2340: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1113, torch.bfloat16);  view_1113 = None
        view_1114: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_351, [32, 128, 8192]);  mm_351 = None
        convert_element_type_2341: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1114, torch.float32);  view_1114 = None
        convert_element_type_2342: "f32[2048, 8192][8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_352, torch.float32);  mm_352 = None
        convert_element_type_2343: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2340, torch.float32);  convert_element_type_2340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_19: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [32, 128, 8192]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_756: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2341, mul_4);  mul_4 = None
        convert_element_type_32: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32)
        pow_1: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_32, 3.0)
        mul_5: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_10: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(view_19, mul_5);  view_19 = mul_5 = None
        mul_6: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_10, 0.7978845608028654);  add_10 = None
        tanh: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_11: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_757: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2341, add_11);  convert_element_type_2341 = add_11 = None
        convert_element_type_2344: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_757, torch.bfloat16);  mul_757 = None
        mul_758: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_242: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_758);  mul_758 = None
        mul_759: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_756, sub_242);  mul_756 = sub_242 = None
        mul_760: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_759, 0.7978845608028654);  mul_759 = None
        convert_element_type_2345: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_760, torch.bfloat16)
        mul_761: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_760, 0.044715);  mul_760 = None
        pow_48: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_32, 2.0);  convert_element_type_32 = None
        mul_762: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_48, 3.0);  pow_48 = None
        mul_763: "f32[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_761, mul_762);  mul_761 = mul_762 = None
        convert_element_type_2346: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_763, torch.bfloat16);  mul_763 = None
        add_361: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2345, convert_element_type_2346);  convert_element_type_2345 = convert_element_type_2346 = None
        mul_764: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2344, 0.5);  convert_element_type_2344 = None
        add_362: "bf16[32, 128, 8192][1048576, 8192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_361, mul_764);  add_361 = mul_764 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_1115: "bf16[4096, 8192][8192, 1]cuda:0" = torch.ops.aten.reshape.default(add_362, [4096, 8192]);  add_362 = None
        mm_353: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1115, permute_1032);  permute_1032 = None
        permute_1033: "bf16[8192, 4096][1, 8192]cuda:0" = torch.ops.aten.permute.default(view_1115, [1, 0])
        mm_354: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1033, view_18);  permute_1033 = view_18 = None
        sum_310: "f32[1, 8192][8192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1115, [0], True, dtype = torch.float32);  view_1115 = None
        view_1116: "f32[8192][1]cuda:0" = torch.ops.aten.reshape.default(sum_310, [8192]);  sum_310 = None
        convert_element_type_2351: "bf16[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1116, torch.bfloat16);  view_1116 = None
        view_1117: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_353, [32, 128, 2048]);  mm_353 = None
        convert_element_type_2352: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1117, torch.float32);  view_1117 = None
        convert_element_type_2353: "f32[8192, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_354, torch.float32);  mm_354 = None
        convert_element_type_2354: "f32[8192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2351, torch.float32);  convert_element_type_2351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_766: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2352, primals_12);  primals_12 = None
        mul_767: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_766, 2048)
        sum_311: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_766, [2], True)
        mul_768: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_766, mul_2);  mul_766 = None
        sum_312: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_768, [2], True);  mul_768 = None
        mul_769: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, sum_312);  sum_312 = None
        sub_244: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_767, sum_311);  mul_767 = sum_311 = None
        sub_245: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_244, mul_769);  sub_244 = mul_769 = None
        mul_770: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_73, sub_245);  div_73 = sub_245 = None
        mul_771: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2352, mul_2);  mul_2 = None
        sum_313: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 1]);  mul_771 = None
        sum_314: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2352, [0, 1]);  convert_element_type_2352 = None
        add_363: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_360, mul_770);  add_360 = mul_770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342 in forward, code: hidden_states = attn_output + residual
        convert_element_type_2355: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_363, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_1118: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2355, [4096, 2048]);  convert_element_type_2355 = None
        mm_355: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1118, permute_1036);  permute_1036 = None
        permute_1037: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1118, [1, 0])
        mm_356: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1037, view_16);  permute_1037 = view_16 = None
        sum_315: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1118, [0], True, dtype = torch.float32);  view_1118 = None
        view_1119: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_315, [2048]);  sum_315 = None
        convert_element_type_2360: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1119, torch.bfloat16);  view_1119 = None
        view_1120: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_355, [32, 128, 2048]);  mm_355 = None
        convert_element_type_2361: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_356, torch.float32);  mm_356 = None
        convert_element_type_2362: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2360, torch.float32);  convert_element_type_2360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_1121: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1120, [32, 128, 16, 128]);  view_1120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1040: "bf16[32, 16, 128, 128][262144, 128, 2048, 1]cuda:0" = torch.ops.aten.permute.default(view_1121, [0, 2, 1, 3]);  view_1121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_261: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_1040, memory_format = torch.contiguous_format);  permute_1040 = None
        view_1122: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_261, [512, 128, 128]);  clone_261 = None
        bmm_140: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1041, view_1122);  permute_1041 = None
        bmm_141: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1122, permute_1042);  view_1122 = permute_1042 = None
        view_1123: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_140, [32, 16, 128, 128]);  bmm_140 = None
        view_1124: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_141, [32, 16, 128, 128]);  bmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:125 in _attn, code: attn_weights = attn_weights.to(value.dtype)
        convert_element_type_2367: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1124, torch.float32);  view_1124 = None

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
        mul_772: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2367, div);  convert_element_type_2367 = None
        sum_316: "f32[32, 16, 128, 1][2048, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_772, [-1], True)
        neg_24: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div);  div = None
        fma_23: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_24, sum_316, mul_772);  neg_24 = sum_316 = mul_772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:122 in _attn, code: attn_weights = attn_weights + attention_mask
        convert_element_type_2368: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_23, torch.bfloat16);  fma_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_52: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(slice_5, convert_element_type_2368, full_default_32);  slice_5 = convert_element_type_2368 = full_default_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1125: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(where_52, [512, 128, 128]);  where_52 = None
        bmm_142: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1043, view_1125);  permute_1043 = None
        bmm_143: "bf16[512, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_1125, permute_1044);  view_1125 = permute_1044 = None
        view_1126: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_142, [32, 16, 128, 128]);  bmm_142 = None
        view_1127: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_143, [32, 16, 128, 128]);  bmm_143 = None
        convert_element_type_2374: "f32[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1126, torch.float32);  view_1126 = None
        permute_1045: "f32[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2374, [0, 1, 3, 2]);  convert_element_type_2374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:108 in _attn, code: key = key.to(torch.float32)
        convert_element_type_2375: "bf16[32, 16, 128, 128][262144, 16384, 1, 128]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1045, torch.bfloat16);  permute_1045 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:107 in _attn, code: query = query.to(torch.float32)
        convert_element_type_default: "bf16[32, 16, 128, 128][262144, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1127, torch.bfloat16);  view_1127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1046: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_1123, [0, 2, 1, 3]);  view_1123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_262: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_1046, memory_format = torch.contiguous_format);  permute_1046 = None
        view_1128: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_262, [32, 128, 2048]);  clone_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1047: "bf16[32, 128, 16, 128][262144, 1, 16384, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_2375, [0, 2, 1, 3]);  convert_element_type_2375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1129: "bf16[32, 128, 2048][262144, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_1047, [32, 128, 2048]);  permute_1047 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1048: "bf16[32, 128, 16, 128][262144, 128, 16384, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_default, [0, 2, 1, 3]);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_263: "bf16[32, 128, 16, 128][262144, 2048, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_1048, memory_format = torch.contiguous_format);  permute_1048 = None
        view_1130: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_263, [32, 128, 2048]);  clone_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1131: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1128, [4096, 2048]);  view_1128 = None
        permute_1049: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1131, [1, 0])
        mm_357: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1049, view);  permute_1049 = None
        mm_358: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1131, permute_1051);  view_1131 = permute_1051 = None
        view_1132: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_358, [32, 128, 2048]);  mm_358 = None
        convert_element_type_2381: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1132, torch.float32);  view_1132 = None
        convert_element_type_2382: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_357, torch.float32);  mm_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_264: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.clone.default(view_1129, memory_format = torch.contiguous_format);  view_1129 = None
        view_1133: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(clone_264, [4096, 2048]);  clone_264 = None
        permute_1053: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1133, [1, 0])
        mm_359: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1053, view);  permute_1053 = None
        mm_360: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1133, permute_1055);  view_1133 = permute_1055 = None
        view_1134: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_360, [32, 128, 2048]);  mm_360 = None
        convert_element_type_2387: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1134, torch.float32);  view_1134 = None
        add_364: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2381, convert_element_type_2387);  convert_element_type_2381 = convert_element_type_2387 = None
        convert_element_type_2388: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_359, torch.float32);  mm_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1135: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(view_1130, [4096, 2048]);  view_1130 = None
        permute_1057: "bf16[2048, 4096][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1135, [1, 0])
        mm_361: "bf16[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_1057, view);  permute_1057 = view = None
        mm_362: "bf16[4096, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1135, permute_1059);  view_1135 = permute_1059 = None
        view_1136: "bf16[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_362, [32, 128, 2048]);  mm_362 = None
        convert_element_type_2393: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1136, torch.float32);  view_1136 = None
        add_365: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_364, convert_element_type_2393);  add_364 = convert_element_type_2393 = None
        convert_element_type_2394: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_361, torch.float32);  mm_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_774: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_365, primals_4);  primals_4 = None
        mul_775: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_774, 2048)
        sum_317: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_774, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:463 in forward, code: hidden_states = inputs_embeds + position_embeds
        add_3: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_2: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_3, getitem_1);  add_3 = getitem_1 = None
        mul: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_776: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_774, mul);  mul_774 = None
        sum_318: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_776, [2], True);  mul_776 = None
        mul_777: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_318);  sum_318 = None
        sub_247: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_775, sum_317);  mul_775 = sum_317 = None
        sub_248: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_247, mul_777);  sub_247 = mul_777 = None
        div_74: "f32[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 2048);  rsqrt = None
        mul_778: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_74, sub_248);  div_74 = sub_248 = None
        mul_779: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_365, mul);  mul = None
        sum_319: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_779, [0, 1]);  mul_779 = None
        sum_320: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_365, [0, 1]);  add_365 = None
        add_366: "f32[32, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(add_363, mul_778);  add_363 = mul_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:463 in forward, code: hidden_states = inputs_embeds + position_embeds
        sum_321: "f32[1, 128, 2048][262144, 2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_366, [0], True, dtype = torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:462 in forward, code: position_embeds = self.wpe(position_ids)
        full_default_56: "b8[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.full.default([1, 128, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_57: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.aten.full.default([2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate: "f32[2048, 2048][2048, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_57, full_default_56, [unsqueeze], sum_321);  full_default_57 = full_default_56 = unsqueeze = sum_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:444 in forward, code: inputs_embeds = self.wte(input_ids)
        ge_1: "b8[32, 128][128, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_1, 0)
        lt_1: "b8[32, 128][128, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_1, 50257)
        bitwise_and_4: "b8[32, 128][128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_8: "b8[32, 128][128, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_1, -1)
        bitwise_and_5: "b8[32, 128][128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, ne_8);  bitwise_and_4 = ne_8 = None
        unsqueeze_13: "b8[32, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_5, -1);  bitwise_and_5 = None
        full_default_58: "f32[50257, 2048][2048, 1]cuda:0" = torch.ops.aten.full.default([50257, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_1: "f32[50257, 2048][2048, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_58, unsqueeze_13, [primals_1], add_366);  full_default_58 = unsqueeze_13 = primals_1 = add_366 = None
        return (None, _unsafe_masked_index_put_accumulate_1, _unsafe_masked_index_put_accumulate, sum_319, sum_320, convert_element_type_2394, convert_element_type_2388, convert_element_type_2382, None, convert_element_type_2361, convert_element_type_2362, sum_313, sum_314, convert_element_type_2353, convert_element_type_2354, convert_element_type_2342, convert_element_type_2343, sum_307, sum_308, convert_element_type_2334, convert_element_type_2328, convert_element_type_2322, None, convert_element_type_2301, convert_element_type_2302, sum_301, sum_302, convert_element_type_2293, convert_element_type_2294, convert_element_type_2282, convert_element_type_2283, sum_295, sum_296, convert_element_type_2274, convert_element_type_2268, convert_element_type_2262, None, convert_element_type_2241, convert_element_type_2242, sum_289, sum_290, convert_element_type_2233, convert_element_type_2234, convert_element_type_2222, convert_element_type_2223, sum_283, sum_284, convert_element_type_2214, convert_element_type_2208, convert_element_type_2202, None, convert_element_type_2181, convert_element_type_2182, sum_277, sum_278, convert_element_type_2173, convert_element_type_2174, convert_element_type_2162, convert_element_type_2163, sum_271, sum_272, convert_element_type_2154, convert_element_type_2148, convert_element_type_2142, None, convert_element_type_2121, convert_element_type_2122, sum_265, sum_266, convert_element_type_2113, convert_element_type_2114, convert_element_type_2102, convert_element_type_2103, sum_259, sum_260, convert_element_type_2094, convert_element_type_2088, convert_element_type_2082, None, convert_element_type_2061, convert_element_type_2062, sum_253, sum_254, convert_element_type_2053, convert_element_type_2054, convert_element_type_2042, convert_element_type_2043, sum_247, sum_248, convert_element_type_2034, convert_element_type_2028, convert_element_type_2022, None, convert_element_type_2001, convert_element_type_2002, sum_241, sum_242, convert_element_type_1993, convert_element_type_1994, convert_element_type_1982, convert_element_type_1983, sum_235, sum_236, convert_element_type_1974, convert_element_type_1968, convert_element_type_1962, None, convert_element_type_1941, convert_element_type_1942, sum_229, sum_230, convert_element_type_1933, convert_element_type_1934, convert_element_type_1922, convert_element_type_1923, sum_223, sum_224, convert_element_type_1914, convert_element_type_1908, convert_element_type_1902, None, convert_element_type_1881, convert_element_type_1882, sum_217, sum_218, convert_element_type_1873, convert_element_type_1874, convert_element_type_1862, convert_element_type_1863, sum_211, sum_212, convert_element_type_1854, convert_element_type_1848, convert_element_type_1842, None, convert_element_type_1821, convert_element_type_1822, sum_205, sum_206, convert_element_type_1813, convert_element_type_1814, convert_element_type_1802, convert_element_type_1803, sum_199, sum_200, convert_element_type_1794, convert_element_type_1788, convert_element_type_1782, None, convert_element_type_1761, convert_element_type_1762, sum_193, sum_194, convert_element_type_1753, convert_element_type_1754, convert_element_type_1742, convert_element_type_1743, sum_187, sum_188, convert_element_type_1734, convert_element_type_1728, convert_element_type_1722, None, convert_element_type_1701, convert_element_type_1702, sum_181, sum_182, convert_element_type_1693, convert_element_type_1694, convert_element_type_1682, convert_element_type_1683, sum_175, sum_176, convert_element_type_1674, convert_element_type_1668, convert_element_type_1662, None, convert_element_type_1641, convert_element_type_1642, sum_169, sum_170, convert_element_type_1633, convert_element_type_1634, convert_element_type_1622, convert_element_type_1623, sum_163, sum_164, convert_element_type_1614, convert_element_type_1608, convert_element_type_1602, None, convert_element_type_1581, convert_element_type_1582, sum_157, sum_158, convert_element_type_1573, convert_element_type_1574, convert_element_type_1562, convert_element_type_1563, sum_151, sum_152, convert_element_type_1554, convert_element_type_1548, convert_element_type_1542, None, convert_element_type_1521, convert_element_type_1522, sum_145, sum_146, convert_element_type_1513, convert_element_type_1514, convert_element_type_1502, convert_element_type_1503, sum_139, sum_140, convert_element_type_1494, convert_element_type_1488, convert_element_type_1482, None, convert_element_type_1461, convert_element_type_1462, sum_133, sum_134, convert_element_type_1453, convert_element_type_1454, convert_element_type_1442, convert_element_type_1443, sum_127, sum_128, convert_element_type_1434, convert_element_type_1428, convert_element_type_1422, None, convert_element_type_1401, convert_element_type_1402, sum_121, sum_122, convert_element_type_1393, convert_element_type_1394, convert_element_type_1382, convert_element_type_1383, sum_115, sum_116, convert_element_type_1374, convert_element_type_1368, convert_element_type_1362, None, convert_element_type_1341, convert_element_type_1342, sum_109, sum_110, convert_element_type_1333, convert_element_type_1334, convert_element_type_1322, convert_element_type_1323, sum_103, sum_104, convert_element_type_1314, convert_element_type_1308, convert_element_type_1302, None, convert_element_type_1281, convert_element_type_1282, sum_97, sum_98, convert_element_type_1273, convert_element_type_1274, convert_element_type_1262, convert_element_type_1263, sum_91, sum_92, convert_element_type_1254, convert_element_type_1248, convert_element_type_1242, None, convert_element_type_1221, convert_element_type_1222, sum_85, sum_86, convert_element_type_1213, convert_element_type_1214, convert_element_type_1202, convert_element_type_1203, sum_79, sum_80, convert_element_type_1194, convert_element_type_1188, convert_element_type_1182, None, convert_element_type_1161, convert_element_type_1162, sum_73, sum_74, convert_element_type_1153, convert_element_type_1154, convert_element_type_1142, convert_element_type_1143, sum_67, sum_68, convert_element_type_1134, convert_element_type_1128, convert_element_type_1122, None, convert_element_type_1101, convert_element_type_1102, sum_61, sum_62, convert_element_type_1093, convert_element_type_1094, convert_element_type_1082, convert_element_type_1083, sum_55, sum_56, convert_element_type_1074, convert_element_type_1068, convert_element_type_1062, None, convert_element_type_1041, convert_element_type_1042, sum_49, sum_50, convert_element_type_1033, convert_element_type_1034, convert_element_type_1022, convert_element_type_1023, sum_43, sum_44, convert_element_type_1014, convert_element_type_1008, convert_element_type_1002, None, convert_element_type_981, convert_element_type_982, sum_37, sum_38, convert_element_type_973, convert_element_type_974, convert_element_type_962, convert_element_type_963, sum_31, sum_32, convert_element_type_954, None)
