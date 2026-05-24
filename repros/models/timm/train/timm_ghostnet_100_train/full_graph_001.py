import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 3, 3, 3][27, 1, 9, 3]cuda:0", primals_2: "f32[512, 3, 224, 224][150528, 1, 672, 3]cuda:0", primals_6: "f32[16][1]cuda:0", primals_8: "f32[8, 16, 1, 1][16, 1, 16, 16]cuda:0", primals_12: "f32[8][1]cuda:0", primals_14: "f32[8, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_18: "f32[8][1]cuda:0", primals_19: "f32[8][1]cuda:0", primals_20: "f32[8, 16, 1, 1][16, 1, 16, 16]cuda:0", primals_24: "f32[8][1]cuda:0", primals_26: "f32[8, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_30: "f32[8][1]cuda:0", primals_32: "f32[24, 16, 1, 1][16, 1, 16, 16]cuda:0", primals_36: "f32[24][1]cuda:0", primals_38: "f32[24, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_42: "f32[24][1]cuda:0", primals_43: "f32[24][1]cuda:0", primals_44: "f32[48, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_48: "f32[48][1]cuda:0", primals_50: "f32[12, 48, 1, 1][48, 1, 48, 48]cuda:0", primals_54: "f32[12][1]cuda:0", primals_56: "f32[12, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_60: "f32[12][1]cuda:0", primals_62: "f32[16, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_66: "f32[16][1]cuda:0", primals_68: "f32[24, 16, 1, 1][16, 1, 16, 16]cuda:0", primals_72: "f32[24][1]cuda:0", primals_74: "f32[36, 24, 1, 1][24, 1, 24, 24]cuda:0", primals_78: "f32[36][1]cuda:0", primals_80: "f32[36, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_84: "f32[36][1]cuda:0", primals_85: "f32[36][1]cuda:0", primals_86: "f32[12, 72, 1, 1][72, 1, 72, 72]cuda:0", primals_90: "f32[12][1]cuda:0", primals_92: "f32[12, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_96: "f32[12][1]cuda:0", primals_98: "f32[36, 24, 1, 1][24, 1, 24, 24]cuda:0", primals_102: "f32[36][1]cuda:0", primals_104: "f32[36, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_108: "f32[36][1]cuda:0", primals_109: "f32[36][1]cuda:0", primals_110: "f32[72, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_114: "f32[72][1]cuda:0", primals_115: "f32[72][1]cuda:0", primals_116: "f32[20, 72, 1, 1][72, 1, 72, 72]cuda:0", primals_118: "f32[72, 20, 1, 1][20, 1, 20, 20]cuda:0", primals_120: "f32[20, 72, 1, 1][72, 1, 72, 72]cuda:0", primals_124: "f32[20][1]cuda:0", primals_126: "f32[20, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_130: "f32[20][1]cuda:0", primals_132: "f32[24, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_136: "f32[24][1]cuda:0", primals_138: "f32[40, 24, 1, 1][24, 1, 24, 24]cuda:0", primals_142: "f32[40][1]cuda:0", primals_144: "f32[60, 40, 1, 1][40, 1, 40, 40]cuda:0", primals_148: "f32[60][1]cuda:0", primals_150: "f32[60, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_154: "f32[60][1]cuda:0", primals_155: "f32[60][1]cuda:0", primals_156: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0", primals_158: "f32[120, 32, 1, 1][32, 1, 32, 32]cuda:0", primals_160: "f32[20, 120, 1, 1][120, 1, 120, 120]cuda:0", primals_164: "f32[20][1]cuda:0", primals_166: "f32[20, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_170: "f32[20][1]cuda:0", primals_172: "f32[120, 40, 1, 1][40, 1, 40, 40]cuda:0", primals_176: "f32[120][1]cuda:0", primals_178: "f32[120, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_182: "f32[120][1]cuda:0", primals_183: "f32[120][1]cuda:0", primals_184: "f32[240, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_188: "f32[240][1]cuda:0", primals_190: "f32[40, 240, 1, 1][240, 1, 240, 240]cuda:0", primals_194: "f32[40][1]cuda:0", primals_196: "f32[40, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_200: "f32[40][1]cuda:0", primals_202: "f32[40, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_206: "f32[40][1]cuda:0", primals_208: "f32[80, 40, 1, 1][40, 1, 40, 40]cuda:0", primals_212: "f32[80][1]cuda:0", primals_214: "f32[100, 80, 1, 1][80, 1, 80, 80]cuda:0", primals_218: "f32[100][1]cuda:0", primals_220: "f32[100, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_224: "f32[100][1]cuda:0", primals_225: "f32[100][1]cuda:0", primals_226: "f32[40, 200, 1, 1][200, 1, 200, 200]cuda:0", primals_230: "f32[40][1]cuda:0", primals_232: "f32[40, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_236: "f32[40][1]cuda:0", primals_238: "f32[92, 80, 1, 1][80, 1, 80, 80]cuda:0", primals_242: "f32[92][1]cuda:0", primals_244: "f32[92, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_248: "f32[92][1]cuda:0", primals_249: "f32[92][1]cuda:0", primals_250: "f32[40, 184, 1, 1][184, 1, 184, 184]cuda:0", primals_254: "f32[40][1]cuda:0", primals_256: "f32[40, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_260: "f32[40][1]cuda:0", primals_262: "f32[92, 80, 1, 1][80, 1, 80, 80]cuda:0", primals_266: "f32[92][1]cuda:0", primals_268: "f32[92, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_272: "f32[92][1]cuda:0", primals_273: "f32[92][1]cuda:0", primals_274: "f32[40, 184, 1, 1][184, 1, 184, 184]cuda:0", primals_278: "f32[40][1]cuda:0", primals_280: "f32[40, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_284: "f32[40][1]cuda:0", primals_286: "f32[240, 80, 1, 1][80, 1, 80, 80]cuda:0", primals_290: "f32[240][1]cuda:0", primals_292: "f32[240, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_296: "f32[240][1]cuda:0", primals_297: "f32[240][1]cuda:0", primals_298: "f32[120, 480, 1, 1][480, 1, 480, 480]cuda:0", primals_300: "f32[480, 120, 1, 1][120, 1, 120, 120]cuda:0", primals_302: "f32[56, 480, 1, 1][480, 1, 480, 480]cuda:0", primals_306: "f32[56][1]cuda:0", primals_308: "f32[56, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_312: "f32[56][1]cuda:0", primals_314: "f32[80, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_318: "f32[80][1]cuda:0", primals_320: "f32[112, 80, 1, 1][80, 1, 80, 80]cuda:0", primals_324: "f32[112][1]cuda:0", primals_326: "f32[336, 112, 1, 1][112, 1, 112, 112]cuda:0", primals_330: "f32[336][1]cuda:0", primals_332: "f32[336, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_336: "f32[336][1]cuda:0", primals_337: "f32[336][1]cuda:0", primals_338: "f32[168, 672, 1, 1][672, 1, 672, 672]cuda:0", primals_340: "f32[672, 168, 1, 1][168, 1, 168, 168]cuda:0", primals_342: "f32[56, 672, 1, 1][672, 1, 672, 672]cuda:0", primals_346: "f32[56][1]cuda:0", primals_348: "f32[56, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_352: "f32[56][1]cuda:0", primals_354: "f32[336, 112, 1, 1][112, 1, 112, 112]cuda:0", primals_358: "f32[336][1]cuda:0", primals_360: "f32[336, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_364: "f32[336][1]cuda:0", primals_365: "f32[336][1]cuda:0", primals_366: "f32[672, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_370: "f32[672][1]cuda:0", primals_371: "f32[672][1]cuda:0", primals_372: "f32[168, 672, 1, 1][672, 1, 672, 672]cuda:0", primals_374: "f32[672, 168, 1, 1][168, 1, 168, 168]cuda:0", primals_376: "f32[80, 672, 1, 1][672, 1, 672, 672]cuda:0", primals_380: "f32[80][1]cuda:0", primals_382: "f32[80, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_386: "f32[80][1]cuda:0", primals_388: "f32[112, 1, 5, 5][25, 1, 5, 1]cuda:0", primals_392: "f32[112][1]cuda:0", primals_394: "f32[160, 112, 1, 1][112, 1, 112, 112]cuda:0", primals_398: "f32[160][1]cuda:0", primals_400: "f32[480, 160, 1, 1][160, 1, 160, 160]cuda:0", primals_404: "f32[480][1]cuda:0", primals_406: "f32[480, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_410: "f32[480][1]cuda:0", primals_411: "f32[480][1]cuda:0", primals_412: "f32[80, 960, 1, 1][960, 1, 960, 960]cuda:0", primals_416: "f32[80][1]cuda:0", primals_418: "f32[80, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_422: "f32[80][1]cuda:0", primals_424: "f32[480, 160, 1, 1][160, 1, 160, 160]cuda:0", primals_428: "f32[480][1]cuda:0", primals_430: "f32[480, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_434: "f32[480][1]cuda:0", primals_435: "f32[480][1]cuda:0", primals_436: "f32[240, 960, 1, 1][960, 1, 960, 960]cuda:0", primals_438: "f32[960, 240, 1, 1][240, 1, 240, 240]cuda:0", primals_440: "f32[80, 960, 1, 1][960, 1, 960, 960]cuda:0", primals_444: "f32[80][1]cuda:0", primals_446: "f32[80, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_450: "f32[80][1]cuda:0", primals_452: "f32[480, 160, 1, 1][160, 1, 160, 160]cuda:0", primals_456: "f32[480][1]cuda:0", primals_458: "f32[480, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_462: "f32[480][1]cuda:0", primals_463: "f32[480][1]cuda:0", primals_464: "f32[80, 960, 1, 1][960, 1, 960, 960]cuda:0", primals_468: "f32[80][1]cuda:0", primals_470: "f32[80, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_474: "f32[80][1]cuda:0", primals_476: "f32[480, 160, 1, 1][160, 1, 160, 160]cuda:0", primals_480: "f32[480][1]cuda:0", primals_482: "f32[480, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_486: "f32[480][1]cuda:0", primals_487: "f32[480][1]cuda:0", primals_488: "f32[240, 960, 1, 1][960, 1, 960, 960]cuda:0", primals_490: "f32[960, 240, 1, 1][240, 1, 240, 240]cuda:0", primals_492: "f32[80, 960, 1, 1][960, 1, 960, 960]cuda:0", primals_496: "f32[80][1]cuda:0", primals_498: "f32[80, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_502: "f32[80][1]cuda:0", primals_504: "f32[960, 160, 1, 1][160, 1, 160, 160]cuda:0", primals_508: "f32[960][1]cuda:0", primals_509: "f32[960][1]cuda:0", primals_510: "f32[1280, 960, 1, 1][960, 1, 960, 960]cuda:0", primals_512: "f32[1000, 1280][1280, 1]cuda:0", convolution: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0", squeeze_1: "f32[16][1]cuda:0", relu: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0", convolution_1: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0", squeeze_4: "f32[8][1]cuda:0", relu_1: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0", convolution_2: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0", getitem_5: "f32[1, 8, 1, 1][8, 1, 8, 8]cuda:0", rsqrt_2: "f32[1, 8, 1, 1][8, 1, 8, 8]cuda:0", cat: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0", convolution_3: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0", squeeze_10: "f32[8][1]cuda:0", add_19: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0", convolution_4: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0", squeeze_13: "f32[8][1]cuda:0", add_25: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0", convolution_5: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0", squeeze_16: "f32[24][1]cuda:0", relu_3: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0", convolution_6: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0", getitem_13: "f32[1, 24, 1, 1][24, 1, 24, 24]cuda:0", rsqrt_6: "f32[1, 24, 1, 1][24, 1, 24, 24]cuda:0", cat_2: "f32[512, 48, 112, 112][602112, 1, 5376, 48]cuda:0", convolution_7: "f32[512, 48, 56, 56][150528, 1, 2688, 48]cuda:0", squeeze_22: "f32[48][1]cuda:0", add_40: "f32[512, 48, 56, 56][150528, 1, 2688, 48]cuda:0", convolution_8: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0", squeeze_25: "f32[12][1]cuda:0", add_45: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0", convolution_9: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0", squeeze_28: "f32[12][1]cuda:0", convolution_10: "f32[512, 16, 56, 56][50176, 1, 896, 16]cuda:0", squeeze_31: "f32[16][1]cuda:0", add_55: "f32[512, 16, 56, 56][50176, 1, 896, 16]cuda:0", convolution_11: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0", squeeze_34: "f32[24][1]cuda:0", add_61: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0", convolution_12: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0", squeeze_37: "f32[36][1]cuda:0", relu_5: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0", convolution_13: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0", getitem_27: "f32[1, 36, 1, 1][36, 1, 36, 36]cuda:0", rsqrt_13: "f32[1, 36, 1, 1][36, 1, 36, 36]cuda:0", cat_4: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0", convolution_14: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0", squeeze_43: "f32[12][1]cuda:0", add_76: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0", convolution_15: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0", squeeze_46: "f32[12][1]cuda:0", add_82: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0", convolution_16: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0", squeeze_49: "f32[36][1]cuda:0", relu_7: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0", convolution_17: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0", getitem_35: "f32[1, 36, 1, 1][36, 1, 36, 36]cuda:0", rsqrt_17: "f32[1, 36, 1, 1][36, 1, 36, 36]cuda:0", cat_6: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0", convolution_18: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0", getitem_37: "f32[1, 72, 1, 1][72, 1, 72, 72]cuda:0", rsqrt_18: "f32[1, 72, 1, 1][72, 1, 72, 72]cuda:0", mean: "f32[512, 72, 1, 1][72, 1, 72, 72]cuda:0", relu_9: "f32[512, 20, 1, 1][20, 1, 20, 20]cuda:0", convolution_20: "f32[512, 72, 1, 1][72, 1, 72, 72]cuda:0", mul_133: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0", convolution_21: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0", squeeze_58: "f32[20][1]cuda:0", add_103: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0", convolution_22: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0", squeeze_61: "f32[20][1]cuda:0", convolution_23: "f32[512, 24, 28, 28][18816, 1, 672, 24]cuda:0", squeeze_64: "f32[24][1]cuda:0", add_113: "f32[512, 24, 28, 28][18816, 1, 672, 24]cuda:0", convolution_24: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0", squeeze_67: "f32[40][1]cuda:0", add_119: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0", convolution_25: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0", squeeze_70: "f32[60][1]cuda:0", relu_10: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0", convolution_26: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0", getitem_49: "f32[1, 60, 1, 1][60, 1, 60, 60]cuda:0", rsqrt_24: "f32[1, 60, 1, 1][60, 1, 60, 60]cuda:0", cat_8: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0", mean_1: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0", relu_12: "f32[512, 32, 1, 1][32, 1, 32, 32]cuda:0", convolution_28: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0", mul_176: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0", convolution_29: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0", squeeze_76: "f32[20][1]cuda:0", add_135: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0", convolution_30: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0", squeeze_79: "f32[20][1]cuda:0", add_141: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0", convolution_31: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0", squeeze_82: "f32[120][1]cuda:0", relu_13: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0", convolution_32: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0", getitem_57: "f32[1, 120, 1, 1][120, 1, 120, 120]cuda:0", rsqrt_28: "f32[1, 120, 1, 1][120, 1, 120, 120]cuda:0", cat_10: "f32[512, 240, 28, 28][188160, 1, 6720, 240]cuda:0", convolution_33: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0", squeeze_88: "f32[240][1]cuda:0", add_156: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0", convolution_34: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0", squeeze_91: "f32[40][1]cuda:0", add_161: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0", convolution_35: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0", squeeze_94: "f32[40][1]cuda:0", convolution_36: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0", squeeze_97: "f32[40][1]cuda:0", add_171: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0", convolution_37: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0", squeeze_100: "f32[80][1]cuda:0", add_177: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0", convolution_38: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0", squeeze_103: "f32[100][1]cuda:0", relu_15: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0", convolution_39: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0", getitem_71: "f32[1, 100, 1, 1][100, 1, 100, 100]cuda:0", rsqrt_35: "f32[1, 100, 1, 1][100, 1, 100, 100]cuda:0", cat_12: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0", convolution_40: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0", squeeze_109: "f32[40][1]cuda:0", add_192: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0", convolution_41: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0", squeeze_112: "f32[40][1]cuda:0", add_198: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0", convolution_42: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0", squeeze_115: "f32[92][1]cuda:0", relu_17: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0", convolution_43: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0", getitem_79: "f32[1, 92, 1, 1][92, 1, 92, 92]cuda:0", rsqrt_39: "f32[1, 92, 1, 1][92, 1, 92, 92]cuda:0", cat_14: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0", convolution_44: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0", squeeze_121: "f32[40][1]cuda:0", add_213: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0", convolution_45: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0", squeeze_124: "f32[40][1]cuda:0", add_219: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0", convolution_46: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0", squeeze_127: "f32[92][1]cuda:0", relu_19: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0", convolution_47: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0", getitem_87: "f32[1, 92, 1, 1][92, 1, 92, 92]cuda:0", rsqrt_43: "f32[1, 92, 1, 1][92, 1, 92, 92]cuda:0", cat_16: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0", convolution_48: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0", squeeze_133: "f32[40][1]cuda:0", add_234: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0", convolution_49: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0", squeeze_136: "f32[40][1]cuda:0", add_240: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0", convolution_50: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0", squeeze_139: "f32[240][1]cuda:0", relu_21: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0", convolution_51: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0", getitem_95: "f32[1, 240, 1, 1][240, 1, 240, 240]cuda:0", rsqrt_47: "f32[1, 240, 1, 1][240, 1, 240, 240]cuda:0", cat_18: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0", mean_2: "f32[512, 480, 1, 1][480, 1, 480, 480]cuda:0", relu_23: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0", convolution_53: "f32[512, 480, 1, 1][480, 1, 480, 480]cuda:0", mul_338: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0", convolution_54: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0", squeeze_145: "f32[56][1]cuda:0", add_256: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0", convolution_55: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0", squeeze_148: "f32[56][1]cuda:0", convolution_56: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0", squeeze_151: "f32[80][1]cuda:0", add_266: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0", convolution_57: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0", squeeze_154: "f32[112][1]cuda:0", add_272: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0", convolution_58: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0", squeeze_157: "f32[336][1]cuda:0", relu_24: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0", convolution_59: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0", getitem_107: "f32[1, 336, 1, 1][336, 1, 336, 336]cuda:0", rsqrt_53: "f32[1, 336, 1, 1][336, 1, 336, 336]cuda:0", cat_20: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0", mean_3: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0", relu_26: "f32[512, 168, 1, 1][168, 1, 168, 168]cuda:0", convolution_61: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0", mul_381: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0", convolution_62: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0", squeeze_163: "f32[56][1]cuda:0", add_288: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0", convolution_63: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0", squeeze_166: "f32[56][1]cuda:0", add_294: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0", convolution_64: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0", squeeze_169: "f32[336][1]cuda:0", relu_27: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0", convolution_65: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0", getitem_115: "f32[1, 336, 1, 1][336, 1, 336, 336]cuda:0", rsqrt_57: "f32[1, 336, 1, 1][336, 1, 336, 336]cuda:0", cat_22: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0", convolution_66: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0", getitem_117: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", rsqrt_58: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", mean_4: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0", relu_29: "f32[512, 168, 1, 1][168, 1, 168, 168]cuda:0", convolution_68: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0", mul_417: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0", convolution_69: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0", squeeze_178: "f32[80][1]cuda:0", add_315: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0", convolution_70: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0", squeeze_181: "f32[80][1]cuda:0", convolution_71: "f32[512, 112, 7, 7][5488, 1, 784, 112]cuda:0", squeeze_184: "f32[112][1]cuda:0", add_325: "f32[512, 112, 7, 7][5488, 1, 784, 112]cuda:0", convolution_72: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0", squeeze_187: "f32[160][1]cuda:0", add_331: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0", convolution_73: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0", squeeze_190: "f32[480][1]cuda:0", relu_30: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0", convolution_74: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0", getitem_129: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", rsqrt_64: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", cat_24: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0", convolution_75: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0", squeeze_196: "f32[80][1]cuda:0", add_346: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0", convolution_76: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0", squeeze_199: "f32[80][1]cuda:0", add_352: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0", convolution_77: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0", squeeze_202: "f32[480][1]cuda:0", relu_32: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0", convolution_78: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0", getitem_137: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", rsqrt_68: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", cat_26: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0", mean_5: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0", relu_34: "f32[512, 240, 1, 1][240, 1, 240, 240]cuda:0", convolution_80: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0", mul_488: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0", convolution_81: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0", squeeze_208: "f32[80][1]cuda:0", add_368: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0", convolution_82: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0", squeeze_211: "f32[80][1]cuda:0", add_374: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0", convolution_83: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0", squeeze_214: "f32[480][1]cuda:0", relu_35: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0", convolution_84: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0", getitem_145: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", rsqrt_72: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", cat_28: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0", convolution_85: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0", squeeze_220: "f32[80][1]cuda:0", add_389: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0", convolution_86: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0", squeeze_223: "f32[80][1]cuda:0", add_395: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0", convolution_87: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0", squeeze_226: "f32[480][1]cuda:0", relu_37: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0", convolution_88: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0", getitem_153: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", rsqrt_76: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", cat_30: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0", mean_6: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0", relu_39: "f32[512, 240, 1, 1][240, 1, 240, 240]cuda:0", convolution_90: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0", mul_545: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0", convolution_91: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0", squeeze_232: "f32[80][1]cuda:0", add_411: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0", convolution_92: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0", squeeze_235: "f32[80][1]cuda:0", add_417: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0", convolution_93: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0", getitem_159: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", rsqrt_79: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", as_strided: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0", gt: "b8[512, 1280][1280, 1]cuda:0", mul_568: "f32[512, 1280][1280, 1]cuda:0", le: "b8[512, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", unsqueeze_334: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_346: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_370: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0", unsqueeze_382: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_394: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_418: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0", unsqueeze_430: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_442: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_466: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0", unsqueeze_478: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_490: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_514: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0", unsqueeze_526: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_538: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0", unsqueeze_550: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_562: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_598: "f32[1, 336, 1, 1][336, 1, 1, 1]cuda:0", unsqueeze_610: "f32[1, 56, 1, 1][56, 1, 1, 1]cuda:0", unsqueeze_622: "f32[1, 56, 1, 1][56, 1, 1, 1]cuda:0", unsqueeze_646: "f32[1, 336, 1, 1][336, 1, 1, 1]cuda:0", unsqueeze_658: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0", unsqueeze_670: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_682: "f32[1, 56, 1, 1][56, 1, 1, 1]cuda:0", unsqueeze_694: "f32[1, 56, 1, 1][56, 1, 1, 1]cuda:0", unsqueeze_718: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0", unsqueeze_730: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0", unsqueeze_742: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0", unsqueeze_766: "f32[1, 92, 1, 1][92, 1, 1, 1]cuda:0", unsqueeze_778: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0", unsqueeze_790: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0", unsqueeze_814: "f32[1, 92, 1, 1][92, 1, 1, 1]cuda:0", unsqueeze_826: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0", unsqueeze_838: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0", unsqueeze_862: "f32[1, 100, 1, 1][100, 1, 1, 1]cuda:0", unsqueeze_874: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_886: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0", unsqueeze_898: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0", unsqueeze_910: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0", unsqueeze_922: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0", unsqueeze_946: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0", unsqueeze_958: "f32[1, 20, 1, 1][20, 1, 1, 1]cuda:0", unsqueeze_970: "f32[1, 20, 1, 1][20, 1, 1, 1]cuda:0", unsqueeze_994: "f32[1, 60, 1, 1][60, 1, 1, 1]cuda:0", unsqueeze_1006: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0", unsqueeze_1018: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0", unsqueeze_1030: "f32[1, 20, 1, 1][20, 1, 1, 1]cuda:0", unsqueeze_1042: "f32[1, 20, 1, 1][20, 1, 1, 1]cuda:0", unsqueeze_1078: "f32[1, 36, 1, 1][36, 1, 1, 1]cuda:0", unsqueeze_1090: "f32[1, 12, 1, 1][12, 1, 1, 1]cuda:0", unsqueeze_1102: "f32[1, 12, 1, 1][12, 1, 1, 1]cuda:0", unsqueeze_1126: "f32[1, 36, 1, 1][36, 1, 1, 1]cuda:0", unsqueeze_1138: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0", unsqueeze_1150: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0", unsqueeze_1162: "f32[1, 12, 1, 1][12, 1, 1, 1]cuda:0", unsqueeze_1174: "f32[1, 12, 1, 1][12, 1, 1, 1]cuda:0", unsqueeze_1186: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0", unsqueeze_1210: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0", unsqueeze_1222: "f32[1, 8, 1, 1][8, 1, 1, 1]cuda:0", unsqueeze_1234: "f32[1, 8, 1, 1][8, 1, 1, 1]cuda:0", unsqueeze_1258: "f32[1, 8, 1, 1][8, 1, 1, 1]cuda:0", unsqueeze_1270: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0", tangents_1: "f32[512, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/linear.py:19 in forward, code: return F.linear(input, self.weight, self.bias)
        permute: "f32[1280, 1000][1, 1280]cuda:0" = torch.ops.aten.permute.default(primals_512, [1, 0]);  primals_512 = None
        permute_1: "f32[1000, 1280][1280, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm: "f32[512, 1280][1280, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "f32[1000, 512][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f32[1000, 1280][1280, 1]cuda:0" = torch.ops.aten.mm.default(permute_2, mul_568);  permute_2 = mul_568 = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_2: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:836 in forward_head, code: x = F.dropout(x, p=self.drop_rate, training=self.training)
        convert_element_type: "f32[512, 1280][1280, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_569: "f32[512, 1280][1280, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type, 1.25);  convert_element_type = None
        mul_570: "f32[512, 1280][1280, 1]cuda:0" = torch.ops.aten.mul.Tensor(mm, mul_569);  mm = mul_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:834 in forward_head, code: x = self.flatten(x)
        view_3: "f32[512, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_570, [512, 1280, 1, 1]);  mul_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:833 in forward_head, code: x = self.act2(x)
        full_default: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[512, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.where.self(le, full_default, view_3);  le = view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:832 in forward_head, code: x = self.conv_head(x)
        sum_2: "f32[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(where, as_strided, primals_510, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where = as_strided = primals_510 = None
        getitem_160: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward[0]
        getitem_161: "f32[1280, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward[1];  convolution_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze_240: "f32[512, 960, 1][960, 1, 960]cuda:0" = torch.ops.aten.squeeze.dim(getitem_160, 3);  getitem_160 = None
        squeeze_241: "f32[512, 960][960, 1]cuda:0" = torch.ops.aten.squeeze.dim(squeeze_240, 2);  squeeze_240 = None
        full_80: "f32[491520][1]cuda:0" = torch.ops.aten.full.default([491520], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter: "f32[491520][1]cuda:0" = torch.ops.aten.as_strided_scatter.default(full_80, squeeze_241, [512, 960], [960, 1], 0);  full_80 = squeeze_241 = None
        as_strided_5: "f32[512, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(as_strided_scatter, [512, 960, 1, 1], [960, 1, 1, 1], 0);  as_strided_scatter = None
        expand_1: "f32[512, 960, 7, 7][960, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(as_strided_5, [512, 960, 7, 7]);  as_strided_5 = None
        div_7: "f32[512, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_79: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_93, getitem_159)
        mul_560: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = None
        unsqueeze_316: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_508, -1)
        unsqueeze_317: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_566: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_560, unsqueeze_317);  mul_560 = unsqueeze_317 = None
        unsqueeze_318: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_509, -1);  primals_509 = None
        unsqueeze_319: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_422: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_566, unsqueeze_319);  mul_566 = unsqueeze_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_40: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.relu.default(add_422);  add_422 = None
        le_1: "b8[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
        where_1: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(le_1, full_default, div_7);  le_1 = div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_237: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
        unsqueeze_320: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_237, 0);  squeeze_237 = None
        unsqueeze_321: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
        unsqueeze_322: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
        sum_3: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        sub_80: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_322);  convolution_93 = unsqueeze_322 = None
        mul_571: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(where_1, sub_80)
        sum_4: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_571, [0, 2, 3]);  mul_571 = None
        mul_572: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 3.985969387755102e-05)
        unsqueeze_323: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
        unsqueeze_324: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
        unsqueeze_325: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
        mul_573: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 3.985969387755102e-05)
        squeeze_238: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
        mul_574: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
        mul_575: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_573, mul_574);  mul_573 = mul_574 = None
        unsqueeze_326: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
        unsqueeze_327: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
        unsqueeze_328: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
        mul_576: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_238, primals_508);  primals_508 = None
        unsqueeze_329: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
        unsqueeze_330: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
        unsqueeze_331: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
        mul_577: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_328);  sub_80 = unsqueeze_328 = None
        sub_82: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(where_1, mul_577);  where_1 = mul_577 = None
        sub_83: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(sub_82, unsqueeze_325);  sub_82 = unsqueeze_325 = None
        mul_578: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_331);  sub_83 = unsqueeze_331 = None
        mul_579: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, squeeze_238);  sum_4 = squeeze_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:135 in forward, code: x = self.conv(x)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_578, add_417, primals_504, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_578 = add_417 = primals_504 = None
        getitem_163: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = convolution_backward_1[0]
        getitem_164: "f32[960, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        clone_1: "f32[512, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(getitem_163, memory_format = torch.contiguous_format)
        copy_2: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.copy.default(getitem_163, clone_1);  getitem_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_3: "f32[512, 80, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.slice.Tensor(copy_2, 1, 80, 160)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_5: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_3, [0, 2, 3])
        sub_84: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_334);  convolution_92 = unsqueeze_334 = None
        mul_580: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(slice_3, sub_84)
        sum_6: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_580, [0, 2, 3]);  mul_580 = None
        mul_581: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, 3.985969387755102e-05)
        unsqueeze_335: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_581, 0);  mul_581 = None
        unsqueeze_336: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
        unsqueeze_337: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
        mul_582: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, 3.985969387755102e-05)
        mul_583: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
        mul_584: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
        unsqueeze_338: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_584, 0);  mul_584 = None
        unsqueeze_339: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
        unsqueeze_340: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
        mul_585: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_235, primals_502);  primals_502 = None
        unsqueeze_341: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
        unsqueeze_342: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
        unsqueeze_343: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
        mul_586: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_340);  sub_84 = unsqueeze_340 = None
        sub_86: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(slice_3, mul_586);  slice_3 = mul_586 = None
        sub_87: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_337);  sub_86 = unsqueeze_337 = None
        mul_587: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_343);  sub_87 = unsqueeze_343 = None
        mul_588: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, squeeze_235);  sum_6 = squeeze_235 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_587, add_411, primals_498, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False]);  mul_587 = add_411 = primals_498 = None
        getitem_166: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = convolution_backward_2[0]
        getitem_167: "f32[80, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_4: "f32[512, 80, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.slice.Tensor(copy_2, 1, 0, 80);  copy_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_423: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.add.Tensor(slice_4, getitem_166);  slice_4 = getitem_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_7: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_423, [0, 2, 3])
        sub_88: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_346);  convolution_91 = unsqueeze_346 = None
        mul_589: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(add_423, sub_88)
        sum_8: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_589, [0, 2, 3]);  mul_589 = None
        mul_590: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, 3.985969387755102e-05)
        unsqueeze_347: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_590, 0);  mul_590 = None
        unsqueeze_348: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
        unsqueeze_349: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
        mul_591: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_8, 3.985969387755102e-05)
        mul_592: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
        mul_593: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_591, mul_592);  mul_591 = mul_592 = None
        unsqueeze_350: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_593, 0);  mul_593 = None
        unsqueeze_351: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
        unsqueeze_352: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
        mul_594: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_232, primals_496);  primals_496 = None
        unsqueeze_353: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
        unsqueeze_354: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
        unsqueeze_355: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
        mul_595: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_352);  sub_88 = unsqueeze_352 = None
        sub_90: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(add_423, mul_595);  add_423 = mul_595 = None
        sub_91: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, unsqueeze_349);  sub_90 = unsqueeze_349 = None
        mul_596: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_355);  sub_91 = unsqueeze_355 = None
        mul_597: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_8, squeeze_232);  sum_8 = squeeze_232 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_596, mul_545, primals_492, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_596 = mul_545 = primals_492 = None
        getitem_169: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = convolution_backward_3[0]
        getitem_170: "f32[80, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_598: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(getitem_169, cat_30);  cat_30 = None
        add_406: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.add.Tensor(convolution_90, 3)
        clamp_min_6: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.clamp_min.default(add_406, 0);  add_406 = None
        clamp_max_6: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
        div_6: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_6, 6);  clamp_max_6 = None
        mul_599: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(getitem_169, div_6);  getitem_169 = div_6 = None
        sum_9: "f32[512, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_598, [2, 3], True);  mul_598 = None
        gt_1: "b8[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.gt.Scalar(convolution_90, -3.0)
        lt: "b8[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.lt.Scalar(convolution_90, 3.0);  convolution_90 = None
        bitwise_and: "b8[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.bitwise_and.Tensor(gt_1, lt);  gt_1 = lt = None
        mul_600: "f32[512, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, 0.16666666666666666);  sum_9 = None
        where_2: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.where.self(bitwise_and, mul_600, full_default);  bitwise_and = mul_600 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_10: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_2, relu_39, primals_490, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_2 = primals_490 = None
        getitem_172: "f32[512, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_4[0]
        getitem_173: "f32[960, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        le_2: "b8[512, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
        where_3: "f32[512, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.where.self(le_2, full_default, getitem_172);  le_2 = getitem_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_11: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(where_3, mean_6, primals_488, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_3 = mean_6 = primals_488 = None
        getitem_175: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_5[0]
        getitem_176: "f32[240, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_2: "f32[512, 960, 7, 7][960, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_175, [512, 960, 7, 7]);  getitem_175 = None
        div_8: "f32[512, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
        add_424: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_599, div_8);  mul_599 = div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_5: "f32[512, 480, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.slice.Tensor(add_424, 1, 0, 480)
        slice_6: "f32[512, 480, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.slice.Tensor(add_424, 1, 480, 960);  add_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_76: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_88, getitem_153)
        mul_538: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = None
        unsqueeze_304: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_486, -1)
        unsqueeze_305: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        mul_544: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_538, unsqueeze_305);  mul_538 = unsqueeze_305 = None
        unsqueeze_306: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_487, -1);  primals_487 = None
        unsqueeze_307: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        add_405: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_544, unsqueeze_307);  mul_544 = unsqueeze_307 = None
        relu_38: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.relu.default(add_405);  add_405 = None
        le_3: "b8[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
        where_4: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.where.self(le_3, full_default, slice_6);  le_3 = slice_6 = None
        squeeze_228: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
        unsqueeze_356: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_228, 0);  squeeze_228 = None
        unsqueeze_357: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
        unsqueeze_358: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
        sum_12: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_92: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_358);  convolution_88 = unsqueeze_358 = None
        mul_601: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(where_4, sub_92)
        sum_13: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_601, [0, 2, 3]);  mul_601 = None
        mul_602: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, 3.985969387755102e-05)
        unsqueeze_359: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
        unsqueeze_360: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
        unsqueeze_361: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
        mul_603: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, 3.985969387755102e-05)
        squeeze_229: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
        mul_604: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
        mul_605: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_603, mul_604);  mul_603 = mul_604 = None
        unsqueeze_362: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_605, 0);  mul_605 = None
        unsqueeze_363: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
        unsqueeze_364: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
        mul_606: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_229, primals_486);  primals_486 = None
        unsqueeze_365: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_606, 0);  mul_606 = None
        unsqueeze_366: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
        unsqueeze_367: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
        mul_607: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_364);  sub_92 = unsqueeze_364 = None
        sub_94: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(where_4, mul_607);  where_4 = mul_607 = None
        sub_95: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(sub_94, unsqueeze_361);  sub_94 = unsqueeze_361 = None
        mul_608: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_367);  sub_95 = unsqueeze_367 = None
        mul_609: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, squeeze_229);  sum_13 = squeeze_229 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_608, relu_37, primals_482, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_608 = primals_482 = None
        getitem_178: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = convolution_backward_6[0]
        getitem_179: "f32[480, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        add_425: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.add.Tensor(slice_5, getitem_178);  slice_5 = getitem_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_4: "b8[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
        where_5: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.where.self(le_4, full_default, add_425);  le_4 = add_425 = None
        sum_14: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        sub_96: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_370);  convolution_87 = unsqueeze_370 = None
        mul_610: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(where_5, sub_96)
        sum_15: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_610, [0, 2, 3]);  mul_610 = None
        mul_611: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_14, 3.985969387755102e-05)
        unsqueeze_371: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
        unsqueeze_372: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
        unsqueeze_373: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
        mul_612: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, 3.985969387755102e-05)
        mul_613: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
        mul_614: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_612, mul_613);  mul_612 = mul_613 = None
        unsqueeze_374: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_614, 0);  mul_614 = None
        unsqueeze_375: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
        unsqueeze_376: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
        mul_615: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_226, primals_480);  primals_480 = None
        unsqueeze_377: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_615, 0);  mul_615 = None
        unsqueeze_378: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
        unsqueeze_379: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
        mul_616: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_376);  sub_96 = unsqueeze_376 = None
        sub_98: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(where_5, mul_616);  where_5 = mul_616 = None
        sub_99: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_373);  sub_98 = unsqueeze_373 = None
        mul_617: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_379);  sub_99 = unsqueeze_379 = None
        mul_618: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, squeeze_226);  sum_15 = squeeze_226 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_617, add_395, primals_476, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_617 = add_395 = primals_476 = None
        getitem_181: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = convolution_backward_7[0]
        getitem_182: "f32[480, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        add_426: "f32[512, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(clone_1, getitem_181);  clone_1 = getitem_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        new_empty_strided_1: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.new_empty_strided.default(add_426, [512, 160, 7, 7], [7840, 1, 1120, 160])
        copy_3: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.copy.default(new_empty_strided_1, add_426);  new_empty_strided_1 = add_426 = None
        clone_2: "f32[512, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(copy_3, memory_format = torch.contiguous_format)
        copy_4: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.copy.default(copy_3, clone_2);  copy_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_9: "f32[512, 80, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.slice.Tensor(copy_4, 1, 80, 160)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_16: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_9, [0, 2, 3])
        sub_100: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_382);  convolution_86 = unsqueeze_382 = None
        mul_619: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(slice_9, sub_100)
        sum_17: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_619, [0, 2, 3]);  mul_619 = None
        mul_620: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, 3.985969387755102e-05)
        unsqueeze_383: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
        unsqueeze_384: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
        unsqueeze_385: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
        mul_621: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, 3.985969387755102e-05)
        mul_622: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
        mul_623: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_621, mul_622);  mul_621 = mul_622 = None
        unsqueeze_386: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_623, 0);  mul_623 = None
        unsqueeze_387: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
        unsqueeze_388: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
        mul_624: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_223, primals_474);  primals_474 = None
        unsqueeze_389: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
        unsqueeze_390: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
        unsqueeze_391: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
        mul_625: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_388);  sub_100 = unsqueeze_388 = None
        sub_102: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(slice_9, mul_625);  slice_9 = mul_625 = None
        sub_103: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_385);  sub_102 = unsqueeze_385 = None
        mul_626: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_391);  sub_103 = unsqueeze_391 = None
        mul_627: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, squeeze_223);  sum_17 = squeeze_223 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_626, add_389, primals_470, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False]);  mul_626 = add_389 = primals_470 = None
        getitem_184: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = convolution_backward_8[0]
        getitem_185: "f32[80, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_10: "f32[512, 80, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.slice.Tensor(copy_4, 1, 0, 80);  copy_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_427: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.add.Tensor(slice_10, getitem_184);  slice_10 = getitem_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_18: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_427, [0, 2, 3])
        sub_104: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_394);  convolution_85 = unsqueeze_394 = None
        mul_628: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(add_427, sub_104)
        sum_19: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_628, [0, 2, 3]);  mul_628 = None
        mul_629: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_18, 3.985969387755102e-05)
        unsqueeze_395: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_629, 0);  mul_629 = None
        unsqueeze_396: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
        unsqueeze_397: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
        mul_630: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, 3.985969387755102e-05)
        mul_631: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
        mul_632: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_630, mul_631);  mul_630 = mul_631 = None
        unsqueeze_398: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
        unsqueeze_399: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
        unsqueeze_400: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
        mul_633: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_220, primals_468);  primals_468 = None
        unsqueeze_401: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_633, 0);  mul_633 = None
        unsqueeze_402: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
        unsqueeze_403: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
        mul_634: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_400);  sub_104 = unsqueeze_400 = None
        sub_106: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(add_427, mul_634);  add_427 = mul_634 = None
        sub_107: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_397);  sub_106 = unsqueeze_397 = None
        mul_635: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_403);  sub_107 = unsqueeze_403 = None
        mul_636: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, squeeze_220);  sum_19 = squeeze_220 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_635, cat_28, primals_464, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_635 = cat_28 = primals_464 = None
        getitem_187: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = convolution_backward_9[0]
        getitem_188: "f32[80, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_11: "f32[512, 480, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.slice.Tensor(getitem_187, 1, 0, 480)
        slice_12: "f32[512, 480, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.slice.Tensor(getitem_187, 1, 480, 960);  getitem_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_72: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_84, getitem_145)
        mul_510: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = None
        unsqueeze_288: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_462, -1)
        unsqueeze_289: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        mul_516: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_510, unsqueeze_289);  mul_510 = unsqueeze_289 = None
        unsqueeze_290: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_463, -1);  primals_463 = None
        unsqueeze_291: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        add_384: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_516, unsqueeze_291);  mul_516 = unsqueeze_291 = None
        relu_36: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.relu.default(add_384);  add_384 = None
        le_5: "b8[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
        where_6: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.where.self(le_5, full_default, slice_12);  le_5 = slice_12 = None
        squeeze_216: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
        unsqueeze_404: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_216, 0);  squeeze_216 = None
        unsqueeze_405: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
        unsqueeze_406: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
        sum_20: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_108: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_406);  convolution_84 = unsqueeze_406 = None
        mul_637: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(where_6, sub_108)
        sum_21: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_637, [0, 2, 3]);  mul_637 = None
        mul_638: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_20, 3.985969387755102e-05)
        unsqueeze_407: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
        unsqueeze_408: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
        unsqueeze_409: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
        mul_639: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, 3.985969387755102e-05)
        squeeze_217: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_72, [0, 2, 3]);  rsqrt_72 = None
        mul_640: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
        mul_641: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_639, mul_640);  mul_639 = mul_640 = None
        unsqueeze_410: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_641, 0);  mul_641 = None
        unsqueeze_411: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
        unsqueeze_412: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
        mul_642: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_217, primals_462);  primals_462 = None
        unsqueeze_413: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_642, 0);  mul_642 = None
        unsqueeze_414: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
        unsqueeze_415: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
        mul_643: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_412);  sub_108 = unsqueeze_412 = None
        sub_110: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(where_6, mul_643);  where_6 = mul_643 = None
        sub_111: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_409);  sub_110 = unsqueeze_409 = None
        mul_644: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_415);  sub_111 = unsqueeze_415 = None
        mul_645: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, squeeze_217);  sum_21 = squeeze_217 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_644, relu_35, primals_458, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_644 = primals_458 = None
        getitem_190: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = convolution_backward_10[0]
        getitem_191: "f32[480, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        add_428: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.add.Tensor(slice_11, getitem_190);  slice_11 = getitem_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_6: "b8[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        where_7: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.where.self(le_6, full_default, add_428);  le_6 = add_428 = None
        sum_22: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        sub_112: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_418);  convolution_83 = unsqueeze_418 = None
        mul_646: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(where_7, sub_112)
        sum_23: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_646, [0, 2, 3]);  mul_646 = None
        mul_647: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_22, 3.985969387755102e-05)
        unsqueeze_419: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_647, 0);  mul_647 = None
        unsqueeze_420: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
        unsqueeze_421: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
        mul_648: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, 3.985969387755102e-05)
        mul_649: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
        mul_650: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_648, mul_649);  mul_648 = mul_649 = None
        unsqueeze_422: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_650, 0);  mul_650 = None
        unsqueeze_423: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
        unsqueeze_424: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
        mul_651: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_214, primals_456);  primals_456 = None
        unsqueeze_425: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_651, 0);  mul_651 = None
        unsqueeze_426: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
        unsqueeze_427: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
        mul_652: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_424);  sub_112 = unsqueeze_424 = None
        sub_114: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(where_7, mul_652);  where_7 = mul_652 = None
        sub_115: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_421);  sub_114 = unsqueeze_421 = None
        mul_653: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_427);  sub_115 = unsqueeze_427 = None
        mul_654: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, squeeze_214);  sum_23 = squeeze_214 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_653, add_374, primals_452, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_653 = add_374 = primals_452 = None
        getitem_193: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = convolution_backward_11[0]
        getitem_194: "f32[480, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        add_429: "f32[512, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(clone_2, getitem_193);  clone_2 = getitem_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        new_empty_strided_2: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.new_empty_strided.default(add_429, [512, 160, 7, 7], [7840, 1, 1120, 160])
        copy_5: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.copy.default(new_empty_strided_2, add_429);  new_empty_strided_2 = add_429 = None
        clone_3: "f32[512, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(copy_5, memory_format = torch.contiguous_format)
        copy_6: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.copy.default(copy_5, clone_3);  copy_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_15: "f32[512, 80, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.slice.Tensor(copy_6, 1, 80, 160)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_24: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_15, [0, 2, 3])
        sub_116: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_430);  convolution_82 = unsqueeze_430 = None
        mul_655: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(slice_15, sub_116)
        sum_25: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_655, [0, 2, 3]);  mul_655 = None
        mul_656: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_24, 3.985969387755102e-05)
        unsqueeze_431: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
        unsqueeze_432: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
        unsqueeze_433: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
        mul_657: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, 3.985969387755102e-05)
        mul_658: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
        mul_659: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_657, mul_658);  mul_657 = mul_658 = None
        unsqueeze_434: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_659, 0);  mul_659 = None
        unsqueeze_435: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
        unsqueeze_436: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
        mul_660: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_211, primals_450);  primals_450 = None
        unsqueeze_437: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_660, 0);  mul_660 = None
        unsqueeze_438: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
        unsqueeze_439: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
        mul_661: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_436);  sub_116 = unsqueeze_436 = None
        sub_118: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(slice_15, mul_661);  slice_15 = mul_661 = None
        sub_119: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_433);  sub_118 = unsqueeze_433 = None
        mul_662: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_439);  sub_119 = unsqueeze_439 = None
        mul_663: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, squeeze_211);  sum_25 = squeeze_211 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_662, add_368, primals_446, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False]);  mul_662 = add_368 = primals_446 = None
        getitem_196: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = convolution_backward_12[0]
        getitem_197: "f32[80, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_16: "f32[512, 80, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.slice.Tensor(copy_6, 1, 0, 80);  copy_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_430: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.add.Tensor(slice_16, getitem_196);  slice_16 = getitem_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_26: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_430, [0, 2, 3])
        sub_120: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_442);  convolution_81 = unsqueeze_442 = None
        mul_664: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(add_430, sub_120)
        sum_27: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_664, [0, 2, 3]);  mul_664 = None
        mul_665: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_26, 3.985969387755102e-05)
        unsqueeze_443: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
        unsqueeze_444: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
        unsqueeze_445: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
        mul_666: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, 3.985969387755102e-05)
        mul_667: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
        mul_668: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_666, mul_667);  mul_666 = mul_667 = None
        unsqueeze_446: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_668, 0);  mul_668 = None
        unsqueeze_447: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
        unsqueeze_448: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
        mul_669: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_208, primals_444);  primals_444 = None
        unsqueeze_449: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_669, 0);  mul_669 = None
        unsqueeze_450: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
        unsqueeze_451: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
        mul_670: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_448);  sub_120 = unsqueeze_448 = None
        sub_122: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(add_430, mul_670);  add_430 = mul_670 = None
        sub_123: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_445);  sub_122 = unsqueeze_445 = None
        mul_671: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_451);  sub_123 = unsqueeze_451 = None
        mul_672: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, squeeze_208);  sum_27 = squeeze_208 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_671, mul_488, primals_440, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_671 = mul_488 = primals_440 = None
        getitem_199: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = convolution_backward_13[0]
        getitem_200: "f32[80, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_673: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(getitem_199, cat_26);  cat_26 = None
        add_363: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.add.Tensor(convolution_80, 3)
        clamp_min_5: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.clamp_min.default(add_363, 0);  add_363 = None
        clamp_max_5: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
        div_5: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_5, 6);  clamp_max_5 = None
        mul_674: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(getitem_199, div_5);  getitem_199 = div_5 = None
        sum_28: "f32[512, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_673, [2, 3], True);  mul_673 = None
        gt_2: "b8[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.gt.Scalar(convolution_80, -3.0)
        lt_1: "b8[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.lt.Scalar(convolution_80, 3.0);  convolution_80 = None
        bitwise_and_1: "b8[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.bitwise_and.Tensor(gt_2, lt_1);  gt_2 = lt_1 = None
        mul_675: "f32[512, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, 0.16666666666666666);  sum_28 = None
        where_8: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.where.self(bitwise_and_1, mul_675, full_default);  bitwise_and_1 = mul_675 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_29: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_8, relu_34, primals_438, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_8 = primals_438 = None
        getitem_202: "f32[512, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_14[0]
        getitem_203: "f32[960, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        le_7: "b8[512, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        where_9: "f32[512, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.where.self(le_7, full_default, getitem_202);  le_7 = getitem_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_30: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(where_9, mean_5, primals_436, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_9 = mean_5 = primals_436 = None
        getitem_205: "f32[512, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_15[0]
        getitem_206: "f32[240, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_3: "f32[512, 960, 7, 7][960, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_205, [512, 960, 7, 7]);  getitem_205 = None
        div_9: "f32[512, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
        add_431: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_674, div_9);  mul_674 = div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_17: "f32[512, 480, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.slice.Tensor(add_431, 1, 0, 480)
        slice_18: "f32[512, 480, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.slice.Tensor(add_431, 1, 480, 960);  add_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_68: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_78, getitem_137)
        mul_481: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = None
        unsqueeze_272: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_434, -1)
        unsqueeze_273: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        mul_487: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_481, unsqueeze_273);  mul_481 = unsqueeze_273 = None
        unsqueeze_274: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_435, -1);  primals_435 = None
        unsqueeze_275: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        add_362: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_487, unsqueeze_275);  mul_487 = unsqueeze_275 = None
        relu_33: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.relu.default(add_362);  add_362 = None
        le_8: "b8[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        where_10: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.where.self(le_8, full_default, slice_18);  le_8 = slice_18 = None
        squeeze_204: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
        unsqueeze_452: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_204, 0);  squeeze_204 = None
        unsqueeze_453: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
        unsqueeze_454: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
        sum_31: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_124: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_454);  convolution_78 = unsqueeze_454 = None
        mul_676: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(where_10, sub_124)
        sum_32: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_676, [0, 2, 3]);  mul_676 = None
        mul_677: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, 3.985969387755102e-05)
        unsqueeze_455: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_677, 0);  mul_677 = None
        unsqueeze_456: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
        unsqueeze_457: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
        mul_678: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_32, 3.985969387755102e-05)
        squeeze_205: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
        mul_679: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
        mul_680: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_678, mul_679);  mul_678 = mul_679 = None
        unsqueeze_458: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
        unsqueeze_459: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
        unsqueeze_460: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
        mul_681: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_205, primals_434);  primals_434 = None
        unsqueeze_461: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
        unsqueeze_462: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
        unsqueeze_463: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
        mul_682: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_460);  sub_124 = unsqueeze_460 = None
        sub_126: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(where_10, mul_682);  where_10 = mul_682 = None
        sub_127: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_457);  sub_126 = unsqueeze_457 = None
        mul_683: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_463);  sub_127 = unsqueeze_463 = None
        mul_684: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_32, squeeze_205);  sum_32 = squeeze_205 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_683, relu_32, primals_430, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_683 = primals_430 = None
        getitem_208: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = convolution_backward_16[0]
        getitem_209: "f32[480, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        add_432: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.add.Tensor(slice_17, getitem_208);  slice_17 = getitem_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_9: "b8[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
        where_11: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.where.self(le_9, full_default, add_432);  le_9 = add_432 = None
        sum_33: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        sub_128: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_466);  convolution_77 = unsqueeze_466 = None
        mul_685: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(where_11, sub_128)
        sum_34: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_685, [0, 2, 3]);  mul_685 = None
        mul_686: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, 3.985969387755102e-05)
        unsqueeze_467: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
        unsqueeze_468: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
        unsqueeze_469: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
        mul_687: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, 3.985969387755102e-05)
        mul_688: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
        mul_689: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_687, mul_688);  mul_687 = mul_688 = None
        unsqueeze_470: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
        unsqueeze_471: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
        unsqueeze_472: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
        mul_690: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_202, primals_428);  primals_428 = None
        unsqueeze_473: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
        unsqueeze_474: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
        unsqueeze_475: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
        mul_691: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_472);  sub_128 = unsqueeze_472 = None
        sub_130: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(where_11, mul_691);  where_11 = mul_691 = None
        sub_131: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_469);  sub_130 = unsqueeze_469 = None
        mul_692: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_475);  sub_131 = unsqueeze_475 = None
        mul_693: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, squeeze_202);  sum_34 = squeeze_202 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_692, add_352, primals_424, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_692 = add_352 = primals_424 = None
        getitem_211: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = convolution_backward_17[0]
        getitem_212: "f32[480, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        add_433: "f32[512, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(clone_3, getitem_211);  clone_3 = getitem_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        new_empty_strided_3: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.new_empty_strided.default(add_433, [512, 160, 7, 7], [7840, 1, 1120, 160])
        copy_7: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.copy.default(new_empty_strided_3, add_433);  new_empty_strided_3 = add_433 = None
        clone_4: "f32[512, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(copy_7, memory_format = torch.contiguous_format)
        copy_8: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.copy.default(copy_7, clone_4);  copy_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_21: "f32[512, 80, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.slice.Tensor(copy_8, 1, 80, 160)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_35: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_21, [0, 2, 3])
        sub_132: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_478);  convolution_76 = unsqueeze_478 = None
        mul_694: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(slice_21, sub_132)
        sum_36: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_694, [0, 2, 3]);  mul_694 = None
        mul_695: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, 3.985969387755102e-05)
        unsqueeze_479: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
        unsqueeze_480: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
        unsqueeze_481: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
        mul_696: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_36, 3.985969387755102e-05)
        mul_697: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
        mul_698: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_696, mul_697);  mul_696 = mul_697 = None
        unsqueeze_482: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
        unsqueeze_483: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
        unsqueeze_484: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
        mul_699: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_199, primals_422);  primals_422 = None
        unsqueeze_485: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
        unsqueeze_486: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
        unsqueeze_487: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
        mul_700: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_484);  sub_132 = unsqueeze_484 = None
        sub_134: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(slice_21, mul_700);  slice_21 = mul_700 = None
        sub_135: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_481);  sub_134 = unsqueeze_481 = None
        mul_701: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_487);  sub_135 = unsqueeze_487 = None
        mul_702: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_36, squeeze_199);  sum_36 = squeeze_199 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_701, add_346, primals_418, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False]);  mul_701 = add_346 = primals_418 = None
        getitem_214: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = convolution_backward_18[0]
        getitem_215: "f32[80, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_22: "f32[512, 80, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.slice.Tensor(copy_8, 1, 0, 80);  copy_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_434: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.add.Tensor(slice_22, getitem_214);  slice_22 = getitem_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_37: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_434, [0, 2, 3])
        sub_136: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_490);  convolution_75 = unsqueeze_490 = None
        mul_703: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(add_434, sub_136)
        sum_38: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_703, [0, 2, 3]);  mul_703 = None
        mul_704: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, 3.985969387755102e-05)
        unsqueeze_491: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_704, 0);  mul_704 = None
        unsqueeze_492: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
        unsqueeze_493: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
        mul_705: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_38, 3.985969387755102e-05)
        mul_706: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
        mul_707: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_705, mul_706);  mul_705 = mul_706 = None
        unsqueeze_494: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_707, 0);  mul_707 = None
        unsqueeze_495: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
        unsqueeze_496: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
        mul_708: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_196, primals_416);  primals_416 = None
        unsqueeze_497: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
        unsqueeze_498: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
        unsqueeze_499: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
        mul_709: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_496);  sub_136 = unsqueeze_496 = None
        sub_138: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(add_434, mul_709);  add_434 = mul_709 = None
        sub_139: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_493);  sub_138 = unsqueeze_493 = None
        mul_710: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_499);  sub_139 = unsqueeze_499 = None
        mul_711: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_38, squeeze_196);  sum_38 = squeeze_196 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_710, cat_24, primals_412, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_710 = cat_24 = primals_412 = None
        getitem_217: "f32[512, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = convolution_backward_19[0]
        getitem_218: "f32[80, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_23: "f32[512, 480, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.slice.Tensor(getitem_217, 1, 0, 480)
        slice_24: "f32[512, 480, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.slice.Tensor(getitem_217, 1, 480, 960);  getitem_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_64: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_74, getitem_129)
        mul_453: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
        unsqueeze_256: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_410, -1)
        unsqueeze_257: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        mul_459: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_453, unsqueeze_257);  mul_453 = unsqueeze_257 = None
        unsqueeze_258: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_411, -1);  primals_411 = None
        unsqueeze_259: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        add_341: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_459, unsqueeze_259);  mul_459 = unsqueeze_259 = None
        relu_31: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.relu.default(add_341);  add_341 = None
        le_10: "b8[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        where_12: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.where.self(le_10, full_default, slice_24);  le_10 = slice_24 = None
        squeeze_192: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
        unsqueeze_500: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
        unsqueeze_501: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
        unsqueeze_502: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
        sum_39: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        sub_140: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_502);  convolution_74 = unsqueeze_502 = None
        mul_712: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(where_12, sub_140)
        sum_40: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_712, [0, 2, 3]);  mul_712 = None
        mul_713: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, 3.985969387755102e-05)
        unsqueeze_503: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_713, 0);  mul_713 = None
        unsqueeze_504: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
        unsqueeze_505: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
        mul_714: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, 3.985969387755102e-05)
        squeeze_193: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
        mul_715: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
        mul_716: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_714, mul_715);  mul_714 = mul_715 = None
        unsqueeze_506: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_716, 0);  mul_716 = None
        unsqueeze_507: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
        unsqueeze_508: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
        mul_717: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_193, primals_410);  primals_410 = None
        unsqueeze_509: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
        unsqueeze_510: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
        unsqueeze_511: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
        mul_718: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_508);  sub_140 = unsqueeze_508 = None
        sub_142: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(where_12, mul_718);  where_12 = mul_718 = None
        sub_143: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_505);  sub_142 = unsqueeze_505 = None
        mul_719: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_511);  sub_143 = unsqueeze_511 = None
        mul_720: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, squeeze_193);  sum_40 = squeeze_193 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_719, relu_30, primals_406, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_719 = primals_406 = None
        getitem_220: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = convolution_backward_20[0]
        getitem_221: "f32[480, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        add_435: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.add.Tensor(slice_23, getitem_220);  slice_23 = getitem_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_11: "b8[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        where_13: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.where.self(le_11, full_default, add_435);  le_11 = add_435 = None
        sum_41: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        sub_144: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_514);  convolution_73 = unsqueeze_514 = None
        mul_721: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(where_13, sub_144)
        sum_42: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_721, [0, 2, 3]);  mul_721 = None
        mul_722: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, 3.985969387755102e-05)
        unsqueeze_515: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
        unsqueeze_516: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
        unsqueeze_517: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
        mul_723: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, 3.985969387755102e-05)
        mul_724: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
        mul_725: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_723, mul_724);  mul_723 = mul_724 = None
        unsqueeze_518: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_725, 0);  mul_725 = None
        unsqueeze_519: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
        unsqueeze_520: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
        mul_726: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_190, primals_404);  primals_404 = None
        unsqueeze_521: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
        unsqueeze_522: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
        unsqueeze_523: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
        mul_727: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_520);  sub_144 = unsqueeze_520 = None
        sub_146: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(where_13, mul_727);  where_13 = mul_727 = None
        sub_147: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_517);  sub_146 = unsqueeze_517 = None
        mul_728: "f32[512, 480, 7, 7][23520, 1, 3360, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_523);  sub_147 = unsqueeze_523 = None
        mul_729: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, squeeze_190);  sum_42 = squeeze_190 = None
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_728, add_331, primals_400, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_728 = add_331 = primals_400 = None
        getitem_223: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = convolution_backward_21[0]
        getitem_224: "f32[480, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        add_436: "f32[512, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(clone_4, getitem_223);  clone_4 = getitem_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        new_empty_strided_4: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.new_empty_strided.default(add_436, [512, 160, 7, 7], [7840, 1, 1120, 160])
        copy_9: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.copy.default(new_empty_strided_4, add_436);  new_empty_strided_4 = add_436 = None
        clone_5: "f32[512, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(copy_9, memory_format = torch.contiguous_format)
        copy_10: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.copy.default(copy_9, clone_5);  copy_9 = None
        sum_43: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(clone_5, [0, 2, 3])
        sub_148: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_526);  convolution_72 = unsqueeze_526 = None
        mul_730: "f32[512, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(clone_5, sub_148)
        sum_44: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_730, [0, 2, 3]);  mul_730 = None
        mul_731: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, 3.985969387755102e-05)
        unsqueeze_527: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_731, 0);  mul_731 = None
        unsqueeze_528: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
        unsqueeze_529: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
        mul_732: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_44, 3.985969387755102e-05)
        mul_733: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
        mul_734: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_732, mul_733);  mul_732 = mul_733 = None
        unsqueeze_530: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
        unsqueeze_531: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
        unsqueeze_532: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
        mul_735: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_187, primals_398);  primals_398 = None
        unsqueeze_533: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_735, 0);  mul_735 = None
        unsqueeze_534: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
        unsqueeze_535: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
        mul_736: "f32[512, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_532);  sub_148 = unsqueeze_532 = None
        sub_150: "f32[512, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_5, mul_736);  clone_5 = mul_736 = None
        sub_151: "f32[512, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_529);  sub_150 = unsqueeze_529 = None
        mul_737: "f32[512, 160, 7, 7][7840, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_535);  sub_151 = unsqueeze_535 = None
        mul_738: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_44, squeeze_187);  sum_44 = squeeze_187 = None
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_737, add_325, primals_394, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_737 = add_325 = primals_394 = None
        getitem_226: "f32[512, 112, 7, 7][5488, 1, 784, 112]cuda:0" = convolution_backward_22[0]
        getitem_227: "f32[160, 112, 1, 1][112, 1, 112, 112]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        sum_45: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_226, [0, 2, 3])
        sub_152: "f32[512, 112, 7, 7][5488, 1, 784, 112]cuda:0" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_538);  convolution_71 = unsqueeze_538 = None
        mul_739: "f32[512, 112, 7, 7][5488, 1, 784, 112]cuda:0" = torch.ops.aten.mul.Tensor(getitem_226, sub_152)
        sum_46: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_739, [0, 2, 3]);  mul_739 = None
        mul_740: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, 3.985969387755102e-05)
        unsqueeze_539: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
        unsqueeze_540: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
        unsqueeze_541: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
        mul_741: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, 3.985969387755102e-05)
        mul_742: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
        mul_743: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_741, mul_742);  mul_741 = mul_742 = None
        unsqueeze_542: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
        unsqueeze_543: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
        unsqueeze_544: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
        mul_744: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_184, primals_392);  primals_392 = None
        unsqueeze_545: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
        unsqueeze_546: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
        unsqueeze_547: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
        mul_745: "f32[512, 112, 7, 7][5488, 1, 784, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_544);  sub_152 = unsqueeze_544 = None
        sub_154: "f32[512, 112, 7, 7][5488, 1, 784, 112]cuda:0" = torch.ops.aten.sub.Tensor(getitem_226, mul_745);  getitem_226 = mul_745 = None
        sub_155: "f32[512, 112, 7, 7][5488, 1, 784, 112]cuda:0" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_541);  sub_154 = unsqueeze_541 = None
        mul_746: "f32[512, 112, 7, 7][5488, 1, 784, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_547);  sub_155 = unsqueeze_547 = None
        mul_747: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, squeeze_184);  sum_46 = squeeze_184 = None
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_746, add_294, primals_388, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 112, [True, True, False]);  mul_746 = primals_388 = None
        getitem_229: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = convolution_backward_23[0]
        getitem_230: "f32[112, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_27: "f32[512, 80, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.slice.Tensor(copy_10, 1, 80, 160)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_47: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_27, [0, 2, 3])
        sub_156: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_550);  convolution_70 = unsqueeze_550 = None
        mul_748: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(slice_27, sub_156)
        sum_48: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_748, [0, 2, 3]);  mul_748 = None
        mul_749: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, 3.985969387755102e-05)
        unsqueeze_551: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
        unsqueeze_552: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
        unsqueeze_553: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
        mul_750: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, 3.985969387755102e-05)
        mul_751: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
        mul_752: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_750, mul_751);  mul_750 = mul_751 = None
        unsqueeze_554: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
        unsqueeze_555: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
        unsqueeze_556: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
        mul_753: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_181, primals_386);  primals_386 = None
        unsqueeze_557: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
        unsqueeze_558: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
        unsqueeze_559: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
        mul_754: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_556);  sub_156 = unsqueeze_556 = None
        sub_158: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(slice_27, mul_754);  slice_27 = mul_754 = None
        sub_159: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_553);  sub_158 = unsqueeze_553 = None
        mul_755: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_559);  sub_159 = unsqueeze_559 = None
        mul_756: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, squeeze_181);  sum_48 = squeeze_181 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_755, add_315, primals_382, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False]);  mul_755 = add_315 = primals_382 = None
        getitem_232: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = convolution_backward_24[0]
        getitem_233: "f32[80, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_28: "f32[512, 80, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.slice.Tensor(copy_10, 1, 0, 80);  copy_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_437: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.add.Tensor(slice_28, getitem_232);  slice_28 = getitem_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_49: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_437, [0, 2, 3])
        sub_160: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_562);  convolution_69 = unsqueeze_562 = None
        mul_757: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(add_437, sub_160)
        sum_50: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_757, [0, 2, 3]);  mul_757 = None
        mul_758: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, 3.985969387755102e-05)
        unsqueeze_563: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
        unsqueeze_564: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 2);  unsqueeze_563 = None
        unsqueeze_565: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 3);  unsqueeze_564 = None
        mul_759: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, 3.985969387755102e-05)
        mul_760: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
        mul_761: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_759, mul_760);  mul_759 = mul_760 = None
        unsqueeze_566: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
        unsqueeze_567: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
        unsqueeze_568: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
        mul_762: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_178, primals_380);  primals_380 = None
        unsqueeze_569: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
        unsqueeze_570: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
        unsqueeze_571: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
        mul_763: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_568);  sub_160 = unsqueeze_568 = None
        sub_162: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(add_437, mul_763);  add_437 = mul_763 = None
        sub_163: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_162, unsqueeze_565);  sub_162 = unsqueeze_565 = None
        mul_764: "f32[512, 80, 7, 7][3920, 1, 560, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_571);  sub_163 = unsqueeze_571 = None
        mul_765: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, squeeze_178);  sum_50 = squeeze_178 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_764, mul_417, primals_376, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_764 = mul_417 = primals_376 = None
        getitem_235: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = convolution_backward_25[0]
        getitem_236: "f32[80, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        sub_58: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_66, getitem_117)
        mul_410: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
        unsqueeze_232: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_370, -1)
        unsqueeze_233: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        mul_416: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_410, unsqueeze_233);  mul_410 = unsqueeze_233 = None
        unsqueeze_234: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_371, -1);  primals_371 = None
        unsqueeze_235: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        add_309: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_416, unsqueeze_235);  mul_416 = unsqueeze_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_766: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(getitem_235, add_309);  add_309 = None
        add_310: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.add.Tensor(convolution_68, 3)
        clamp_min_4: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.clamp_min.default(add_310, 0);  add_310 = None
        clamp_max_4: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
        div_4: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_4, 6);  clamp_max_4 = None
        mul_767: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(getitem_235, div_4);  getitem_235 = div_4 = None
        sum_51: "f32[512, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_766, [2, 3], True);  mul_766 = None
        gt_3: "b8[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.gt.Scalar(convolution_68, -3.0)
        lt_2: "b8[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.lt.Scalar(convolution_68, 3.0);  convolution_68 = None
        bitwise_and_2: "b8[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.bitwise_and.Tensor(gt_3, lt_2);  gt_3 = lt_2 = None
        mul_768: "f32[512, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, 0.16666666666666666);  sum_51 = None
        where_14: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.where.self(bitwise_and_2, mul_768, full_default);  bitwise_and_2 = mul_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_52: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(where_14, relu_29, primals_374, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_14 = primals_374 = None
        getitem_238: "f32[512, 168, 1, 1][168, 1, 168, 168]cuda:0" = convolution_backward_26[0]
        getitem_239: "f32[672, 168, 1, 1][168, 1, 168, 168]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        le_12: "b8[512, 168, 1, 1][168, 1, 168, 168]cuda:0" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        where_15: "f32[512, 168, 1, 1][168, 1, 168, 168]cuda:0" = torch.ops.aten.where.self(le_12, full_default, getitem_238);  le_12 = getitem_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_53: "f32[168][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_15, mean_4, primals_372, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_15 = mean_4 = primals_372 = None
        getitem_241: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_27[0]
        getitem_242: "f32[168, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_4: "f32[512, 672, 7, 7][672, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_241, [512, 672, 7, 7]);  getitem_241 = None
        div_10: "f32[512, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_4, 49);  expand_4 = None
        add_438: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_767, div_10);  mul_767 = div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        squeeze_174: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
        unsqueeze_572: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
        unsqueeze_573: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
        unsqueeze_574: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
        sum_54: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_438, [0, 2, 3])
        sub_164: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_574);  convolution_66 = unsqueeze_574 = None
        mul_769: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(add_438, sub_164)
        sum_55: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_769, [0, 2, 3]);  mul_769 = None
        mul_770: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_54, 3.985969387755102e-05)
        unsqueeze_575: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
        unsqueeze_576: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
        unsqueeze_577: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
        mul_771: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, 3.985969387755102e-05)
        squeeze_175: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
        mul_772: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
        mul_773: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
        unsqueeze_578: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
        unsqueeze_579: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
        unsqueeze_580: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
        mul_774: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_175, primals_370);  primals_370 = None
        unsqueeze_581: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
        unsqueeze_582: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
        unsqueeze_583: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
        mul_775: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_580);  sub_164 = unsqueeze_580 = None
        sub_166: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sub.Tensor(add_438, mul_775);  add_438 = mul_775 = None
        sub_167: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sub.Tensor(sub_166, unsqueeze_577);  sub_166 = unsqueeze_577 = None
        mul_776: "f32[512, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_583);  sub_167 = unsqueeze_583 = None
        mul_777: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, squeeze_175);  sum_55 = squeeze_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:435 in forward, code: x = self.conv_dw(x)
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_776, cat_22, primals_366, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_776 = cat_22 = primals_366 = None
        getitem_244: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = convolution_backward_28[0]
        getitem_245: "f32[672, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_29: "f32[512, 336, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.slice.Tensor(getitem_244, 1, 0, 336)
        slice_30: "f32[512, 336, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.slice.Tensor(getitem_244, 1, 336, 672);  getitem_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_57: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.sub.Tensor(convolution_65, getitem_115)
        mul_403: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
        unsqueeze_228: "f32[336, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_364, -1)
        unsqueeze_229: "f32[336, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_409: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.mul.Tensor(mul_403, unsqueeze_229);  mul_403 = unsqueeze_229 = None
        unsqueeze_230: "f32[336, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_365, -1);  primals_365 = None
        unsqueeze_231: "f32[336, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_304: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.add.Tensor(mul_409, unsqueeze_231);  mul_409 = unsqueeze_231 = None
        relu_28: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.relu.default(add_304);  add_304 = None
        le_13: "b8[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
        where_16: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.where.self(le_13, full_default, slice_30);  le_13 = slice_30 = None
        squeeze_171: "f32[336][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
        unsqueeze_584: "f32[1, 336][336, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
        unsqueeze_585: "f32[1, 336, 1][336, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
        unsqueeze_586: "f32[1, 336, 1, 1][336, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
        sum_56: "f32[336][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        sub_168: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_586);  convolution_65 = unsqueeze_586 = None
        mul_778: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.mul.Tensor(where_16, sub_168)
        sum_57: "f32[336][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_778, [0, 2, 3]);  mul_778 = None
        mul_779: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_56, 9.964923469387754e-06)
        unsqueeze_587: "f32[1, 336][336, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_779, 0);  mul_779 = None
        unsqueeze_588: "f32[1, 336, 1][336, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
        unsqueeze_589: "f32[1, 336, 1, 1][336, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
        mul_780: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, 9.964923469387754e-06)
        squeeze_172: "f32[336][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
        mul_781: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
        mul_782: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_780, mul_781);  mul_780 = mul_781 = None
        unsqueeze_590: "f32[1, 336][336, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
        unsqueeze_591: "f32[1, 336, 1][336, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
        unsqueeze_592: "f32[1, 336, 1, 1][336, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
        mul_783: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_172, primals_364);  primals_364 = None
        unsqueeze_593: "f32[1, 336][336, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
        unsqueeze_594: "f32[1, 336, 1][336, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
        unsqueeze_595: "f32[1, 336, 1, 1][336, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
        mul_784: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_592);  sub_168 = unsqueeze_592 = None
        sub_170: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.sub.Tensor(where_16, mul_784);  where_16 = mul_784 = None
        sub_171: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_589);  sub_170 = unsqueeze_589 = None
        mul_785: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_595);  sub_171 = unsqueeze_595 = None
        mul_786: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, squeeze_172);  sum_57 = squeeze_172 = None
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_785, relu_27, primals_360, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  mul_785 = primals_360 = None
        getitem_247: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = convolution_backward_29[0]
        getitem_248: "f32[336, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None
        add_439: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.add.Tensor(slice_29, getitem_247);  slice_29 = getitem_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_14: "b8[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        where_17: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.where.self(le_14, full_default, add_439);  le_14 = add_439 = None
        sum_58: "f32[336][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        sub_172: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_598);  convolution_64 = unsqueeze_598 = None
        mul_787: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.mul.Tensor(where_17, sub_172)
        sum_59: "f32[336][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_787, [0, 2, 3]);  mul_787 = None
        mul_788: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, 9.964923469387754e-06)
        unsqueeze_599: "f32[1, 336][336, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_788, 0);  mul_788 = None
        unsqueeze_600: "f32[1, 336, 1][336, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
        unsqueeze_601: "f32[1, 336, 1, 1][336, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
        mul_789: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, 9.964923469387754e-06)
        mul_790: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
        mul_791: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_789, mul_790);  mul_789 = mul_790 = None
        unsqueeze_602: "f32[1, 336][336, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
        unsqueeze_603: "f32[1, 336, 1][336, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
        unsqueeze_604: "f32[1, 336, 1, 1][336, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
        mul_792: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_169, primals_358);  primals_358 = None
        unsqueeze_605: "f32[1, 336][336, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
        unsqueeze_606: "f32[1, 336, 1][336, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
        unsqueeze_607: "f32[1, 336, 1, 1][336, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
        mul_793: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_604);  sub_172 = unsqueeze_604 = None
        sub_174: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.sub.Tensor(where_17, mul_793);  where_17 = mul_793 = None
        sub_175: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_601);  sub_174 = unsqueeze_601 = None
        mul_794: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_607);  sub_175 = unsqueeze_607 = None
        mul_795: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, squeeze_169);  sum_59 = squeeze_169 = None
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_794, add_294, primals_354, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_794 = add_294 = primals_354 = None
        getitem_250: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = convolution_backward_30[0]
        getitem_251: "f32[336, 112, 1, 1][112, 1, 112, 112]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        add_440: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(getitem_229, getitem_250);  getitem_229 = getitem_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        clone_6: "f32[512, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(add_440, memory_format = torch.contiguous_format)
        copy_12: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.copy.default(add_440, clone_6);  add_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_33: "f32[512, 56, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.slice.Tensor(copy_12, 1, 56, 112)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_60: "f32[56][1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_33, [0, 2, 3])
        sub_176: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_610);  convolution_63 = unsqueeze_610 = None
        mul_796: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.mul.Tensor(slice_33, sub_176)
        sum_61: "f32[56][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_796, [0, 2, 3]);  mul_796 = None
        mul_797: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_60, 9.964923469387754e-06)
        unsqueeze_611: "f32[1, 56][56, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
        unsqueeze_612: "f32[1, 56, 1][56, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 2);  unsqueeze_611 = None
        unsqueeze_613: "f32[1, 56, 1, 1][56, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 3);  unsqueeze_612 = None
        mul_798: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, 9.964923469387754e-06)
        mul_799: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
        mul_800: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_798, mul_799);  mul_798 = mul_799 = None
        unsqueeze_614: "f32[1, 56][56, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
        unsqueeze_615: "f32[1, 56, 1][56, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 2);  unsqueeze_614 = None
        unsqueeze_616: "f32[1, 56, 1, 1][56, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 3);  unsqueeze_615 = None
        mul_801: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_166, primals_352);  primals_352 = None
        unsqueeze_617: "f32[1, 56][56, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
        unsqueeze_618: "f32[1, 56, 1][56, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
        unsqueeze_619: "f32[1, 56, 1, 1][56, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
        mul_802: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_616);  sub_176 = unsqueeze_616 = None
        sub_178: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.sub.Tensor(slice_33, mul_802);  slice_33 = mul_802 = None
        sub_179: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.sub.Tensor(sub_178, unsqueeze_613);  sub_178 = unsqueeze_613 = None
        mul_803: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_619);  sub_179 = unsqueeze_619 = None
        mul_804: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, squeeze_166);  sum_61 = squeeze_166 = None
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_803, add_288, primals_348, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 56, [True, True, False]);  mul_803 = add_288 = primals_348 = None
        getitem_253: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = convolution_backward_31[0]
        getitem_254: "f32[56, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_34: "f32[512, 56, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.slice.Tensor(copy_12, 1, 0, 56);  copy_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_441: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.add.Tensor(slice_34, getitem_253);  slice_34 = getitem_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_62: "f32[56][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_441, [0, 2, 3])
        sub_180: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_622);  convolution_62 = unsqueeze_622 = None
        mul_805: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.mul.Tensor(add_441, sub_180)
        sum_63: "f32[56][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_805, [0, 2, 3]);  mul_805 = None
        mul_806: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_62, 9.964923469387754e-06)
        unsqueeze_623: "f32[1, 56][56, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
        unsqueeze_624: "f32[1, 56, 1][56, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
        unsqueeze_625: "f32[1, 56, 1, 1][56, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
        mul_807: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, 9.964923469387754e-06)
        mul_808: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
        mul_809: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_807, mul_808);  mul_807 = mul_808 = None
        unsqueeze_626: "f32[1, 56][56, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
        unsqueeze_627: "f32[1, 56, 1][56, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
        unsqueeze_628: "f32[1, 56, 1, 1][56, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
        mul_810: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_163, primals_346);  primals_346 = None
        unsqueeze_629: "f32[1, 56][56, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
        unsqueeze_630: "f32[1, 56, 1][56, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
        unsqueeze_631: "f32[1, 56, 1, 1][56, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
        mul_811: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_628);  sub_180 = unsqueeze_628 = None
        sub_182: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.sub.Tensor(add_441, mul_811);  add_441 = mul_811 = None
        sub_183: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_625);  sub_182 = unsqueeze_625 = None
        mul_812: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_631);  sub_183 = unsqueeze_631 = None
        mul_813: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, squeeze_163);  sum_63 = squeeze_163 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_812, mul_381, primals_342, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_812 = mul_381 = primals_342 = None
        getitem_256: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = convolution_backward_32[0]
        getitem_257: "f32[56, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_814: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(getitem_256, cat_20);  cat_20 = None
        add_283: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.add.Tensor(convolution_61, 3)
        clamp_min_3: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.clamp_min.default(add_283, 0);  add_283 = None
        clamp_max_3: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
        div_3: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_3, 6);  clamp_max_3 = None
        mul_815: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(getitem_256, div_3);  getitem_256 = div_3 = None
        sum_64: "f32[512, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_814, [2, 3], True);  mul_814 = None
        gt_4: "b8[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.gt.Scalar(convolution_61, -3.0)
        lt_3: "b8[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.lt.Scalar(convolution_61, 3.0);  convolution_61 = None
        bitwise_and_3: "b8[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.bitwise_and.Tensor(gt_4, lt_3);  gt_4 = lt_3 = None
        mul_816: "f32[512, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, 0.16666666666666666);  sum_64 = None
        where_18: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.where.self(bitwise_and_3, mul_816, full_default);  bitwise_and_3 = mul_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_65: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_18, relu_26, primals_340, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_18 = primals_340 = None
        getitem_259: "f32[512, 168, 1, 1][168, 1, 168, 168]cuda:0" = convolution_backward_33[0]
        getitem_260: "f32[672, 168, 1, 1][168, 1, 168, 168]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        le_15: "b8[512, 168, 1, 1][168, 1, 168, 168]cuda:0" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        where_19: "f32[512, 168, 1, 1][168, 1, 168, 168]cuda:0" = torch.ops.aten.where.self(le_15, full_default, getitem_259);  le_15 = getitem_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_66: "f32[168][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(where_19, mean_3, primals_338, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_19 = mean_3 = primals_338 = None
        getitem_262: "f32[512, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_34[0]
        getitem_263: "f32[168, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_5: "f32[512, 672, 14, 14][672, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_262, [512, 672, 14, 14]);  getitem_262 = None
        div_11: "f32[512, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_5, 196);  expand_5 = None
        add_442: "f32[512, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_815, div_11);  mul_815 = div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_35: "f32[512, 336, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.slice.Tensor(add_442, 1, 0, 336)
        slice_36: "f32[512, 336, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.slice.Tensor(add_442, 1, 336, 672);  add_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_53: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.sub.Tensor(convolution_59, getitem_107)
        mul_374: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
        unsqueeze_212: "f32[336, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_336, -1)
        unsqueeze_213: "f32[336, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_380: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.mul.Tensor(mul_374, unsqueeze_213);  mul_374 = unsqueeze_213 = None
        unsqueeze_214: "f32[336, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_337, -1);  primals_337 = None
        unsqueeze_215: "f32[336, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_282: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.add.Tensor(mul_380, unsqueeze_215);  mul_380 = unsqueeze_215 = None
        relu_25: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.relu.default(add_282);  add_282 = None
        le_16: "b8[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        where_20: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.where.self(le_16, full_default, slice_36);  le_16 = slice_36 = None
        squeeze_159: "f32[336][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        unsqueeze_632: "f32[1, 336][336, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
        unsqueeze_633: "f32[1, 336, 1][336, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
        unsqueeze_634: "f32[1, 336, 1, 1][336, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
        sum_67: "f32[336][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
        sub_184: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_634);  convolution_59 = unsqueeze_634 = None
        mul_817: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.mul.Tensor(where_20, sub_184)
        sum_68: "f32[336][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_817, [0, 2, 3]);  mul_817 = None
        mul_818: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, 9.964923469387754e-06)
        unsqueeze_635: "f32[1, 336][336, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
        unsqueeze_636: "f32[1, 336, 1][336, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
        unsqueeze_637: "f32[1, 336, 1, 1][336, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
        mul_819: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_68, 9.964923469387754e-06)
        squeeze_160: "f32[336][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
        mul_820: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
        mul_821: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_819, mul_820);  mul_819 = mul_820 = None
        unsqueeze_638: "f32[1, 336][336, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
        unsqueeze_639: "f32[1, 336, 1][336, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
        unsqueeze_640: "f32[1, 336, 1, 1][336, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
        mul_822: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_160, primals_336);  primals_336 = None
        unsqueeze_641: "f32[1, 336][336, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_822, 0);  mul_822 = None
        unsqueeze_642: "f32[1, 336, 1][336, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
        unsqueeze_643: "f32[1, 336, 1, 1][336, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
        mul_823: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_640);  sub_184 = unsqueeze_640 = None
        sub_186: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.sub.Tensor(where_20, mul_823);  where_20 = mul_823 = None
        sub_187: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_637);  sub_186 = unsqueeze_637 = None
        mul_824: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_643);  sub_187 = unsqueeze_643 = None
        mul_825: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_68, squeeze_160);  sum_68 = squeeze_160 = None
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_824, relu_24, primals_332, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 336, [True, True, False]);  mul_824 = primals_332 = None
        getitem_265: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = convolution_backward_35[0]
        getitem_266: "f32[336, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None
        add_443: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.add.Tensor(slice_35, getitem_265);  slice_35 = getitem_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_17: "b8[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        where_21: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.where.self(le_17, full_default, add_443);  le_17 = add_443 = None
        sum_69: "f32[336][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
        sub_188: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_646);  convolution_58 = unsqueeze_646 = None
        mul_826: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.mul.Tensor(where_21, sub_188)
        sum_70: "f32[336][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_826, [0, 2, 3]);  mul_826 = None
        mul_827: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, 9.964923469387754e-06)
        unsqueeze_647: "f32[1, 336][336, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
        unsqueeze_648: "f32[1, 336, 1][336, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
        unsqueeze_649: "f32[1, 336, 1, 1][336, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
        mul_828: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_70, 9.964923469387754e-06)
        mul_829: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
        mul_830: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_828, mul_829);  mul_828 = mul_829 = None
        unsqueeze_650: "f32[1, 336][336, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_830, 0);  mul_830 = None
        unsqueeze_651: "f32[1, 336, 1][336, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
        unsqueeze_652: "f32[1, 336, 1, 1][336, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
        mul_831: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_157, primals_330);  primals_330 = None
        unsqueeze_653: "f32[1, 336][336, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_831, 0);  mul_831 = None
        unsqueeze_654: "f32[1, 336, 1][336, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
        unsqueeze_655: "f32[1, 336, 1, 1][336, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
        mul_832: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_652);  sub_188 = unsqueeze_652 = None
        sub_190: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.sub.Tensor(where_21, mul_832);  where_21 = mul_832 = None
        sub_191: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_649);  sub_190 = unsqueeze_649 = None
        mul_833: "f32[512, 336, 14, 14][65856, 1, 4704, 336]cuda:0" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_655);  sub_191 = unsqueeze_655 = None
        mul_834: "f32[336][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_70, squeeze_157);  sum_70 = squeeze_157 = None
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_833, add_272, primals_326, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_833 = add_272 = primals_326 = None
        getitem_268: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = convolution_backward_36[0]
        getitem_269: "f32[336, 112, 1, 1][112, 1, 112, 112]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None
        add_444: "f32[512, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(clone_6, getitem_268);  clone_6 = getitem_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        new_empty_strided_6: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.new_empty_strided.default(add_444, [512, 112, 14, 14], [21952, 1, 1568, 112])
        copy_13: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.copy.default(new_empty_strided_6, add_444);  new_empty_strided_6 = add_444 = None
        clone_7: "f32[512, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(copy_13, memory_format = torch.contiguous_format)
        copy_14: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.copy.default(copy_13, clone_7);  copy_13 = None
        sum_71: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(clone_7, [0, 2, 3])
        sub_192: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_658);  convolution_57 = unsqueeze_658 = None
        mul_835: "f32[512, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(clone_7, sub_192)
        sum_72: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_835, [0, 2, 3]);  mul_835 = None
        mul_836: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, 9.964923469387754e-06)
        unsqueeze_659: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_836, 0);  mul_836 = None
        unsqueeze_660: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 2);  unsqueeze_659 = None
        unsqueeze_661: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 3);  unsqueeze_660 = None
        mul_837: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_72, 9.964923469387754e-06)
        mul_838: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_839: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_837, mul_838);  mul_837 = mul_838 = None
        unsqueeze_662: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_839, 0);  mul_839 = None
        unsqueeze_663: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 2);  unsqueeze_662 = None
        unsqueeze_664: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 3);  unsqueeze_663 = None
        mul_840: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, primals_324);  primals_324 = None
        unsqueeze_665: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_840, 0);  mul_840 = None
        unsqueeze_666: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
        unsqueeze_667: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
        mul_841: "f32[512, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_664);  sub_192 = unsqueeze_664 = None
        sub_194: "f32[512, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_7, mul_841);  clone_7 = mul_841 = None
        sub_195: "f32[512, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_661);  sub_194 = unsqueeze_661 = None
        mul_842: "f32[512, 112, 14, 14][21952, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_667);  sub_195 = unsqueeze_667 = None
        mul_843: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_72, squeeze_154);  sum_72 = squeeze_154 = None
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_842, add_266, primals_320, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_842 = add_266 = primals_320 = None
        getitem_271: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = convolution_backward_37[0]
        getitem_272: "f32[112, 80, 1, 1][80, 1, 80, 80]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None
        sum_73: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_271, [0, 2, 3])
        sub_196: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_670);  convolution_56 = unsqueeze_670 = None
        mul_844: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(getitem_271, sub_196)
        sum_74: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_844, [0, 2, 3]);  mul_844 = None
        mul_845: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, 9.964923469387754e-06)
        unsqueeze_671: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_845, 0);  mul_845 = None
        unsqueeze_672: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 2);  unsqueeze_671 = None
        unsqueeze_673: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 3);  unsqueeze_672 = None
        mul_846: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_74, 9.964923469387754e-06)
        mul_847: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_848: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_846, mul_847);  mul_846 = mul_847 = None
        unsqueeze_674: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_848, 0);  mul_848 = None
        unsqueeze_675: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 2);  unsqueeze_674 = None
        unsqueeze_676: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 3);  unsqueeze_675 = None
        mul_849: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, primals_318);  primals_318 = None
        unsqueeze_677: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_849, 0);  mul_849 = None
        unsqueeze_678: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
        unsqueeze_679: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
        mul_850: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_676);  sub_196 = unsqueeze_676 = None
        sub_198: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(getitem_271, mul_850);  getitem_271 = mul_850 = None
        sub_199: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_673);  sub_198 = unsqueeze_673 = None
        mul_851: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_679);  sub_199 = unsqueeze_679 = None
        mul_852: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_74, squeeze_151);  sum_74 = squeeze_151 = None
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_851, add_240, primals_314, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, False]);  mul_851 = primals_314 = None
        getitem_274: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = convolution_backward_38[0]
        getitem_275: "f32[80, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_39: "f32[512, 56, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.slice.Tensor(copy_14, 1, 56, 112)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_75: "f32[56][1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_39, [0, 2, 3])
        sub_200: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_682);  convolution_55 = unsqueeze_682 = None
        mul_853: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.mul.Tensor(slice_39, sub_200)
        sum_76: "f32[56][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_853, [0, 2, 3]);  mul_853 = None
        mul_854: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, 9.964923469387754e-06)
        unsqueeze_683: "f32[1, 56][56, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_854, 0);  mul_854 = None
        unsqueeze_684: "f32[1, 56, 1][56, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 2);  unsqueeze_683 = None
        unsqueeze_685: "f32[1, 56, 1, 1][56, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 3);  unsqueeze_684 = None
        mul_855: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, 9.964923469387754e-06)
        mul_856: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_857: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_855, mul_856);  mul_855 = mul_856 = None
        unsqueeze_686: "f32[1, 56][56, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_857, 0);  mul_857 = None
        unsqueeze_687: "f32[1, 56, 1][56, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 2);  unsqueeze_686 = None
        unsqueeze_688: "f32[1, 56, 1, 1][56, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 3);  unsqueeze_687 = None
        mul_858: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, primals_312);  primals_312 = None
        unsqueeze_689: "f32[1, 56][56, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_858, 0);  mul_858 = None
        unsqueeze_690: "f32[1, 56, 1][56, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
        unsqueeze_691: "f32[1, 56, 1, 1][56, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
        mul_859: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_688);  sub_200 = unsqueeze_688 = None
        sub_202: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.sub.Tensor(slice_39, mul_859);  slice_39 = mul_859 = None
        sub_203: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.sub.Tensor(sub_202, unsqueeze_685);  sub_202 = unsqueeze_685 = None
        mul_860: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_691);  sub_203 = unsqueeze_691 = None
        mul_861: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, squeeze_148);  sum_76 = squeeze_148 = None
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_860, add_256, primals_308, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 56, [True, True, False]);  mul_860 = add_256 = primals_308 = None
        getitem_277: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = convolution_backward_39[0]
        getitem_278: "f32[56, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_40: "f32[512, 56, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.slice.Tensor(copy_14, 1, 0, 56);  copy_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_445: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.add.Tensor(slice_40, getitem_277);  slice_40 = getitem_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_77: "f32[56][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_445, [0, 2, 3])
        sub_204: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_694);  convolution_54 = unsqueeze_694 = None
        mul_862: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.mul.Tensor(add_445, sub_204)
        sum_78: "f32[56][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_862, [0, 2, 3]);  mul_862 = None
        mul_863: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, 9.964923469387754e-06)
        unsqueeze_695: "f32[1, 56][56, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_863, 0);  mul_863 = None
        unsqueeze_696: "f32[1, 56, 1][56, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 2);  unsqueeze_695 = None
        unsqueeze_697: "f32[1, 56, 1, 1][56, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 3);  unsqueeze_696 = None
        mul_864: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_78, 9.964923469387754e-06)
        mul_865: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_866: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_864, mul_865);  mul_864 = mul_865 = None
        unsqueeze_698: "f32[1, 56][56, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_866, 0);  mul_866 = None
        unsqueeze_699: "f32[1, 56, 1][56, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 2);  unsqueeze_698 = None
        unsqueeze_700: "f32[1, 56, 1, 1][56, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 3);  unsqueeze_699 = None
        mul_867: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, primals_306);  primals_306 = None
        unsqueeze_701: "f32[1, 56][56, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_867, 0);  mul_867 = None
        unsqueeze_702: "f32[1, 56, 1][56, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
        unsqueeze_703: "f32[1, 56, 1, 1][56, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
        mul_868: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_700);  sub_204 = unsqueeze_700 = None
        sub_206: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.sub.Tensor(add_445, mul_868);  add_445 = mul_868 = None
        sub_207: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_697);  sub_206 = unsqueeze_697 = None
        mul_869: "f32[512, 56, 14, 14][10976, 1, 784, 56]cuda:0" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_703);  sub_207 = unsqueeze_703 = None
        mul_870: "f32[56][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_78, squeeze_145);  sum_78 = squeeze_145 = None
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_869, mul_338, primals_302, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_869 = mul_338 = primals_302 = None
        getitem_280: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = convolution_backward_40[0]
        getitem_281: "f32[56, 480, 1, 1][480, 1, 480, 480]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_871: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(getitem_280, cat_18);  cat_18 = None
        add_251: "f32[512, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.add.Tensor(convolution_53, 3)
        clamp_min_2: "f32[512, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.clamp_min.default(add_251, 0);  add_251 = None
        clamp_max_2: "f32[512, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
        div_2: "f32[512, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_2, 6);  clamp_max_2 = None
        mul_872: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(getitem_280, div_2);  getitem_280 = div_2 = None
        sum_79: "f32[512, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_871, [2, 3], True);  mul_871 = None
        gt_5: "b8[512, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.gt.Scalar(convolution_53, -3.0)
        lt_4: "b8[512, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.lt.Scalar(convolution_53, 3.0);  convolution_53 = None
        bitwise_and_4: "b8[512, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.bitwise_and.Tensor(gt_5, lt_4);  gt_5 = lt_4 = None
        mul_873: "f32[512, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, 0.16666666666666666);  sum_79 = None
        where_22: "f32[512, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.where.self(bitwise_and_4, mul_873, full_default);  bitwise_and_4 = mul_873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_80: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(where_22, relu_23, primals_300, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_22 = primals_300 = None
        getitem_283: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = convolution_backward_41[0]
        getitem_284: "f32[480, 120, 1, 1][120, 1, 120, 120]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        le_18: "b8[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        where_23: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.where.self(le_18, full_default, getitem_283);  le_18 = getitem_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_81: "f32[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(where_23, mean_2, primals_298, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_23 = mean_2 = primals_298 = None
        getitem_286: "f32[512, 480, 1, 1][480, 1, 480, 480]cuda:0" = convolution_backward_42[0]
        getitem_287: "f32[120, 480, 1, 1][480, 1, 480, 480]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_6: "f32[512, 480, 14, 14][480, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_286, [512, 480, 14, 14]);  getitem_286 = None
        div_12: "f32[512, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_6, 196);  expand_6 = None
        add_446: "f32[512, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_872, div_12);  mul_872 = div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_41: "f32[512, 240, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.slice.Tensor(add_446, 1, 0, 240)
        slice_42: "f32[512, 240, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.slice.Tensor(add_446, 1, 240, 480);  add_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_47: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_51, getitem_95)
        mul_331: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        unsqueeze_188: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_296, -1)
        unsqueeze_189: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_337: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_189);  mul_331 = unsqueeze_189 = None
        unsqueeze_190: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_297, -1);  primals_297 = None
        unsqueeze_191: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_250: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_337, unsqueeze_191);  mul_337 = unsqueeze_191 = None
        relu_22: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.relu.default(add_250);  add_250 = None
        le_19: "b8[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_24: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.where.self(le_19, full_default, slice_42);  le_19 = slice_42 = None
        squeeze_141: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        unsqueeze_704: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
        unsqueeze_705: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
        unsqueeze_706: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None
        sum_82: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
        sub_208: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_706);  convolution_51 = unsqueeze_706 = None
        mul_874: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(where_24, sub_208)
        sum_83: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_874, [0, 2, 3]);  mul_874 = None
        mul_875: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, 9.964923469387754e-06)
        unsqueeze_707: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
        unsqueeze_708: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 2);  unsqueeze_707 = None
        unsqueeze_709: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 3);  unsqueeze_708 = None
        mul_876: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, 9.964923469387754e-06)
        squeeze_142: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_877: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_878: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_876, mul_877);  mul_876 = mul_877 = None
        unsqueeze_710: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_878, 0);  mul_878 = None
        unsqueeze_711: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 2);  unsqueeze_710 = None
        unsqueeze_712: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 3);  unsqueeze_711 = None
        mul_879: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, primals_296);  primals_296 = None
        unsqueeze_713: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_879, 0);  mul_879 = None
        unsqueeze_714: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
        unsqueeze_715: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
        mul_880: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_712);  sub_208 = unsqueeze_712 = None
        sub_210: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(where_24, mul_880);  where_24 = mul_880 = None
        sub_211: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_709);  sub_210 = unsqueeze_709 = None
        mul_881: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_715);  sub_211 = unsqueeze_715 = None
        mul_882: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, squeeze_142);  sum_83 = squeeze_142 = None
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_881, relu_21, primals_292, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_881 = primals_292 = None
        getitem_289: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = convolution_backward_43[0]
        getitem_290: "f32[240, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None
        add_447: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.add.Tensor(slice_41, getitem_289);  slice_41 = getitem_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_20: "b8[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        where_25: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.where.self(le_20, full_default, add_447);  le_20 = add_447 = None
        sum_84: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
        sub_212: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_718);  convolution_50 = unsqueeze_718 = None
        mul_883: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(where_25, sub_212)
        sum_85: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_883, [0, 2, 3]);  mul_883 = None
        mul_884: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_84, 9.964923469387754e-06)
        unsqueeze_719: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
        unsqueeze_720: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 2);  unsqueeze_719 = None
        unsqueeze_721: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 3);  unsqueeze_720 = None
        mul_885: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, 9.964923469387754e-06)
        mul_886: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_887: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_885, mul_886);  mul_885 = mul_886 = None
        unsqueeze_722: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_887, 0);  mul_887 = None
        unsqueeze_723: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 2);  unsqueeze_722 = None
        unsqueeze_724: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 3);  unsqueeze_723 = None
        mul_888: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, primals_290);  primals_290 = None
        unsqueeze_725: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_888, 0);  mul_888 = None
        unsqueeze_726: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
        unsqueeze_727: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
        mul_889: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_724);  sub_212 = unsqueeze_724 = None
        sub_214: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(where_25, mul_889);  where_25 = mul_889 = None
        sub_215: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(sub_214, unsqueeze_721);  sub_214 = unsqueeze_721 = None
        mul_890: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_727);  sub_215 = unsqueeze_727 = None
        mul_891: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, squeeze_139);  sum_85 = squeeze_139 = None
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_890, add_240, primals_286, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_890 = add_240 = primals_286 = None
        getitem_292: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = convolution_backward_44[0]
        getitem_293: "f32[240, 80, 1, 1][80, 1, 80, 80]cuda:0" = convolution_backward_44[1];  convolution_backward_44 = None
        add_448: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(getitem_274, getitem_292);  getitem_274 = getitem_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        clone_8: "f32[512, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(add_448, memory_format = torch.contiguous_format)
        copy_16: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.copy.default(add_448, clone_8);  add_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_45: "f32[512, 40, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.slice.Tensor(copy_16, 1, 40, 80)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_86: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_45, [0, 2, 3])
        sub_216: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_730);  convolution_49 = unsqueeze_730 = None
        mul_892: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(slice_45, sub_216)
        sum_87: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_892, [0, 2, 3]);  mul_892 = None
        mul_893: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_86, 9.964923469387754e-06)
        unsqueeze_731: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_893, 0);  mul_893 = None
        unsqueeze_732: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 2);  unsqueeze_731 = None
        unsqueeze_733: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 3);  unsqueeze_732 = None
        mul_894: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, 9.964923469387754e-06)
        mul_895: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_896: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_894, mul_895);  mul_894 = mul_895 = None
        unsqueeze_734: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_896, 0);  mul_896 = None
        unsqueeze_735: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 2);  unsqueeze_734 = None
        unsqueeze_736: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 3);  unsqueeze_735 = None
        mul_897: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, primals_284);  primals_284 = None
        unsqueeze_737: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_897, 0);  mul_897 = None
        unsqueeze_738: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
        unsqueeze_739: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
        mul_898: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_736);  sub_216 = unsqueeze_736 = None
        sub_218: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(slice_45, mul_898);  slice_45 = mul_898 = None
        sub_219: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_733);  sub_218 = unsqueeze_733 = None
        mul_899: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_739);  sub_219 = unsqueeze_739 = None
        mul_900: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, squeeze_136);  sum_87 = squeeze_136 = None
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_899, add_234, primals_280, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 40, [True, True, False]);  mul_899 = add_234 = primals_280 = None
        getitem_295: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = convolution_backward_45[0]
        getitem_296: "f32[40, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_45[1];  convolution_backward_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_46: "f32[512, 40, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.slice.Tensor(copy_16, 1, 0, 40);  copy_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_449: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.add.Tensor(slice_46, getitem_295);  slice_46 = getitem_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_88: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_449, [0, 2, 3])
        sub_220: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_742);  convolution_48 = unsqueeze_742 = None
        mul_901: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(add_449, sub_220)
        sum_89: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_901, [0, 2, 3]);  mul_901 = None
        mul_902: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_88, 9.964923469387754e-06)
        unsqueeze_743: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_902, 0);  mul_902 = None
        unsqueeze_744: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 2);  unsqueeze_743 = None
        unsqueeze_745: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 3);  unsqueeze_744 = None
        mul_903: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, 9.964923469387754e-06)
        mul_904: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_905: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_903, mul_904);  mul_903 = mul_904 = None
        unsqueeze_746: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_905, 0);  mul_905 = None
        unsqueeze_747: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 2);  unsqueeze_746 = None
        unsqueeze_748: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 3);  unsqueeze_747 = None
        mul_906: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, primals_278);  primals_278 = None
        unsqueeze_749: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_906, 0);  mul_906 = None
        unsqueeze_750: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
        unsqueeze_751: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
        mul_907: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_748);  sub_220 = unsqueeze_748 = None
        sub_222: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(add_449, mul_907);  add_449 = mul_907 = None
        sub_223: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(sub_222, unsqueeze_745);  sub_222 = unsqueeze_745 = None
        mul_908: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_751);  sub_223 = unsqueeze_751 = None
        mul_909: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, squeeze_133);  sum_89 = squeeze_133 = None
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_908, cat_16, primals_274, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_908 = cat_16 = primals_274 = None
        getitem_298: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = convolution_backward_46[0]
        getitem_299: "f32[40, 184, 1, 1][184, 1, 184, 184]cuda:0" = convolution_backward_46[1];  convolution_backward_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_47: "f32[512, 92, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.slice.Tensor(getitem_298, 1, 0, 92)
        slice_48: "f32[512, 92, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.slice.Tensor(getitem_298, 1, 92, 184);  getitem_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_43: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.sub.Tensor(convolution_47, getitem_87)
        mul_303: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        unsqueeze_172: "f32[92, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_272, -1)
        unsqueeze_173: "f32[92, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_309: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.mul.Tensor(mul_303, unsqueeze_173);  mul_303 = unsqueeze_173 = None
        unsqueeze_174: "f32[92, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_273, -1);  primals_273 = None
        unsqueeze_175: "f32[92, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_229: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.add.Tensor(mul_309, unsqueeze_175);  mul_309 = unsqueeze_175 = None
        relu_20: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.relu.default(add_229);  add_229 = None
        le_21: "b8[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_26: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.where.self(le_21, full_default, slice_48);  le_21 = slice_48 = None
        squeeze_129: "f32[92][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        unsqueeze_752: "f32[1, 92][92, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
        unsqueeze_753: "f32[1, 92, 1][92, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
        unsqueeze_754: "f32[1, 92, 1, 1][92, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None
        sum_90: "f32[92][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
        sub_224: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_754);  convolution_47 = unsqueeze_754 = None
        mul_910: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.mul.Tensor(where_26, sub_224)
        sum_91: "f32[92][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_910, [0, 2, 3]);  mul_910 = None
        mul_911: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_90, 9.964923469387754e-06)
        unsqueeze_755: "f32[1, 92][92, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_911, 0);  mul_911 = None
        unsqueeze_756: "f32[1, 92, 1][92, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 2);  unsqueeze_755 = None
        unsqueeze_757: "f32[1, 92, 1, 1][92, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 3);  unsqueeze_756 = None
        mul_912: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, 9.964923469387754e-06)
        squeeze_130: "f32[92][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_913: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_914: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_912, mul_913);  mul_912 = mul_913 = None
        unsqueeze_758: "f32[1, 92][92, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_914, 0);  mul_914 = None
        unsqueeze_759: "f32[1, 92, 1][92, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 2);  unsqueeze_758 = None
        unsqueeze_760: "f32[1, 92, 1, 1][92, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 3);  unsqueeze_759 = None
        mul_915: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, primals_272);  primals_272 = None
        unsqueeze_761: "f32[1, 92][92, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_915, 0);  mul_915 = None
        unsqueeze_762: "f32[1, 92, 1][92, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
        unsqueeze_763: "f32[1, 92, 1, 1][92, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
        mul_916: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_760);  sub_224 = unsqueeze_760 = None
        sub_226: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.sub.Tensor(where_26, mul_916);  where_26 = mul_916 = None
        sub_227: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_757);  sub_226 = unsqueeze_757 = None
        mul_917: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_763);  sub_227 = unsqueeze_763 = None
        mul_918: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, squeeze_130);  sum_91 = squeeze_130 = None
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_917, relu_19, primals_268, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 92, [True, True, False]);  mul_917 = primals_268 = None
        getitem_301: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = convolution_backward_47[0]
        getitem_302: "f32[92, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_47[1];  convolution_backward_47 = None
        add_450: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.add.Tensor(slice_47, getitem_301);  slice_47 = getitem_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_22: "b8[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_27: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.where.self(le_22, full_default, add_450);  le_22 = add_450 = None
        sum_92: "f32[92][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
        sub_228: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_766);  convolution_46 = unsqueeze_766 = None
        mul_919: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.mul.Tensor(where_27, sub_228)
        sum_93: "f32[92][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_919, [0, 2, 3]);  mul_919 = None
        mul_920: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_92, 9.964923469387754e-06)
        unsqueeze_767: "f32[1, 92][92, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_920, 0);  mul_920 = None
        unsqueeze_768: "f32[1, 92, 1][92, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 2);  unsqueeze_767 = None
        unsqueeze_769: "f32[1, 92, 1, 1][92, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 3);  unsqueeze_768 = None
        mul_921: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, 9.964923469387754e-06)
        mul_922: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_923: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_921, mul_922);  mul_921 = mul_922 = None
        unsqueeze_770: "f32[1, 92][92, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_923, 0);  mul_923 = None
        unsqueeze_771: "f32[1, 92, 1][92, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 2);  unsqueeze_770 = None
        unsqueeze_772: "f32[1, 92, 1, 1][92, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 3);  unsqueeze_771 = None
        mul_924: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, primals_266);  primals_266 = None
        unsqueeze_773: "f32[1, 92][92, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_924, 0);  mul_924 = None
        unsqueeze_774: "f32[1, 92, 1][92, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
        unsqueeze_775: "f32[1, 92, 1, 1][92, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
        mul_925: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_772);  sub_228 = unsqueeze_772 = None
        sub_230: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.sub.Tensor(where_27, mul_925);  where_27 = mul_925 = None
        sub_231: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.sub.Tensor(sub_230, unsqueeze_769);  sub_230 = unsqueeze_769 = None
        mul_926: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_775);  sub_231 = unsqueeze_775 = None
        mul_927: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, squeeze_127);  sum_93 = squeeze_127 = None
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_926, add_219, primals_262, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_926 = add_219 = primals_262 = None
        getitem_304: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = convolution_backward_48[0]
        getitem_305: "f32[92, 80, 1, 1][80, 1, 80, 80]cuda:0" = convolution_backward_48[1];  convolution_backward_48 = None
        add_451: "f32[512, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(clone_8, getitem_304);  clone_8 = getitem_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        new_empty_strided_8: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.new_empty_strided.default(add_451, [512, 80, 14, 14], [15680, 1, 1120, 80])
        copy_17: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.copy.default(new_empty_strided_8, add_451);  new_empty_strided_8 = add_451 = None
        clone_9: "f32[512, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(copy_17, memory_format = torch.contiguous_format)
        copy_18: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.copy.default(copy_17, clone_9);  copy_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_51: "f32[512, 40, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.slice.Tensor(copy_18, 1, 40, 80)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_94: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_51, [0, 2, 3])
        sub_232: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_778);  convolution_45 = unsqueeze_778 = None
        mul_928: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(slice_51, sub_232)
        sum_95: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_928, [0, 2, 3]);  mul_928 = None
        mul_929: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_94, 9.964923469387754e-06)
        unsqueeze_779: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_929, 0);  mul_929 = None
        unsqueeze_780: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 2);  unsqueeze_779 = None
        unsqueeze_781: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 3);  unsqueeze_780 = None
        mul_930: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, 9.964923469387754e-06)
        mul_931: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_932: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_930, mul_931);  mul_930 = mul_931 = None
        unsqueeze_782: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
        unsqueeze_783: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 2);  unsqueeze_782 = None
        unsqueeze_784: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 3);  unsqueeze_783 = None
        mul_933: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, primals_260);  primals_260 = None
        unsqueeze_785: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_933, 0);  mul_933 = None
        unsqueeze_786: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
        unsqueeze_787: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
        mul_934: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_784);  sub_232 = unsqueeze_784 = None
        sub_234: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(slice_51, mul_934);  slice_51 = mul_934 = None
        sub_235: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(sub_234, unsqueeze_781);  sub_234 = unsqueeze_781 = None
        mul_935: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_787);  sub_235 = unsqueeze_787 = None
        mul_936: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, squeeze_124);  sum_95 = squeeze_124 = None
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_935, add_213, primals_256, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 40, [True, True, False]);  mul_935 = add_213 = primals_256 = None
        getitem_307: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = convolution_backward_49[0]
        getitem_308: "f32[40, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_49[1];  convolution_backward_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_52: "f32[512, 40, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.slice.Tensor(copy_18, 1, 0, 40);  copy_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_452: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.add.Tensor(slice_52, getitem_307);  slice_52 = getitem_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_96: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_452, [0, 2, 3])
        sub_236: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_790);  convolution_44 = unsqueeze_790 = None
        mul_937: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(add_452, sub_236)
        sum_97: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_937, [0, 2, 3]);  mul_937 = None
        mul_938: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_96, 9.964923469387754e-06)
        unsqueeze_791: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_938, 0);  mul_938 = None
        unsqueeze_792: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 2);  unsqueeze_791 = None
        unsqueeze_793: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 3);  unsqueeze_792 = None
        mul_939: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, 9.964923469387754e-06)
        mul_940: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_941: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_939, mul_940);  mul_939 = mul_940 = None
        unsqueeze_794: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_941, 0);  mul_941 = None
        unsqueeze_795: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 2);  unsqueeze_794 = None
        unsqueeze_796: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 3);  unsqueeze_795 = None
        mul_942: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, primals_254);  primals_254 = None
        unsqueeze_797: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_942, 0);  mul_942 = None
        unsqueeze_798: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
        unsqueeze_799: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
        mul_943: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_796);  sub_236 = unsqueeze_796 = None
        sub_238: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(add_452, mul_943);  add_452 = mul_943 = None
        sub_239: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_793);  sub_238 = unsqueeze_793 = None
        mul_944: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_799);  sub_239 = unsqueeze_799 = None
        mul_945: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, squeeze_121);  sum_97 = squeeze_121 = None
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_944, cat_14, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_944 = cat_14 = primals_250 = None
        getitem_310: "f32[512, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = convolution_backward_50[0]
        getitem_311: "f32[40, 184, 1, 1][184, 1, 184, 184]cuda:0" = convolution_backward_50[1];  convolution_backward_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_53: "f32[512, 92, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.slice.Tensor(getitem_310, 1, 0, 92)
        slice_54: "f32[512, 92, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.slice.Tensor(getitem_310, 1, 92, 184);  getitem_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_39: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.sub.Tensor(convolution_43, getitem_79)
        mul_275: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        unsqueeze_156: "f32[92, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_248, -1)
        unsqueeze_157: "f32[92, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_281: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.mul.Tensor(mul_275, unsqueeze_157);  mul_275 = unsqueeze_157 = None
        unsqueeze_158: "f32[92, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_249, -1);  primals_249 = None
        unsqueeze_159: "f32[92, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_208: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.add.Tensor(mul_281, unsqueeze_159);  mul_281 = unsqueeze_159 = None
        relu_18: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.relu.default(add_208);  add_208 = None
        le_23: "b8[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        where_28: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.where.self(le_23, full_default, slice_54);  le_23 = slice_54 = None
        squeeze_117: "f32[92][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        unsqueeze_800: "f32[1, 92][92, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
        unsqueeze_801: "f32[1, 92, 1][92, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
        unsqueeze_802: "f32[1, 92, 1, 1][92, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
        sum_98: "f32[92][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
        sub_240: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_802);  convolution_43 = unsqueeze_802 = None
        mul_946: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.mul.Tensor(where_28, sub_240)
        sum_99: "f32[92][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_946, [0, 2, 3]);  mul_946 = None
        mul_947: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_98, 9.964923469387754e-06)
        unsqueeze_803: "f32[1, 92][92, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_947, 0);  mul_947 = None
        unsqueeze_804: "f32[1, 92, 1][92, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 2);  unsqueeze_803 = None
        unsqueeze_805: "f32[1, 92, 1, 1][92, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 3);  unsqueeze_804 = None
        mul_948: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, 9.964923469387754e-06)
        squeeze_118: "f32[92][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_949: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_950: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_948, mul_949);  mul_948 = mul_949 = None
        unsqueeze_806: "f32[1, 92][92, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_950, 0);  mul_950 = None
        unsqueeze_807: "f32[1, 92, 1][92, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 2);  unsqueeze_806 = None
        unsqueeze_808: "f32[1, 92, 1, 1][92, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 3);  unsqueeze_807 = None
        mul_951: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, primals_248);  primals_248 = None
        unsqueeze_809: "f32[1, 92][92, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_951, 0);  mul_951 = None
        unsqueeze_810: "f32[1, 92, 1][92, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
        unsqueeze_811: "f32[1, 92, 1, 1][92, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
        mul_952: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_808);  sub_240 = unsqueeze_808 = None
        sub_242: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.sub.Tensor(where_28, mul_952);  where_28 = mul_952 = None
        sub_243: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_805);  sub_242 = unsqueeze_805 = None
        mul_953: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_811);  sub_243 = unsqueeze_811 = None
        mul_954: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, squeeze_118);  sum_99 = squeeze_118 = None
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_953, relu_17, primals_244, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 92, [True, True, False]);  mul_953 = primals_244 = None
        getitem_313: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = convolution_backward_51[0]
        getitem_314: "f32[92, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_51[1];  convolution_backward_51 = None
        add_453: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.add.Tensor(slice_53, getitem_313);  slice_53 = getitem_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_24: "b8[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_29: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.where.self(le_24, full_default, add_453);  le_24 = add_453 = None
        sum_100: "f32[92][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
        sub_244: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_814);  convolution_42 = unsqueeze_814 = None
        mul_955: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.mul.Tensor(where_29, sub_244)
        sum_101: "f32[92][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_955, [0, 2, 3]);  mul_955 = None
        mul_956: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_100, 9.964923469387754e-06)
        unsqueeze_815: "f32[1, 92][92, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_956, 0);  mul_956 = None
        unsqueeze_816: "f32[1, 92, 1][92, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 2);  unsqueeze_815 = None
        unsqueeze_817: "f32[1, 92, 1, 1][92, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 3);  unsqueeze_816 = None
        mul_957: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, 9.964923469387754e-06)
        mul_958: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_959: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_957, mul_958);  mul_957 = mul_958 = None
        unsqueeze_818: "f32[1, 92][92, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_959, 0);  mul_959 = None
        unsqueeze_819: "f32[1, 92, 1][92, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 2);  unsqueeze_818 = None
        unsqueeze_820: "f32[1, 92, 1, 1][92, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 3);  unsqueeze_819 = None
        mul_960: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, primals_242);  primals_242 = None
        unsqueeze_821: "f32[1, 92][92, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_960, 0);  mul_960 = None
        unsqueeze_822: "f32[1, 92, 1][92, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
        unsqueeze_823: "f32[1, 92, 1, 1][92, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
        mul_961: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_820);  sub_244 = unsqueeze_820 = None
        sub_246: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.sub.Tensor(where_29, mul_961);  where_29 = mul_961 = None
        sub_247: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_817);  sub_246 = unsqueeze_817 = None
        mul_962: "f32[512, 92, 14, 14][18032, 1, 1288, 92]cuda:0" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_823);  sub_247 = unsqueeze_823 = None
        mul_963: "f32[92][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, squeeze_115);  sum_101 = squeeze_115 = None
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_962, add_198, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_962 = add_198 = primals_238 = None
        getitem_316: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = convolution_backward_52[0]
        getitem_317: "f32[92, 80, 1, 1][80, 1, 80, 80]cuda:0" = convolution_backward_52[1];  convolution_backward_52 = None
        add_454: "f32[512, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(clone_9, getitem_316);  clone_9 = getitem_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        new_empty_strided_9: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.new_empty_strided.default(add_454, [512, 80, 14, 14], [15680, 1, 1120, 80])
        copy_19: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.copy.default(new_empty_strided_9, add_454);  new_empty_strided_9 = add_454 = None
        clone_10: "f32[512, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(copy_19, memory_format = torch.contiguous_format)
        copy_20: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.copy.default(copy_19, clone_10);  copy_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_57: "f32[512, 40, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.slice.Tensor(copy_20, 1, 40, 80)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_102: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_57, [0, 2, 3])
        sub_248: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_826);  convolution_41 = unsqueeze_826 = None
        mul_964: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(slice_57, sub_248)
        sum_103: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_964, [0, 2, 3]);  mul_964 = None
        mul_965: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_102, 9.964923469387754e-06)
        unsqueeze_827: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_965, 0);  mul_965 = None
        unsqueeze_828: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 2);  unsqueeze_827 = None
        unsqueeze_829: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 3);  unsqueeze_828 = None
        mul_966: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, 9.964923469387754e-06)
        mul_967: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_968: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_966, mul_967);  mul_966 = mul_967 = None
        unsqueeze_830: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_968, 0);  mul_968 = None
        unsqueeze_831: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 2);  unsqueeze_830 = None
        unsqueeze_832: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 3);  unsqueeze_831 = None
        mul_969: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, primals_236);  primals_236 = None
        unsqueeze_833: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_969, 0);  mul_969 = None
        unsqueeze_834: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
        unsqueeze_835: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
        mul_970: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_832);  sub_248 = unsqueeze_832 = None
        sub_250: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(slice_57, mul_970);  slice_57 = mul_970 = None
        sub_251: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_829);  sub_250 = unsqueeze_829 = None
        mul_971: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_835);  sub_251 = unsqueeze_835 = None
        mul_972: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, squeeze_112);  sum_103 = squeeze_112 = None
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_971, add_192, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 40, [True, True, False]);  mul_971 = add_192 = primals_232 = None
        getitem_319: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = convolution_backward_53[0]
        getitem_320: "f32[40, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_53[1];  convolution_backward_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_58: "f32[512, 40, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.slice.Tensor(copy_20, 1, 0, 40);  copy_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_455: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.add.Tensor(slice_58, getitem_319);  slice_58 = getitem_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_104: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_455, [0, 2, 3])
        sub_252: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_838);  convolution_40 = unsqueeze_838 = None
        mul_973: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(add_455, sub_252)
        sum_105: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_973, [0, 2, 3]);  mul_973 = None
        mul_974: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_104, 9.964923469387754e-06)
        unsqueeze_839: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_974, 0);  mul_974 = None
        unsqueeze_840: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 2);  unsqueeze_839 = None
        unsqueeze_841: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 3);  unsqueeze_840 = None
        mul_975: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, 9.964923469387754e-06)
        mul_976: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_977: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_975, mul_976);  mul_975 = mul_976 = None
        unsqueeze_842: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_977, 0);  mul_977 = None
        unsqueeze_843: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 2);  unsqueeze_842 = None
        unsqueeze_844: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 3);  unsqueeze_843 = None
        mul_978: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, primals_230);  primals_230 = None
        unsqueeze_845: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_978, 0);  mul_978 = None
        unsqueeze_846: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
        unsqueeze_847: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
        mul_979: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_844);  sub_252 = unsqueeze_844 = None
        sub_254: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(add_455, mul_979);  add_455 = mul_979 = None
        sub_255: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_841);  sub_254 = unsqueeze_841 = None
        mul_980: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_847);  sub_255 = unsqueeze_847 = None
        mul_981: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, squeeze_109);  sum_105 = squeeze_109 = None
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_980, cat_12, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_980 = cat_12 = primals_226 = None
        getitem_322: "f32[512, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = convolution_backward_54[0]
        getitem_323: "f32[40, 200, 1, 1][200, 1, 200, 200]cuda:0" = convolution_backward_54[1];  convolution_backward_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_59: "f32[512, 100, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.slice.Tensor(getitem_322, 1, 0, 100)
        slice_60: "f32[512, 100, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.slice.Tensor(getitem_322, 1, 100, 200);  getitem_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_35: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.sub.Tensor(convolution_39, getitem_71)
        mul_247: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        unsqueeze_140: "f32[100, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_224, -1)
        unsqueeze_141: "f32[100, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_253: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.mul.Tensor(mul_247, unsqueeze_141);  mul_247 = unsqueeze_141 = None
        unsqueeze_142: "f32[100, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_225, -1);  primals_225 = None
        unsqueeze_143: "f32[100, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_187: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.add.Tensor(mul_253, unsqueeze_143);  mul_253 = unsqueeze_143 = None
        relu_16: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.relu.default(add_187);  add_187 = None
        le_25: "b8[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_30: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.where.self(le_25, full_default, slice_60);  le_25 = slice_60 = None
        squeeze_105: "f32[100][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        unsqueeze_848: "f32[1, 100][100, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
        unsqueeze_849: "f32[1, 100, 1][100, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 2);  unsqueeze_848 = None
        unsqueeze_850: "f32[1, 100, 1, 1][100, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 3);  unsqueeze_849 = None
        sum_106: "f32[100][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
        sub_256: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_850);  convolution_39 = unsqueeze_850 = None
        mul_982: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.mul.Tensor(where_30, sub_256)
        sum_107: "f32[100][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_982, [0, 2, 3]);  mul_982 = None
        mul_983: "f32[100][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_106, 9.964923469387754e-06)
        unsqueeze_851: "f32[1, 100][100, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
        unsqueeze_852: "f32[1, 100, 1][100, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 2);  unsqueeze_851 = None
        unsqueeze_853: "f32[1, 100, 1, 1][100, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 3);  unsqueeze_852 = None
        mul_984: "f32[100][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, 9.964923469387754e-06)
        squeeze_106: "f32[100][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_985: "f32[100][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_986: "f32[100][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_984, mul_985);  mul_984 = mul_985 = None
        unsqueeze_854: "f32[1, 100][100, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_986, 0);  mul_986 = None
        unsqueeze_855: "f32[1, 100, 1][100, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 2);  unsqueeze_854 = None
        unsqueeze_856: "f32[1, 100, 1, 1][100, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 3);  unsqueeze_855 = None
        mul_987: "f32[100][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, primals_224);  primals_224 = None
        unsqueeze_857: "f32[1, 100][100, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_987, 0);  mul_987 = None
        unsqueeze_858: "f32[1, 100, 1][100, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
        unsqueeze_859: "f32[1, 100, 1, 1][100, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
        mul_988: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_856);  sub_256 = unsqueeze_856 = None
        sub_258: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.sub.Tensor(where_30, mul_988);  where_30 = mul_988 = None
        sub_259: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_853);  sub_258 = unsqueeze_853 = None
        mul_989: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_859);  sub_259 = unsqueeze_859 = None
        mul_990: "f32[100][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, squeeze_106);  sum_107 = squeeze_106 = None
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_989, relu_15, primals_220, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 100, [True, True, False]);  mul_989 = primals_220 = None
        getitem_325: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = convolution_backward_55[0]
        getitem_326: "f32[100, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_55[1];  convolution_backward_55 = None
        add_456: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.add.Tensor(slice_59, getitem_325);  slice_59 = getitem_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_26: "b8[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_31: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.where.self(le_26, full_default, add_456);  le_26 = add_456 = None
        sum_108: "f32[100][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
        sub_260: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_862);  convolution_38 = unsqueeze_862 = None
        mul_991: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.mul.Tensor(where_31, sub_260)
        sum_109: "f32[100][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_991, [0, 2, 3]);  mul_991 = None
        mul_992: "f32[100][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_108, 9.964923469387754e-06)
        unsqueeze_863: "f32[1, 100][100, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_992, 0);  mul_992 = None
        unsqueeze_864: "f32[1, 100, 1][100, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 2);  unsqueeze_863 = None
        unsqueeze_865: "f32[1, 100, 1, 1][100, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 3);  unsqueeze_864 = None
        mul_993: "f32[100][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, 9.964923469387754e-06)
        mul_994: "f32[100][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_995: "f32[100][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_993, mul_994);  mul_993 = mul_994 = None
        unsqueeze_866: "f32[1, 100][100, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_995, 0);  mul_995 = None
        unsqueeze_867: "f32[1, 100, 1][100, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 2);  unsqueeze_866 = None
        unsqueeze_868: "f32[1, 100, 1, 1][100, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 3);  unsqueeze_867 = None
        mul_996: "f32[100][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, primals_218);  primals_218 = None
        unsqueeze_869: "f32[1, 100][100, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_996, 0);  mul_996 = None
        unsqueeze_870: "f32[1, 100, 1][100, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
        unsqueeze_871: "f32[1, 100, 1, 1][100, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
        mul_997: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_868);  sub_260 = unsqueeze_868 = None
        sub_262: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.sub.Tensor(where_31, mul_997);  where_31 = mul_997 = None
        sub_263: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.sub.Tensor(sub_262, unsqueeze_865);  sub_262 = unsqueeze_865 = None
        mul_998: "f32[512, 100, 14, 14][19600, 1, 1400, 100]cuda:0" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_871);  sub_263 = unsqueeze_871 = None
        mul_999: "f32[100][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, squeeze_103);  sum_109 = squeeze_103 = None
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_998, add_177, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_998 = add_177 = primals_214 = None
        getitem_328: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = convolution_backward_56[0]
        getitem_329: "f32[100, 80, 1, 1][80, 1, 80, 80]cuda:0" = convolution_backward_56[1];  convolution_backward_56 = None
        add_457: "f32[512, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(clone_10, getitem_328);  clone_10 = getitem_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        new_empty_strided_10: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.new_empty_strided.default(add_457, [512, 80, 14, 14], [15680, 1, 1120, 80])
        copy_21: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.copy.default(new_empty_strided_10, add_457);  new_empty_strided_10 = add_457 = None
        clone_11: "f32[512, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(copy_21, memory_format = torch.contiguous_format)
        copy_22: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.copy.default(copy_21, clone_11);  copy_21 = None
        sum_110: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(clone_11, [0, 2, 3])
        sub_264: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_874);  convolution_37 = unsqueeze_874 = None
        mul_1000: "f32[512, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(clone_11, sub_264)
        sum_111: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1000, [0, 2, 3]);  mul_1000 = None
        mul_1001: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_110, 9.964923469387754e-06)
        unsqueeze_875: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1001, 0);  mul_1001 = None
        unsqueeze_876: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 2);  unsqueeze_875 = None
        unsqueeze_877: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 3);  unsqueeze_876 = None
        mul_1002: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, 9.964923469387754e-06)
        mul_1003: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_1004: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1002, mul_1003);  mul_1002 = mul_1003 = None
        unsqueeze_878: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1004, 0);  mul_1004 = None
        unsqueeze_879: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 2);  unsqueeze_878 = None
        unsqueeze_880: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 3);  unsqueeze_879 = None
        mul_1005: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, primals_212);  primals_212 = None
        unsqueeze_881: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1005, 0);  mul_1005 = None
        unsqueeze_882: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
        unsqueeze_883: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
        mul_1006: "f32[512, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_880);  sub_264 = unsqueeze_880 = None
        sub_266: "f32[512, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_11, mul_1006);  clone_11 = mul_1006 = None
        sub_267: "f32[512, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_266, unsqueeze_877);  sub_266 = unsqueeze_877 = None
        mul_1007: "f32[512, 80, 14, 14][15680, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_883);  sub_267 = unsqueeze_883 = None
        mul_1008: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, squeeze_100);  sum_111 = squeeze_100 = None
        convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1007, add_171, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1007 = add_171 = primals_208 = None
        getitem_331: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = convolution_backward_57[0]
        getitem_332: "f32[80, 40, 1, 1][40, 1, 40, 40]cuda:0" = convolution_backward_57[1];  convolution_backward_57 = None
        sum_112: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_331, [0, 2, 3])
        sub_268: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_886);  convolution_36 = unsqueeze_886 = None
        mul_1009: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(getitem_331, sub_268)
        sum_113: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1009, [0, 2, 3]);  mul_1009 = None
        mul_1010: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_112, 9.964923469387754e-06)
        unsqueeze_887: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1010, 0);  mul_1010 = None
        unsqueeze_888: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 2);  unsqueeze_887 = None
        unsqueeze_889: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 3);  unsqueeze_888 = None
        mul_1011: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, 9.964923469387754e-06)
        mul_1012: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_1013: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1011, mul_1012);  mul_1011 = mul_1012 = None
        unsqueeze_890: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1013, 0);  mul_1013 = None
        unsqueeze_891: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 2);  unsqueeze_890 = None
        unsqueeze_892: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 3);  unsqueeze_891 = None
        mul_1014: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, primals_206);  primals_206 = None
        unsqueeze_893: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1014, 0);  mul_1014 = None
        unsqueeze_894: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
        unsqueeze_895: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
        mul_1015: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_892);  sub_268 = unsqueeze_892 = None
        sub_270: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(getitem_331, mul_1015);  getitem_331 = mul_1015 = None
        sub_271: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(sub_270, unsqueeze_889);  sub_270 = unsqueeze_889 = None
        mul_1016: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_895);  sub_271 = unsqueeze_895 = None
        mul_1017: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, squeeze_97);  sum_113 = squeeze_97 = None
        convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1016, add_141, primals_202, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 40, [True, True, False]);  mul_1016 = primals_202 = None
        getitem_334: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = convolution_backward_58[0]
        getitem_335: "f32[40, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_58[1];  convolution_backward_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_63: "f32[512, 40, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.slice.Tensor(copy_22, 1, 40, 80)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_114: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_63, [0, 2, 3])
        sub_272: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_898);  convolution_35 = unsqueeze_898 = None
        mul_1018: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(slice_63, sub_272)
        sum_115: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1018, [0, 2, 3]);  mul_1018 = None
        mul_1019: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_114, 9.964923469387754e-06)
        unsqueeze_899: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
        unsqueeze_900: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 2);  unsqueeze_899 = None
        unsqueeze_901: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 3);  unsqueeze_900 = None
        mul_1020: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, 9.964923469387754e-06)
        mul_1021: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_1022: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1020, mul_1021);  mul_1020 = mul_1021 = None
        unsqueeze_902: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1022, 0);  mul_1022 = None
        unsqueeze_903: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 2);  unsqueeze_902 = None
        unsqueeze_904: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 3);  unsqueeze_903 = None
        mul_1023: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, primals_200);  primals_200 = None
        unsqueeze_905: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1023, 0);  mul_1023 = None
        unsqueeze_906: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
        unsqueeze_907: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
        mul_1024: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_904);  sub_272 = unsqueeze_904 = None
        sub_274: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(slice_63, mul_1024);  slice_63 = mul_1024 = None
        sub_275: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(sub_274, unsqueeze_901);  sub_274 = unsqueeze_901 = None
        mul_1025: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_907);  sub_275 = unsqueeze_907 = None
        mul_1026: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, squeeze_94);  sum_115 = squeeze_94 = None
        convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_1025, add_161, primals_196, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 40, [True, True, False]);  mul_1025 = add_161 = primals_196 = None
        getitem_337: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = convolution_backward_59[0]
        getitem_338: "f32[40, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_59[1];  convolution_backward_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_64: "f32[512, 40, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.slice.Tensor(copy_22, 1, 0, 40);  copy_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_458: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.add.Tensor(slice_64, getitem_337);  slice_64 = getitem_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_116: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_458, [0, 2, 3])
        sub_276: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_910);  convolution_34 = unsqueeze_910 = None
        mul_1027: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(add_458, sub_276)
        sum_117: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1027, [0, 2, 3]);  mul_1027 = None
        mul_1028: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_116, 9.964923469387754e-06)
        unsqueeze_911: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1028, 0);  mul_1028 = None
        unsqueeze_912: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 2);  unsqueeze_911 = None
        unsqueeze_913: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 3);  unsqueeze_912 = None
        mul_1029: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, 9.964923469387754e-06)
        mul_1030: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_1031: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1029, mul_1030);  mul_1029 = mul_1030 = None
        unsqueeze_914: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1031, 0);  mul_1031 = None
        unsqueeze_915: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 2);  unsqueeze_914 = None
        unsqueeze_916: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 3);  unsqueeze_915 = None
        mul_1032: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, primals_194);  primals_194 = None
        unsqueeze_917: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1032, 0);  mul_1032 = None
        unsqueeze_918: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
        unsqueeze_919: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
        mul_1033: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_916);  sub_276 = unsqueeze_916 = None
        sub_278: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(add_458, mul_1033);  add_458 = mul_1033 = None
        sub_279: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.sub.Tensor(sub_278, unsqueeze_913);  sub_278 = unsqueeze_913 = None
        mul_1034: "f32[512, 40, 14, 14][7840, 1, 560, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_919);  sub_279 = unsqueeze_919 = None
        mul_1035: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, squeeze_91);  sum_117 = squeeze_91 = None
        convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1034, add_156, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1034 = add_156 = primals_190 = None
        getitem_340: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = convolution_backward_60[0]
        getitem_341: "f32[40, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_60[1];  convolution_backward_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        sum_118: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_340, [0, 2, 3])
        sub_280: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_922);  convolution_33 = unsqueeze_922 = None
        mul_1036: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(getitem_340, sub_280)
        sum_119: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1036, [0, 2, 3]);  mul_1036 = None
        mul_1037: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_118, 9.964923469387754e-06)
        unsqueeze_923: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1037, 0);  mul_1037 = None
        unsqueeze_924: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 2);  unsqueeze_923 = None
        unsqueeze_925: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 3);  unsqueeze_924 = None
        mul_1038: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, 9.964923469387754e-06)
        mul_1039: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_1040: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1038, mul_1039);  mul_1038 = mul_1039 = None
        unsqueeze_926: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1040, 0);  mul_1040 = None
        unsqueeze_927: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 2);  unsqueeze_926 = None
        unsqueeze_928: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 3);  unsqueeze_927 = None
        mul_1041: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, primals_188);  primals_188 = None
        unsqueeze_929: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1041, 0);  mul_1041 = None
        unsqueeze_930: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 2);  unsqueeze_929 = None
        unsqueeze_931: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 3);  unsqueeze_930 = None
        mul_1042: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_928);  sub_280 = unsqueeze_928 = None
        sub_282: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(getitem_340, mul_1042);  getitem_340 = mul_1042 = None
        sub_283: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(sub_282, unsqueeze_925);  sub_282 = unsqueeze_925 = None
        mul_1043: "f32[512, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_931);  sub_283 = unsqueeze_931 = None
        mul_1044: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, squeeze_88);  sum_119 = squeeze_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:435 in forward, code: x = self.conv_dw(x)
        convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1043, cat_10, primals_184, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_1043 = cat_10 = primals_184 = None
        getitem_343: "f32[512, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = convolution_backward_61[0]
        getitem_344: "f32[240, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_61[1];  convolution_backward_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_65: "f32[512, 120, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.slice.Tensor(getitem_343, 1, 0, 120)
        slice_66: "f32[512, 120, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.slice.Tensor(getitem_343, 1, 120, 240);  getitem_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_28: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convolution_32, getitem_57)
        mul_198: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        unsqueeze_112: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_182, -1)
        unsqueeze_113: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_204: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(mul_198, unsqueeze_113);  mul_198 = unsqueeze_113 = None
        unsqueeze_114: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_183, -1);  primals_183 = None
        unsqueeze_115: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_151: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.add.Tensor(mul_204, unsqueeze_115);  mul_204 = unsqueeze_115 = None
        relu_14: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.relu.default(add_151);  add_151 = None
        le_27: "b8[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_32: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.where.self(le_27, full_default, slice_66);  le_27 = slice_66 = None
        squeeze_84: "f32[120][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        unsqueeze_932: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_933: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 2);  unsqueeze_932 = None
        unsqueeze_934: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 3);  unsqueeze_933 = None
        sum_120: "f32[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
        sub_284: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_934);  convolution_32 = unsqueeze_934 = None
        mul_1045: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(where_32, sub_284)
        sum_121: "f32[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1045, [0, 2, 3]);  mul_1045 = None
        mul_1046: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_120, 2.4912308673469386e-06)
        unsqueeze_935: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1046, 0);  mul_1046 = None
        unsqueeze_936: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 2);  unsqueeze_935 = None
        unsqueeze_937: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_936, 3);  unsqueeze_936 = None
        mul_1047: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, 2.4912308673469386e-06)
        squeeze_85: "f32[120][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_1048: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_1049: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1047, mul_1048);  mul_1047 = mul_1048 = None
        unsqueeze_938: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1049, 0);  mul_1049 = None
        unsqueeze_939: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 2);  unsqueeze_938 = None
        unsqueeze_940: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 3);  unsqueeze_939 = None
        mul_1050: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, primals_182);  primals_182 = None
        unsqueeze_941: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1050, 0);  mul_1050 = None
        unsqueeze_942: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 2);  unsqueeze_941 = None
        unsqueeze_943: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 3);  unsqueeze_942 = None
        mul_1051: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_940);  sub_284 = unsqueeze_940 = None
        sub_286: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(where_32, mul_1051);  where_32 = mul_1051 = None
        sub_287: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(sub_286, unsqueeze_937);  sub_286 = unsqueeze_937 = None
        mul_1052: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_943);  sub_287 = unsqueeze_943 = None
        mul_1053: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, squeeze_85);  sum_121 = squeeze_85 = None
        convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1052, relu_13, primals_178, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_1052 = primals_178 = None
        getitem_346: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = convolution_backward_62[0]
        getitem_347: "f32[120, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_62[1];  convolution_backward_62 = None
        add_459: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.add.Tensor(slice_65, getitem_346);  slice_65 = getitem_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_28: "b8[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_33: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.where.self(le_28, full_default, add_459);  le_28 = add_459 = None
        sum_122: "f32[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
        sub_288: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_946);  convolution_31 = unsqueeze_946 = None
        mul_1054: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(where_33, sub_288)
        sum_123: "f32[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1054, [0, 2, 3]);  mul_1054 = None
        mul_1055: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_122, 2.4912308673469386e-06)
        unsqueeze_947: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1055, 0);  mul_1055 = None
        unsqueeze_948: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 2);  unsqueeze_947 = None
        unsqueeze_949: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 3);  unsqueeze_948 = None
        mul_1056: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, 2.4912308673469386e-06)
        mul_1057: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_1058: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1056, mul_1057);  mul_1056 = mul_1057 = None
        unsqueeze_950: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1058, 0);  mul_1058 = None
        unsqueeze_951: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 2);  unsqueeze_950 = None
        unsqueeze_952: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 3);  unsqueeze_951 = None
        mul_1059: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, primals_176);  primals_176 = None
        unsqueeze_953: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1059, 0);  mul_1059 = None
        unsqueeze_954: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 2);  unsqueeze_953 = None
        unsqueeze_955: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 3);  unsqueeze_954 = None
        mul_1060: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_952);  sub_288 = unsqueeze_952 = None
        sub_290: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(where_33, mul_1060);  where_33 = mul_1060 = None
        sub_291: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(sub_290, unsqueeze_949);  sub_290 = unsqueeze_949 = None
        mul_1061: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_955);  sub_291 = unsqueeze_955 = None
        mul_1062: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, squeeze_82);  sum_123 = squeeze_82 = None
        convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1061, add_141, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1061 = add_141 = primals_172 = None
        getitem_349: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = convolution_backward_63[0]
        getitem_350: "f32[120, 40, 1, 1][40, 1, 40, 40]cuda:0" = convolution_backward_63[1];  convolution_backward_63 = None
        add_460: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.add.Tensor(getitem_334, getitem_349);  getitem_334 = getitem_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        clone_12: "f32[512, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.clone.default(add_460, memory_format = torch.contiguous_format)
        copy_24: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.copy.default(add_460, clone_12);  add_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_69: "f32[512, 20, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.slice.Tensor(copy_24, 1, 20, 40)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_124: "f32[20][1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_69, [0, 2, 3])
        sub_292: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_958);  convolution_30 = unsqueeze_958 = None
        mul_1063: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.mul.Tensor(slice_69, sub_292)
        sum_125: "f32[20][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1063, [0, 2, 3]);  mul_1063 = None
        mul_1064: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_124, 2.4912308673469386e-06)
        unsqueeze_959: "f32[1, 20][20, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1064, 0);  mul_1064 = None
        unsqueeze_960: "f32[1, 20, 1][20, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 2);  unsqueeze_959 = None
        unsqueeze_961: "f32[1, 20, 1, 1][20, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 3);  unsqueeze_960 = None
        mul_1065: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_125, 2.4912308673469386e-06)
        mul_1066: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_1067: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1065, mul_1066);  mul_1065 = mul_1066 = None
        unsqueeze_962: "f32[1, 20][20, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1067, 0);  mul_1067 = None
        unsqueeze_963: "f32[1, 20, 1][20, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 2);  unsqueeze_962 = None
        unsqueeze_964: "f32[1, 20, 1, 1][20, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 3);  unsqueeze_963 = None
        mul_1068: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, primals_170);  primals_170 = None
        unsqueeze_965: "f32[1, 20][20, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1068, 0);  mul_1068 = None
        unsqueeze_966: "f32[1, 20, 1][20, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 2);  unsqueeze_965 = None
        unsqueeze_967: "f32[1, 20, 1, 1][20, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 3);  unsqueeze_966 = None
        mul_1069: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_964);  sub_292 = unsqueeze_964 = None
        sub_294: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.sub.Tensor(slice_69, mul_1069);  slice_69 = mul_1069 = None
        sub_295: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.sub.Tensor(sub_294, unsqueeze_961);  sub_294 = unsqueeze_961 = None
        mul_1070: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_967);  sub_295 = unsqueeze_967 = None
        mul_1071: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_125, squeeze_79);  sum_125 = squeeze_79 = None
        convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1070, add_135, primals_166, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 20, [True, True, False]);  mul_1070 = add_135 = primals_166 = None
        getitem_352: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = convolution_backward_64[0]
        getitem_353: "f32[20, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_64[1];  convolution_backward_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_70: "f32[512, 20, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.slice.Tensor(copy_24, 1, 0, 20);  copy_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_461: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.add.Tensor(slice_70, getitem_352);  slice_70 = getitem_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_126: "f32[20][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_461, [0, 2, 3])
        sub_296: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_970);  convolution_29 = unsqueeze_970 = None
        mul_1072: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.mul.Tensor(add_461, sub_296)
        sum_127: "f32[20][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1072, [0, 2, 3]);  mul_1072 = None
        mul_1073: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_126, 2.4912308673469386e-06)
        unsqueeze_971: "f32[1, 20][20, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1073, 0);  mul_1073 = None
        unsqueeze_972: "f32[1, 20, 1][20, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 2);  unsqueeze_971 = None
        unsqueeze_973: "f32[1, 20, 1, 1][20, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 3);  unsqueeze_972 = None
        mul_1074: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_127, 2.4912308673469386e-06)
        mul_1075: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_1076: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1074, mul_1075);  mul_1074 = mul_1075 = None
        unsqueeze_974: "f32[1, 20][20, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1076, 0);  mul_1076 = None
        unsqueeze_975: "f32[1, 20, 1][20, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 2);  unsqueeze_974 = None
        unsqueeze_976: "f32[1, 20, 1, 1][20, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 3);  unsqueeze_975 = None
        mul_1077: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, primals_164);  primals_164 = None
        unsqueeze_977: "f32[1, 20][20, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1077, 0);  mul_1077 = None
        unsqueeze_978: "f32[1, 20, 1][20, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
        unsqueeze_979: "f32[1, 20, 1, 1][20, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
        mul_1078: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_976);  sub_296 = unsqueeze_976 = None
        sub_298: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.sub.Tensor(add_461, mul_1078);  add_461 = mul_1078 = None
        sub_299: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.sub.Tensor(sub_298, unsqueeze_973);  sub_298 = unsqueeze_973 = None
        mul_1079: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_979);  sub_299 = unsqueeze_979 = None
        mul_1080: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_127, squeeze_76);  sum_127 = squeeze_76 = None
        convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1079, mul_176, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1079 = mul_176 = primals_160 = None
        getitem_355: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = convolution_backward_65[0]
        getitem_356: "f32[20, 120, 1, 1][120, 1, 120, 120]cuda:0" = convolution_backward_65[1];  convolution_backward_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_1081: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(getitem_355, cat_8);  cat_8 = None
        add_130: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.add.Tensor(convolution_28, 3)
        clamp_min_1: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.clamp_min.default(add_130, 0);  add_130 = None
        clamp_max_1: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
        div_1: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_1, 6);  clamp_max_1 = None
        mul_1082: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(getitem_355, div_1);  getitem_355 = div_1 = None
        sum_128: "f32[512, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1081, [2, 3], True);  mul_1081 = None
        gt_6: "b8[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.gt.Scalar(convolution_28, -3.0)
        lt_5: "b8[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.lt.Scalar(convolution_28, 3.0);  convolution_28 = None
        bitwise_and_5: "b8[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.bitwise_and.Tensor(gt_6, lt_5);  gt_6 = lt_5 = None
        mul_1083: "f32[512, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(sum_128, 0.16666666666666666);  sum_128 = None
        where_34: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.where.self(bitwise_and_5, mul_1083, full_default);  bitwise_and_5 = mul_1083 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_129: "f32[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
        convolution_backward_66 = torch.ops.aten.convolution_backward.default(where_34, relu_12, primals_158, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_34 = primals_158 = None
        getitem_358: "f32[512, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_66[0]
        getitem_359: "f32[120, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_66[1];  convolution_backward_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        le_29: "b8[512, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_35: "f32[512, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.where.self(le_29, full_default, getitem_358);  le_29 = getitem_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_130: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
        convolution_backward_67 = torch.ops.aten.convolution_backward.default(where_35, mean_1, primals_156, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_35 = mean_1 = primals_156 = None
        getitem_361: "f32[512, 120, 1, 1][120, 1, 120, 120]cuda:0" = convolution_backward_67[0]
        getitem_362: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = convolution_backward_67[1];  convolution_backward_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_7: "f32[512, 120, 28, 28][120, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_361, [512, 120, 28, 28]);  getitem_361 = None
        div_13: "f32[512, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_7, 784);  expand_7 = None
        add_462: "f32[512, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.add.Tensor(mul_1082, div_13);  mul_1082 = div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_71: "f32[512, 60, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.slice.Tensor(add_462, 1, 0, 60)
        slice_72: "f32[512, 60, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.slice.Tensor(add_462, 1, 60, 120);  add_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_24: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, getitem_49)
        mul_169: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        unsqueeze_96: "f32[60, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_154, -1)
        unsqueeze_97: "f32[60, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_175: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_97);  mul_169 = unsqueeze_97 = None
        unsqueeze_98: "f32[60, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_155, -1);  primals_155 = None
        unsqueeze_99: "f32[60, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_129: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.add.Tensor(mul_175, unsqueeze_99);  mul_175 = unsqueeze_99 = None
        relu_11: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.relu.default(add_129);  add_129 = None
        le_30: "b8[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_36: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.where.self(le_30, full_default, slice_72);  le_30 = slice_72 = None
        squeeze_72: "f32[60][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        unsqueeze_980: "f32[1, 60][60, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_981: "f32[1, 60, 1][60, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 2);  unsqueeze_980 = None
        unsqueeze_982: "f32[1, 60, 1, 1][60, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 3);  unsqueeze_981 = None
        sum_131: "f32[60][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
        sub_300: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_982);  convolution_26 = unsqueeze_982 = None
        mul_1084: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.mul.Tensor(where_36, sub_300)
        sum_132: "f32[60][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1084, [0, 2, 3]);  mul_1084 = None
        mul_1085: "f32[60][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_131, 2.4912308673469386e-06)
        unsqueeze_983: "f32[1, 60][60, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1085, 0);  mul_1085 = None
        unsqueeze_984: "f32[1, 60, 1][60, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 2);  unsqueeze_983 = None
        unsqueeze_985: "f32[1, 60, 1, 1][60, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 3);  unsqueeze_984 = None
        mul_1086: "f32[60][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_132, 2.4912308673469386e-06)
        squeeze_73: "f32[60][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_1087: "f32[60][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_1088: "f32[60][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1086, mul_1087);  mul_1086 = mul_1087 = None
        unsqueeze_986: "f32[1, 60][60, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1088, 0);  mul_1088 = None
        unsqueeze_987: "f32[1, 60, 1][60, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 2);  unsqueeze_986 = None
        unsqueeze_988: "f32[1, 60, 1, 1][60, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 3);  unsqueeze_987 = None
        mul_1089: "f32[60][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, primals_154);  primals_154 = None
        unsqueeze_989: "f32[1, 60][60, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1089, 0);  mul_1089 = None
        unsqueeze_990: "f32[1, 60, 1][60, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 2);  unsqueeze_989 = None
        unsqueeze_991: "f32[1, 60, 1, 1][60, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 3);  unsqueeze_990 = None
        mul_1090: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_988);  sub_300 = unsqueeze_988 = None
        sub_302: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.sub.Tensor(where_36, mul_1090);  where_36 = mul_1090 = None
        sub_303: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.sub.Tensor(sub_302, unsqueeze_985);  sub_302 = unsqueeze_985 = None
        mul_1091: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_991);  sub_303 = unsqueeze_991 = None
        mul_1092: "f32[60][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_132, squeeze_73);  sum_132 = squeeze_73 = None
        convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1091, relu_10, primals_150, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 60, [True, True, False]);  mul_1091 = primals_150 = None
        getitem_364: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = convolution_backward_68[0]
        getitem_365: "f32[60, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_68[1];  convolution_backward_68 = None
        add_463: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.add.Tensor(slice_71, getitem_364);  slice_71 = getitem_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_31: "b8[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_37: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.where.self(le_31, full_default, add_463);  le_31 = add_463 = None
        sum_133: "f32[60][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
        sub_304: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_994);  convolution_25 = unsqueeze_994 = None
        mul_1093: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.mul.Tensor(where_37, sub_304)
        sum_134: "f32[60][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1093, [0, 2, 3]);  mul_1093 = None
        mul_1094: "f32[60][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_133, 2.4912308673469386e-06)
        unsqueeze_995: "f32[1, 60][60, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1094, 0);  mul_1094 = None
        unsqueeze_996: "f32[1, 60, 1][60, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 2);  unsqueeze_995 = None
        unsqueeze_997: "f32[1, 60, 1, 1][60, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_996, 3);  unsqueeze_996 = None
        mul_1095: "f32[60][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_134, 2.4912308673469386e-06)
        mul_1096: "f32[60][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_1097: "f32[60][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1095, mul_1096);  mul_1095 = mul_1096 = None
        unsqueeze_998: "f32[1, 60][60, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1097, 0);  mul_1097 = None
        unsqueeze_999: "f32[1, 60, 1][60, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 2);  unsqueeze_998 = None
        unsqueeze_1000: "f32[1, 60, 1, 1][60, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 3);  unsqueeze_999 = None
        mul_1098: "f32[60][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, primals_148);  primals_148 = None
        unsqueeze_1001: "f32[1, 60][60, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1098, 0);  mul_1098 = None
        unsqueeze_1002: "f32[1, 60, 1][60, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 2);  unsqueeze_1001 = None
        unsqueeze_1003: "f32[1, 60, 1, 1][60, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 3);  unsqueeze_1002 = None
        mul_1099: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_1000);  sub_304 = unsqueeze_1000 = None
        sub_306: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.sub.Tensor(where_37, mul_1099);  where_37 = mul_1099 = None
        sub_307: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.sub.Tensor(sub_306, unsqueeze_997);  sub_306 = unsqueeze_997 = None
        mul_1100: "f32[512, 60, 28, 28][47040, 1, 1680, 60]cuda:0" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_1003);  sub_307 = unsqueeze_1003 = None
        mul_1101: "f32[60][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_134, squeeze_70);  sum_134 = squeeze_70 = None
        convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1100, add_119, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1100 = add_119 = primals_144 = None
        getitem_367: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = convolution_backward_69[0]
        getitem_368: "f32[60, 40, 1, 1][40, 1, 40, 40]cuda:0" = convolution_backward_69[1];  convolution_backward_69 = None
        add_464: "f32[512, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(clone_12, getitem_367);  clone_12 = getitem_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        new_empty_strided_12: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.new_empty_strided.default(add_464, [512, 40, 28, 28], [31360, 1, 1120, 40])
        copy_25: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.copy.default(new_empty_strided_12, add_464);  new_empty_strided_12 = add_464 = None
        clone_13: "f32[512, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.clone.default(copy_25, memory_format = torch.contiguous_format)
        copy_26: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.copy.default(copy_25, clone_13);  copy_25 = None
        sum_135: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(clone_13, [0, 2, 3])
        sub_308: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1006);  convolution_24 = unsqueeze_1006 = None
        mul_1102: "f32[512, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(clone_13, sub_308)
        sum_136: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1102, [0, 2, 3]);  mul_1102 = None
        mul_1103: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_135, 2.4912308673469386e-06)
        unsqueeze_1007: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1103, 0);  mul_1103 = None
        unsqueeze_1008: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 2);  unsqueeze_1007 = None
        unsqueeze_1009: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, 3);  unsqueeze_1008 = None
        mul_1104: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_136, 2.4912308673469386e-06)
        mul_1105: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_1106: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1104, mul_1105);  mul_1104 = mul_1105 = None
        unsqueeze_1010: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1106, 0);  mul_1106 = None
        unsqueeze_1011: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 2);  unsqueeze_1010 = None
        unsqueeze_1012: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 3);  unsqueeze_1011 = None
        mul_1107: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, primals_142);  primals_142 = None
        unsqueeze_1013: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1107, 0);  mul_1107 = None
        unsqueeze_1014: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 2);  unsqueeze_1013 = None
        unsqueeze_1015: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 3);  unsqueeze_1014 = None
        mul_1108: "f32[512, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_1012);  sub_308 = unsqueeze_1012 = None
        sub_310: "f32[512, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_13, mul_1108);  clone_13 = mul_1108 = None
        sub_311: "f32[512, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_310, unsqueeze_1009);  sub_310 = unsqueeze_1009 = None
        mul_1109: "f32[512, 40, 28, 28][31360, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_1015);  sub_311 = unsqueeze_1015 = None
        mul_1110: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_136, squeeze_67);  sum_136 = squeeze_67 = None
        convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1109, add_113, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1109 = add_113 = primals_138 = None
        getitem_370: "f32[512, 24, 28, 28][18816, 1, 672, 24]cuda:0" = convolution_backward_70[0]
        getitem_371: "f32[40, 24, 1, 1][24, 1, 24, 24]cuda:0" = convolution_backward_70[1];  convolution_backward_70 = None
        sum_137: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_370, [0, 2, 3])
        sub_312: "f32[512, 24, 28, 28][18816, 1, 672, 24]cuda:0" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1018);  convolution_23 = unsqueeze_1018 = None
        mul_1111: "f32[512, 24, 28, 28][18816, 1, 672, 24]cuda:0" = torch.ops.aten.mul.Tensor(getitem_370, sub_312)
        sum_138: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1111, [0, 2, 3]);  mul_1111 = None
        mul_1112: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_137, 2.4912308673469386e-06)
        unsqueeze_1019: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1112, 0);  mul_1112 = None
        unsqueeze_1020: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 2);  unsqueeze_1019 = None
        unsqueeze_1021: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, 3);  unsqueeze_1020 = None
        mul_1113: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_138, 2.4912308673469386e-06)
        mul_1114: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_1115: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1113, mul_1114);  mul_1113 = mul_1114 = None
        unsqueeze_1022: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1115, 0);  mul_1115 = None
        unsqueeze_1023: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 2);  unsqueeze_1022 = None
        unsqueeze_1024: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 3);  unsqueeze_1023 = None
        mul_1116: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, primals_136);  primals_136 = None
        unsqueeze_1025: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1116, 0);  mul_1116 = None
        unsqueeze_1026: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 2);  unsqueeze_1025 = None
        unsqueeze_1027: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 3);  unsqueeze_1026 = None
        mul_1117: "f32[512, 24, 28, 28][18816, 1, 672, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1024);  sub_312 = unsqueeze_1024 = None
        sub_314: "f32[512, 24, 28, 28][18816, 1, 672, 24]cuda:0" = torch.ops.aten.sub.Tensor(getitem_370, mul_1117);  getitem_370 = mul_1117 = None
        sub_315: "f32[512, 24, 28, 28][18816, 1, 672, 24]cuda:0" = torch.ops.aten.sub.Tensor(sub_314, unsqueeze_1021);  sub_314 = unsqueeze_1021 = None
        mul_1118: "f32[512, 24, 28, 28][18816, 1, 672, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_1027);  sub_315 = unsqueeze_1027 = None
        mul_1119: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_138, squeeze_64);  sum_138 = squeeze_64 = None
        convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1118, add_82, primals_132, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 24, [True, True, False]);  mul_1118 = primals_132 = None
        getitem_373: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = convolution_backward_71[0]
        getitem_374: "f32[24, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_71[1];  convolution_backward_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_75: "f32[512, 20, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.slice.Tensor(copy_26, 1, 20, 40)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_139: "f32[20][1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_75, [0, 2, 3])
        sub_316: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1030);  convolution_22 = unsqueeze_1030 = None
        mul_1120: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.mul.Tensor(slice_75, sub_316)
        sum_140: "f32[20][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1120, [0, 2, 3]);  mul_1120 = None
        mul_1121: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_139, 2.4912308673469386e-06)
        unsqueeze_1031: "f32[1, 20][20, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1121, 0);  mul_1121 = None
        unsqueeze_1032: "f32[1, 20, 1][20, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 2);  unsqueeze_1031 = None
        unsqueeze_1033: "f32[1, 20, 1, 1][20, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, 3);  unsqueeze_1032 = None
        mul_1122: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_140, 2.4912308673469386e-06)
        mul_1123: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_1124: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1122, mul_1123);  mul_1122 = mul_1123 = None
        unsqueeze_1034: "f32[1, 20][20, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1124, 0);  mul_1124 = None
        unsqueeze_1035: "f32[1, 20, 1][20, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 2);  unsqueeze_1034 = None
        unsqueeze_1036: "f32[1, 20, 1, 1][20, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 3);  unsqueeze_1035 = None
        mul_1125: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, primals_130);  primals_130 = None
        unsqueeze_1037: "f32[1, 20][20, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1125, 0);  mul_1125 = None
        unsqueeze_1038: "f32[1, 20, 1][20, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 2);  unsqueeze_1037 = None
        unsqueeze_1039: "f32[1, 20, 1, 1][20, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 3);  unsqueeze_1038 = None
        mul_1126: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1036);  sub_316 = unsqueeze_1036 = None
        sub_318: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.sub.Tensor(slice_75, mul_1126);  slice_75 = mul_1126 = None
        sub_319: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.sub.Tensor(sub_318, unsqueeze_1033);  sub_318 = unsqueeze_1033 = None
        mul_1127: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_1039);  sub_319 = unsqueeze_1039 = None
        mul_1128: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_140, squeeze_61);  sum_140 = squeeze_61 = None
        convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1127, add_103, primals_126, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 20, [True, True, False]);  mul_1127 = add_103 = primals_126 = None
        getitem_376: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = convolution_backward_72[0]
        getitem_377: "f32[20, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_72[1];  convolution_backward_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_76: "f32[512, 20, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.slice.Tensor(copy_26, 1, 0, 20);  copy_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_465: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.add.Tensor(slice_76, getitem_376);  slice_76 = getitem_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_141: "f32[20][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_465, [0, 2, 3])
        sub_320: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1042);  convolution_21 = unsqueeze_1042 = None
        mul_1129: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.mul.Tensor(add_465, sub_320)
        sum_142: "f32[20][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1129, [0, 2, 3]);  mul_1129 = None
        mul_1130: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_141, 2.4912308673469386e-06)
        unsqueeze_1043: "f32[1, 20][20, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1130, 0);  mul_1130 = None
        unsqueeze_1044: "f32[1, 20, 1][20, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1043, 2);  unsqueeze_1043 = None
        unsqueeze_1045: "f32[1, 20, 1, 1][20, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, 3);  unsqueeze_1044 = None
        mul_1131: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_142, 2.4912308673469386e-06)
        mul_1132: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_1133: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1131, mul_1132);  mul_1131 = mul_1132 = None
        unsqueeze_1046: "f32[1, 20][20, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1133, 0);  mul_1133 = None
        unsqueeze_1047: "f32[1, 20, 1][20, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 2);  unsqueeze_1046 = None
        unsqueeze_1048: "f32[1, 20, 1, 1][20, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1047, 3);  unsqueeze_1047 = None
        mul_1134: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, primals_124);  primals_124 = None
        unsqueeze_1049: "f32[1, 20][20, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1134, 0);  mul_1134 = None
        unsqueeze_1050: "f32[1, 20, 1][20, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 2);  unsqueeze_1049 = None
        unsqueeze_1051: "f32[1, 20, 1, 1][20, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 3);  unsqueeze_1050 = None
        mul_1135: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_1048);  sub_320 = unsqueeze_1048 = None
        sub_322: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.sub.Tensor(add_465, mul_1135);  add_465 = mul_1135 = None
        sub_323: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.sub.Tensor(sub_322, unsqueeze_1045);  sub_322 = unsqueeze_1045 = None
        mul_1136: "f32[512, 20, 28, 28][15680, 1, 560, 20]cuda:0" = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_1051);  sub_323 = unsqueeze_1051 = None
        mul_1137: "f32[20][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_142, squeeze_58);  sum_142 = squeeze_58 = None
        convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1136, mul_133, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1136 = mul_133 = primals_120 = None
        getitem_379: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = convolution_backward_73[0]
        getitem_380: "f32[20, 72, 1, 1][72, 1, 72, 72]cuda:0" = convolution_backward_73[1];  convolution_backward_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        sub_18: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, getitem_37)
        mul_126: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        unsqueeze_72: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_114, -1)
        unsqueeze_73: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_132: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
        unsqueeze_74: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
        unsqueeze_75: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_97: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_1138: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(getitem_379, add_97);  add_97 = None
        add_98: "f32[512, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.add.Tensor(convolution_20, 3)
        clamp_min: "f32[512, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.clamp_min.default(add_98, 0);  add_98 = None
        clamp_max: "f32[512, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
        div: "f32[512, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.div.Tensor(clamp_max, 6);  clamp_max = None
        mul_1139: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(getitem_379, div);  getitem_379 = div = None
        sum_143: "f32[512, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1138, [2, 3], True);  mul_1138 = None
        gt_7: "b8[512, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.gt.Scalar(convolution_20, -3.0)
        lt_6: "b8[512, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.lt.Scalar(convolution_20, 3.0);  convolution_20 = None
        bitwise_and_6: "b8[512, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.bitwise_and.Tensor(gt_7, lt_6);  gt_7 = lt_6 = None
        mul_1140: "f32[512, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(sum_143, 0.16666666666666666);  sum_143 = None
        where_38: "f32[512, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.where.self(bitwise_and_6, mul_1140, full_default);  bitwise_and_6 = mul_1140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_144: "f32[72][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
        convolution_backward_74 = torch.ops.aten.convolution_backward.default(where_38, relu_9, primals_118, [72], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_38 = primals_118 = None
        getitem_382: "f32[512, 20, 1, 1][20, 1, 20, 20]cuda:0" = convolution_backward_74[0]
        getitem_383: "f32[72, 20, 1, 1][20, 1, 20, 20]cuda:0" = convolution_backward_74[1];  convolution_backward_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        le_32: "b8[512, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_39: "f32[512, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.where.self(le_32, full_default, getitem_382);  le_32 = getitem_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_145: "f32[20][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
        convolution_backward_75 = torch.ops.aten.convolution_backward.default(where_39, mean, primals_116, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_39 = mean = primals_116 = None
        getitem_385: "f32[512, 72, 1, 1][72, 1, 72, 72]cuda:0" = convolution_backward_75[0]
        getitem_386: "f32[20, 72, 1, 1][72, 1, 72, 72]cuda:0" = convolution_backward_75[1];  convolution_backward_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_8: "f32[512, 72, 28, 28][72, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_385, [512, 72, 28, 28]);  getitem_385 = None
        div_14: "f32[512, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_8, 784);  expand_8 = None
        add_466: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.add.Tensor(mul_1139, div_14);  mul_1139 = div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        squeeze_54: "f32[72][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        unsqueeze_1052: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_1053: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, 2);  unsqueeze_1052 = None
        unsqueeze_1054: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1053, 3);  unsqueeze_1053 = None
        sum_146: "f32[72][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_466, [0, 2, 3])
        sub_324: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1054);  convolution_18 = unsqueeze_1054 = None
        mul_1141: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(add_466, sub_324)
        sum_147: "f32[72][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1141, [0, 2, 3]);  mul_1141 = None
        mul_1142: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_146, 2.4912308673469386e-06)
        unsqueeze_1055: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1142, 0);  mul_1142 = None
        unsqueeze_1056: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1055, 2);  unsqueeze_1055 = None
        unsqueeze_1057: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, 3);  unsqueeze_1056 = None
        mul_1143: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_147, 2.4912308673469386e-06)
        squeeze_55: "f32[72][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_1144: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_1145: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1143, mul_1144);  mul_1143 = mul_1144 = None
        unsqueeze_1058: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1145, 0);  mul_1145 = None
        unsqueeze_1059: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 2);  unsqueeze_1058 = None
        unsqueeze_1060: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1059, 3);  unsqueeze_1059 = None
        mul_1146: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, primals_114);  primals_114 = None
        unsqueeze_1061: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1146, 0);  mul_1146 = None
        unsqueeze_1062: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 2);  unsqueeze_1061 = None
        unsqueeze_1063: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 3);  unsqueeze_1062 = None
        mul_1147: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1060);  sub_324 = unsqueeze_1060 = None
        sub_326: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.sub.Tensor(add_466, mul_1147);  add_466 = mul_1147 = None
        sub_327: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.sub.Tensor(sub_326, unsqueeze_1057);  sub_326 = unsqueeze_1057 = None
        mul_1148: "f32[512, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_1063);  sub_327 = unsqueeze_1063 = None
        mul_1149: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_147, squeeze_55);  sum_147 = squeeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:435 in forward, code: x = self.conv_dw(x)
        convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1148, cat_6, primals_110, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_1148 = cat_6 = primals_110 = None
        getitem_388: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = convolution_backward_76[0]
        getitem_389: "f32[72, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_76[1];  convolution_backward_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_77: "f32[512, 36, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.slice.Tensor(getitem_388, 1, 0, 36)
        slice_78: "f32[512, 36, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.slice.Tensor(getitem_388, 1, 36, 72);  getitem_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_17: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.sub.Tensor(convolution_17, getitem_35)
        mul_119: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        unsqueeze_68: "f32[36, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_108, -1)
        unsqueeze_69: "f32[36, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_125: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
        unsqueeze_70: "f32[36, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
        unsqueeze_71: "f32[36, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_92: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
        relu_8: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.relu.default(add_92);  add_92 = None
        le_33: "b8[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_40: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.where.self(le_33, full_default, slice_78);  le_33 = slice_78 = None
        squeeze_51: "f32[36][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        unsqueeze_1064: "f32[1, 36][36, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_1065: "f32[1, 36, 1][36, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, 2);  unsqueeze_1064 = None
        unsqueeze_1066: "f32[1, 36, 1, 1][36, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1065, 3);  unsqueeze_1065 = None
        sum_148: "f32[36][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
        sub_328: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1066);  convolution_17 = unsqueeze_1066 = None
        mul_1150: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.mul.Tensor(where_40, sub_328)
        sum_149: "f32[36][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1150, [0, 2, 3]);  mul_1150 = None
        mul_1151: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_148, 6.228077168367346e-07)
        unsqueeze_1067: "f32[1, 36][36, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1151, 0);  mul_1151 = None
        unsqueeze_1068: "f32[1, 36, 1][36, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1067, 2);  unsqueeze_1067 = None
        unsqueeze_1069: "f32[1, 36, 1, 1][36, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, 3);  unsqueeze_1068 = None
        mul_1152: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_149, 6.228077168367346e-07)
        squeeze_52: "f32[36][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_1153: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_1154: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1152, mul_1153);  mul_1152 = mul_1153 = None
        unsqueeze_1070: "f32[1, 36][36, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1154, 0);  mul_1154 = None
        unsqueeze_1071: "f32[1, 36, 1][36, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, 2);  unsqueeze_1070 = None
        unsqueeze_1072: "f32[1, 36, 1, 1][36, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1071, 3);  unsqueeze_1071 = None
        mul_1155: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, primals_108);  primals_108 = None
        unsqueeze_1073: "f32[1, 36][36, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1155, 0);  mul_1155 = None
        unsqueeze_1074: "f32[1, 36, 1][36, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1073, 2);  unsqueeze_1073 = None
        unsqueeze_1075: "f32[1, 36, 1, 1][36, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 3);  unsqueeze_1074 = None
        mul_1156: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_1072);  sub_328 = unsqueeze_1072 = None
        sub_330: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.sub.Tensor(where_40, mul_1156);  where_40 = mul_1156 = None
        sub_331: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.sub.Tensor(sub_330, unsqueeze_1069);  sub_330 = unsqueeze_1069 = None
        mul_1157: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_1075);  sub_331 = unsqueeze_1075 = None
        mul_1158: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_149, squeeze_52);  sum_149 = squeeze_52 = None
        convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1157, relu_7, primals_104, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 36, [True, True, False]);  mul_1157 = primals_104 = None
        getitem_391: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = convolution_backward_77[0]
        getitem_392: "f32[36, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_77[1];  convolution_backward_77 = None
        add_467: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.add.Tensor(slice_77, getitem_391);  slice_77 = getitem_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_34: "b8[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_41: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.where.self(le_34, full_default, add_467);  le_34 = add_467 = None
        sum_150: "f32[36][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
        sub_332: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1078);  convolution_16 = unsqueeze_1078 = None
        mul_1159: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.mul.Tensor(where_41, sub_332)
        sum_151: "f32[36][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1159, [0, 2, 3]);  mul_1159 = None
        mul_1160: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_150, 6.228077168367346e-07)
        unsqueeze_1079: "f32[1, 36][36, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1160, 0);  mul_1160 = None
        unsqueeze_1080: "f32[1, 36, 1][36, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1079, 2);  unsqueeze_1079 = None
        unsqueeze_1081: "f32[1, 36, 1, 1][36, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, 3);  unsqueeze_1080 = None
        mul_1161: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_151, 6.228077168367346e-07)
        mul_1162: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_1163: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1161, mul_1162);  mul_1161 = mul_1162 = None
        unsqueeze_1082: "f32[1, 36][36, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1163, 0);  mul_1163 = None
        unsqueeze_1083: "f32[1, 36, 1][36, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, 2);  unsqueeze_1082 = None
        unsqueeze_1084: "f32[1, 36, 1, 1][36, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1083, 3);  unsqueeze_1083 = None
        mul_1164: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, primals_102);  primals_102 = None
        unsqueeze_1085: "f32[1, 36][36, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1164, 0);  mul_1164 = None
        unsqueeze_1086: "f32[1, 36, 1][36, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1085, 2);  unsqueeze_1085 = None
        unsqueeze_1087: "f32[1, 36, 1, 1][36, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 3);  unsqueeze_1086 = None
        mul_1165: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_1084);  sub_332 = unsqueeze_1084 = None
        sub_334: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.sub.Tensor(where_41, mul_1165);  where_41 = mul_1165 = None
        sub_335: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.sub.Tensor(sub_334, unsqueeze_1081);  sub_334 = unsqueeze_1081 = None
        mul_1166: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_1087);  sub_335 = unsqueeze_1087 = None
        mul_1167: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_151, squeeze_49);  sum_151 = squeeze_49 = None
        convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1166, add_82, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1166 = add_82 = primals_98 = None
        getitem_394: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = convolution_backward_78[0]
        getitem_395: "f32[36, 24, 1, 1][24, 1, 24, 24]cuda:0" = convolution_backward_78[1];  convolution_backward_78 = None
        add_468: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.add.Tensor(getitem_373, getitem_394);  getitem_373 = getitem_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        clone_14: "f32[512, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.clone.default(add_468, memory_format = torch.contiguous_format)
        copy_28: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.copy.default(add_468, clone_14);  add_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_81: "f32[512, 12, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.slice.Tensor(copy_28, 1, 12, 24)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_152: "f32[12][1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_81, [0, 2, 3])
        sub_336: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1090);  convolution_15 = unsqueeze_1090 = None
        mul_1168: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.mul.Tensor(slice_81, sub_336)
        sum_153: "f32[12][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1168, [0, 2, 3]);  mul_1168 = None
        mul_1169: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_152, 6.228077168367346e-07)
        unsqueeze_1091: "f32[1, 12][12, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1169, 0);  mul_1169 = None
        unsqueeze_1092: "f32[1, 12, 1][12, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1091, 2);  unsqueeze_1091 = None
        unsqueeze_1093: "f32[1, 12, 1, 1][12, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, 3);  unsqueeze_1092 = None
        mul_1170: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_153, 6.228077168367346e-07)
        mul_1171: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_1172: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1170, mul_1171);  mul_1170 = mul_1171 = None
        unsqueeze_1094: "f32[1, 12][12, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1172, 0);  mul_1172 = None
        unsqueeze_1095: "f32[1, 12, 1][12, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, 2);  unsqueeze_1094 = None
        unsqueeze_1096: "f32[1, 12, 1, 1][12, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1095, 3);  unsqueeze_1095 = None
        mul_1173: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, primals_96);  primals_96 = None
        unsqueeze_1097: "f32[1, 12][12, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1173, 0);  mul_1173 = None
        unsqueeze_1098: "f32[1, 12, 1][12, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1097, 2);  unsqueeze_1097 = None
        unsqueeze_1099: "f32[1, 12, 1, 1][12, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 3);  unsqueeze_1098 = None
        mul_1174: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_1096);  sub_336 = unsqueeze_1096 = None
        sub_338: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.sub.Tensor(slice_81, mul_1174);  slice_81 = mul_1174 = None
        sub_339: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.sub.Tensor(sub_338, unsqueeze_1093);  sub_338 = unsqueeze_1093 = None
        mul_1175: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_1099);  sub_339 = unsqueeze_1099 = None
        mul_1176: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_153, squeeze_46);  sum_153 = squeeze_46 = None
        convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1175, add_76, primals_92, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 12, [True, True, False]);  mul_1175 = add_76 = primals_92 = None
        getitem_397: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = convolution_backward_79[0]
        getitem_398: "f32[12, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_79[1];  convolution_backward_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_82: "f32[512, 12, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.slice.Tensor(copy_28, 1, 0, 12);  copy_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_469: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.add.Tensor(slice_82, getitem_397);  slice_82 = getitem_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_154: "f32[12][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_469, [0, 2, 3])
        sub_340: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1102);  convolution_14 = unsqueeze_1102 = None
        mul_1177: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.mul.Tensor(add_469, sub_340)
        sum_155: "f32[12][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1177, [0, 2, 3]);  mul_1177 = None
        mul_1178: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_154, 6.228077168367346e-07)
        unsqueeze_1103: "f32[1, 12][12, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1178, 0);  mul_1178 = None
        unsqueeze_1104: "f32[1, 12, 1][12, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1103, 2);  unsqueeze_1103 = None
        unsqueeze_1105: "f32[1, 12, 1, 1][12, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, 3);  unsqueeze_1104 = None
        mul_1179: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_155, 6.228077168367346e-07)
        mul_1180: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_1181: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1179, mul_1180);  mul_1179 = mul_1180 = None
        unsqueeze_1106: "f32[1, 12][12, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1181, 0);  mul_1181 = None
        unsqueeze_1107: "f32[1, 12, 1][12, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, 2);  unsqueeze_1106 = None
        unsqueeze_1108: "f32[1, 12, 1, 1][12, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1107, 3);  unsqueeze_1107 = None
        mul_1182: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, primals_90);  primals_90 = None
        unsqueeze_1109: "f32[1, 12][12, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1182, 0);  mul_1182 = None
        unsqueeze_1110: "f32[1, 12, 1][12, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1109, 2);  unsqueeze_1109 = None
        unsqueeze_1111: "f32[1, 12, 1, 1][12, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 3);  unsqueeze_1110 = None
        mul_1183: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_1108);  sub_340 = unsqueeze_1108 = None
        sub_342: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.sub.Tensor(add_469, mul_1183);  add_469 = mul_1183 = None
        sub_343: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.sub.Tensor(sub_342, unsqueeze_1105);  sub_342 = unsqueeze_1105 = None
        mul_1184: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_1111);  sub_343 = unsqueeze_1111 = None
        mul_1185: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_155, squeeze_43);  sum_155 = squeeze_43 = None
        convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1184, cat_4, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1184 = cat_4 = primals_86 = None
        getitem_400: "f32[512, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = convolution_backward_80[0]
        getitem_401: "f32[12, 72, 1, 1][72, 1, 72, 72]cuda:0" = convolution_backward_80[1];  convolution_backward_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_83: "f32[512, 36, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.slice.Tensor(getitem_400, 1, 0, 36)
        slice_84: "f32[512, 36, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.slice.Tensor(getitem_400, 1, 36, 72);  getitem_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_13: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
        mul_91: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        unsqueeze_52: "f32[36, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_53: "f32[36, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[36, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
        unsqueeze_55: "f32[36, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_71: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
        relu_6: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.relu.default(add_71);  add_71 = None
        le_35: "b8[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_42: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.where.self(le_35, full_default, slice_84);  le_35 = slice_84 = None
        squeeze_39: "f32[36][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        unsqueeze_1112: "f32[1, 36][36, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_1113: "f32[1, 36, 1][36, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, 2);  unsqueeze_1112 = None
        unsqueeze_1114: "f32[1, 36, 1, 1][36, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1113, 3);  unsqueeze_1113 = None
        sum_156: "f32[36][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
        sub_344: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1114);  convolution_13 = unsqueeze_1114 = None
        mul_1186: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.mul.Tensor(where_42, sub_344)
        sum_157: "f32[36][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1186, [0, 2, 3]);  mul_1186 = None
        mul_1187: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_156, 6.228077168367346e-07)
        unsqueeze_1115: "f32[1, 36][36, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1187, 0);  mul_1187 = None
        unsqueeze_1116: "f32[1, 36, 1][36, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1115, 2);  unsqueeze_1115 = None
        unsqueeze_1117: "f32[1, 36, 1, 1][36, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, 3);  unsqueeze_1116 = None
        mul_1188: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, 6.228077168367346e-07)
        squeeze_40: "f32[36][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_1189: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_1190: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1188, mul_1189);  mul_1188 = mul_1189 = None
        unsqueeze_1118: "f32[1, 36][36, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1190, 0);  mul_1190 = None
        unsqueeze_1119: "f32[1, 36, 1][36, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, 2);  unsqueeze_1118 = None
        unsqueeze_1120: "f32[1, 36, 1, 1][36, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1119, 3);  unsqueeze_1119 = None
        mul_1191: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, primals_84);  primals_84 = None
        unsqueeze_1121: "f32[1, 36][36, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1191, 0);  mul_1191 = None
        unsqueeze_1122: "f32[1, 36, 1][36, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1121, 2);  unsqueeze_1121 = None
        unsqueeze_1123: "f32[1, 36, 1, 1][36, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 3);  unsqueeze_1122 = None
        mul_1192: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.mul.Tensor(sub_344, unsqueeze_1120);  sub_344 = unsqueeze_1120 = None
        sub_346: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.sub.Tensor(where_42, mul_1192);  where_42 = mul_1192 = None
        sub_347: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.sub.Tensor(sub_346, unsqueeze_1117);  sub_346 = unsqueeze_1117 = None
        mul_1193: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.mul.Tensor(sub_347, unsqueeze_1123);  sub_347 = unsqueeze_1123 = None
        mul_1194: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, squeeze_40);  sum_157 = squeeze_40 = None
        convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1193, relu_5, primals_80, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 36, [True, True, False]);  mul_1193 = primals_80 = None
        getitem_403: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = convolution_backward_81[0]
        getitem_404: "f32[36, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_81[1];  convolution_backward_81 = None
        add_470: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.add.Tensor(slice_83, getitem_403);  slice_83 = getitem_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_36: "b8[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_43: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.where.self(le_36, full_default, add_470);  le_36 = add_470 = None
        sum_158: "f32[36][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
        sub_348: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1126);  convolution_12 = unsqueeze_1126 = None
        mul_1195: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.mul.Tensor(where_43, sub_348)
        sum_159: "f32[36][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1195, [0, 2, 3]);  mul_1195 = None
        mul_1196: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_158, 6.228077168367346e-07)
        unsqueeze_1127: "f32[1, 36][36, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1196, 0);  mul_1196 = None
        unsqueeze_1128: "f32[1, 36, 1][36, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1127, 2);  unsqueeze_1127 = None
        unsqueeze_1129: "f32[1, 36, 1, 1][36, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, 3);  unsqueeze_1128 = None
        mul_1197: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_159, 6.228077168367346e-07)
        mul_1198: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_1199: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1197, mul_1198);  mul_1197 = mul_1198 = None
        unsqueeze_1130: "f32[1, 36][36, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1199, 0);  mul_1199 = None
        unsqueeze_1131: "f32[1, 36, 1][36, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, 2);  unsqueeze_1130 = None
        unsqueeze_1132: "f32[1, 36, 1, 1][36, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1131, 3);  unsqueeze_1131 = None
        mul_1200: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, primals_78);  primals_78 = None
        unsqueeze_1133: "f32[1, 36][36, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1200, 0);  mul_1200 = None
        unsqueeze_1134: "f32[1, 36, 1][36, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1133, 2);  unsqueeze_1133 = None
        unsqueeze_1135: "f32[1, 36, 1, 1][36, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 3);  unsqueeze_1134 = None
        mul_1201: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_1132);  sub_348 = unsqueeze_1132 = None
        sub_350: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.sub.Tensor(where_43, mul_1201);  where_43 = mul_1201 = None
        sub_351: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.sub.Tensor(sub_350, unsqueeze_1129);  sub_350 = unsqueeze_1129 = None
        mul_1202: "f32[512, 36, 56, 56][112896, 1, 2016, 36]cuda:0" = torch.ops.aten.mul.Tensor(sub_351, unsqueeze_1135);  sub_351 = unsqueeze_1135 = None
        mul_1203: "f32[36][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_159, squeeze_37);  sum_159 = squeeze_37 = None
        convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1202, add_61, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1202 = add_61 = primals_74 = None
        getitem_406: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = convolution_backward_82[0]
        getitem_407: "f32[36, 24, 1, 1][24, 1, 24, 24]cuda:0" = convolution_backward_82[1];  convolution_backward_82 = None
        add_471: "f32[512, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(clone_14, getitem_406);  clone_14 = getitem_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        new_empty_strided_14: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.new_empty_strided.default(add_471, [512, 24, 56, 56], [75264, 1, 1344, 24])
        copy_29: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.copy.default(new_empty_strided_14, add_471);  new_empty_strided_14 = add_471 = None
        clone_15: "f32[512, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.clone.default(copy_29, memory_format = torch.contiguous_format)
        copy_30: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.copy.default(copy_29, clone_15);  copy_29 = None
        sum_160: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(clone_15, [0, 2, 3])
        sub_352: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1138);  convolution_11 = unsqueeze_1138 = None
        mul_1204: "f32[512, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(clone_15, sub_352)
        sum_161: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1204, [0, 2, 3]);  mul_1204 = None
        mul_1205: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_160, 6.228077168367346e-07)
        unsqueeze_1139: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1205, 0);  mul_1205 = None
        unsqueeze_1140: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1139, 2);  unsqueeze_1139 = None
        unsqueeze_1141: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, 3);  unsqueeze_1140 = None
        mul_1206: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_161, 6.228077168367346e-07)
        mul_1207: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_1208: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1206, mul_1207);  mul_1206 = mul_1207 = None
        unsqueeze_1142: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1208, 0);  mul_1208 = None
        unsqueeze_1143: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, 2);  unsqueeze_1142 = None
        unsqueeze_1144: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1143, 3);  unsqueeze_1143 = None
        mul_1209: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, primals_72);  primals_72 = None
        unsqueeze_1145: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1209, 0);  mul_1209 = None
        unsqueeze_1146: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1145, 2);  unsqueeze_1145 = None
        unsqueeze_1147: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, 3);  unsqueeze_1146 = None
        mul_1210: "f32[512, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_352, unsqueeze_1144);  sub_352 = unsqueeze_1144 = None
        sub_354: "f32[512, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_15, mul_1210);  clone_15 = mul_1210 = None
        sub_355: "f32[512, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_354, unsqueeze_1141);  sub_354 = unsqueeze_1141 = None
        mul_1211: "f32[512, 24, 56, 56][75264, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_355, unsqueeze_1147);  sub_355 = unsqueeze_1147 = None
        mul_1212: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_161, squeeze_34);  sum_161 = squeeze_34 = None
        convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_1211, add_55, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1211 = add_55 = primals_68 = None
        getitem_409: "f32[512, 16, 56, 56][50176, 1, 896, 16]cuda:0" = convolution_backward_83[0]
        getitem_410: "f32[24, 16, 1, 1][16, 1, 16, 16]cuda:0" = convolution_backward_83[1];  convolution_backward_83 = None
        sum_162: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_409, [0, 2, 3])
        sub_356: "f32[512, 16, 56, 56][50176, 1, 896, 16]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1150);  convolution_10 = unsqueeze_1150 = None
        mul_1213: "f32[512, 16, 56, 56][50176, 1, 896, 16]cuda:0" = torch.ops.aten.mul.Tensor(getitem_409, sub_356)
        sum_163: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1213, [0, 2, 3]);  mul_1213 = None
        mul_1214: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_162, 6.228077168367346e-07)
        unsqueeze_1151: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1214, 0);  mul_1214 = None
        unsqueeze_1152: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1151, 2);  unsqueeze_1151 = None
        unsqueeze_1153: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, 3);  unsqueeze_1152 = None
        mul_1215: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_163, 6.228077168367346e-07)
        mul_1216: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_1217: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1215, mul_1216);  mul_1215 = mul_1216 = None
        unsqueeze_1154: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1217, 0);  mul_1217 = None
        unsqueeze_1155: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, 2);  unsqueeze_1154 = None
        unsqueeze_1156: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1155, 3);  unsqueeze_1155 = None
        mul_1218: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, primals_66);  primals_66 = None
        unsqueeze_1157: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1218, 0);  mul_1218 = None
        unsqueeze_1158: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1157, 2);  unsqueeze_1157 = None
        unsqueeze_1159: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, 3);  unsqueeze_1158 = None
        mul_1219: "f32[512, 16, 56, 56][50176, 1, 896, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_356, unsqueeze_1156);  sub_356 = unsqueeze_1156 = None
        sub_358: "f32[512, 16, 56, 56][50176, 1, 896, 16]cuda:0" = torch.ops.aten.sub.Tensor(getitem_409, mul_1219);  getitem_409 = mul_1219 = None
        sub_359: "f32[512, 16, 56, 56][50176, 1, 896, 16]cuda:0" = torch.ops.aten.sub.Tensor(sub_358, unsqueeze_1153);  sub_358 = unsqueeze_1153 = None
        mul_1220: "f32[512, 16, 56, 56][50176, 1, 896, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_359, unsqueeze_1159);  sub_359 = unsqueeze_1159 = None
        mul_1221: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_163, squeeze_31);  sum_163 = squeeze_31 = None
        convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_1220, add_25, primals_62, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_1220 = primals_62 = None
        getitem_412: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = convolution_backward_84[0]
        getitem_413: "f32[16, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_84[1];  convolution_backward_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_87: "f32[512, 12, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.slice.Tensor(copy_30, 1, 12, 24)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_164: "f32[12][1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_87, [0, 2, 3])
        sub_360: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1162);  convolution_9 = unsqueeze_1162 = None
        mul_1222: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.mul.Tensor(slice_87, sub_360)
        sum_165: "f32[12][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1222, [0, 2, 3]);  mul_1222 = None
        mul_1223: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_164, 6.228077168367346e-07)
        unsqueeze_1163: "f32[1, 12][12, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1223, 0);  mul_1223 = None
        unsqueeze_1164: "f32[1, 12, 1][12, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1163, 2);  unsqueeze_1163 = None
        unsqueeze_1165: "f32[1, 12, 1, 1][12, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, 3);  unsqueeze_1164 = None
        mul_1224: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_165, 6.228077168367346e-07)
        mul_1225: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_1226: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1224, mul_1225);  mul_1224 = mul_1225 = None
        unsqueeze_1166: "f32[1, 12][12, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1226, 0);  mul_1226 = None
        unsqueeze_1167: "f32[1, 12, 1][12, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, 2);  unsqueeze_1166 = None
        unsqueeze_1168: "f32[1, 12, 1, 1][12, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1167, 3);  unsqueeze_1167 = None
        mul_1227: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, primals_60);  primals_60 = None
        unsqueeze_1169: "f32[1, 12][12, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1227, 0);  mul_1227 = None
        unsqueeze_1170: "f32[1, 12, 1][12, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1169, 2);  unsqueeze_1169 = None
        unsqueeze_1171: "f32[1, 12, 1, 1][12, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, 3);  unsqueeze_1170 = None
        mul_1228: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.mul.Tensor(sub_360, unsqueeze_1168);  sub_360 = unsqueeze_1168 = None
        sub_362: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.sub.Tensor(slice_87, mul_1228);  slice_87 = mul_1228 = None
        sub_363: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.sub.Tensor(sub_362, unsqueeze_1165);  sub_362 = unsqueeze_1165 = None
        mul_1229: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.mul.Tensor(sub_363, unsqueeze_1171);  sub_363 = unsqueeze_1171 = None
        mul_1230: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_165, squeeze_28);  sum_165 = squeeze_28 = None
        convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1229, add_45, primals_56, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 12, [True, True, False]);  mul_1229 = add_45 = primals_56 = None
        getitem_415: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = convolution_backward_85[0]
        getitem_416: "f32[12, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_85[1];  convolution_backward_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_88: "f32[512, 12, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.slice.Tensor(copy_30, 1, 0, 12);  copy_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_472: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.add.Tensor(slice_88, getitem_415);  slice_88 = getitem_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_166: "f32[12][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_472, [0, 2, 3])
        sub_364: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1174);  convolution_8 = unsqueeze_1174 = None
        mul_1231: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.mul.Tensor(add_472, sub_364)
        sum_167: "f32[12][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1231, [0, 2, 3]);  mul_1231 = None
        mul_1232: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_166, 6.228077168367346e-07)
        unsqueeze_1175: "f32[1, 12][12, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1232, 0);  mul_1232 = None
        unsqueeze_1176: "f32[1, 12, 1][12, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1175, 2);  unsqueeze_1175 = None
        unsqueeze_1177: "f32[1, 12, 1, 1][12, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, 3);  unsqueeze_1176 = None
        mul_1233: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_167, 6.228077168367346e-07)
        mul_1234: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_1235: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1233, mul_1234);  mul_1233 = mul_1234 = None
        unsqueeze_1178: "f32[1, 12][12, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1235, 0);  mul_1235 = None
        unsqueeze_1179: "f32[1, 12, 1][12, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, 2);  unsqueeze_1178 = None
        unsqueeze_1180: "f32[1, 12, 1, 1][12, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1179, 3);  unsqueeze_1179 = None
        mul_1236: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  primals_54 = None
        unsqueeze_1181: "f32[1, 12][12, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1236, 0);  mul_1236 = None
        unsqueeze_1182: "f32[1, 12, 1][12, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1181, 2);  unsqueeze_1181 = None
        unsqueeze_1183: "f32[1, 12, 1, 1][12, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 3);  unsqueeze_1182 = None
        mul_1237: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.mul.Tensor(sub_364, unsqueeze_1180);  sub_364 = unsqueeze_1180 = None
        sub_366: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.sub.Tensor(add_472, mul_1237);  add_472 = mul_1237 = None
        sub_367: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.sub.Tensor(sub_366, unsqueeze_1177);  sub_366 = unsqueeze_1177 = None
        mul_1238: "f32[512, 12, 56, 56][37632, 1, 672, 12]cuda:0" = torch.ops.aten.mul.Tensor(sub_367, unsqueeze_1183);  sub_367 = unsqueeze_1183 = None
        mul_1239: "f32[12][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_167, squeeze_25);  sum_167 = squeeze_25 = None
        convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1238, add_40, primals_50, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1238 = add_40 = primals_50 = None
        getitem_418: "f32[512, 48, 56, 56][150528, 1, 2688, 48]cuda:0" = convolution_backward_86[0]
        getitem_419: "f32[12, 48, 1, 1][48, 1, 48, 48]cuda:0" = convolution_backward_86[1];  convolution_backward_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        sum_168: "f32[48][1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_418, [0, 2, 3])
        sub_368: "f32[512, 48, 56, 56][150528, 1, 2688, 48]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1186);  convolution_7 = unsqueeze_1186 = None
        mul_1240: "f32[512, 48, 56, 56][150528, 1, 2688, 48]cuda:0" = torch.ops.aten.mul.Tensor(getitem_418, sub_368)
        sum_169: "f32[48][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1240, [0, 2, 3]);  mul_1240 = None
        mul_1241: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_168, 6.228077168367346e-07)
        unsqueeze_1187: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1241, 0);  mul_1241 = None
        unsqueeze_1188: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1187, 2);  unsqueeze_1187 = None
        unsqueeze_1189: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, 3);  unsqueeze_1188 = None
        mul_1242: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, 6.228077168367346e-07)
        mul_1243: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_1244: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1242, mul_1243);  mul_1242 = mul_1243 = None
        unsqueeze_1190: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1244, 0);  mul_1244 = None
        unsqueeze_1191: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, 2);  unsqueeze_1190 = None
        unsqueeze_1192: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1191, 3);  unsqueeze_1191 = None
        mul_1245: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  primals_48 = None
        unsqueeze_1193: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1245, 0);  mul_1245 = None
        unsqueeze_1194: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1193, 2);  unsqueeze_1193 = None
        unsqueeze_1195: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 3);  unsqueeze_1194 = None
        mul_1246: "f32[512, 48, 56, 56][150528, 1, 2688, 48]cuda:0" = torch.ops.aten.mul.Tensor(sub_368, unsqueeze_1192);  sub_368 = unsqueeze_1192 = None
        sub_370: "f32[512, 48, 56, 56][150528, 1, 2688, 48]cuda:0" = torch.ops.aten.sub.Tensor(getitem_418, mul_1246);  getitem_418 = mul_1246 = None
        sub_371: "f32[512, 48, 56, 56][150528, 1, 2688, 48]cuda:0" = torch.ops.aten.sub.Tensor(sub_370, unsqueeze_1189);  sub_370 = unsqueeze_1189 = None
        mul_1247: "f32[512, 48, 56, 56][150528, 1, 2688, 48]cuda:0" = torch.ops.aten.mul.Tensor(sub_371, unsqueeze_1195);  sub_371 = unsqueeze_1195 = None
        mul_1248: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, squeeze_22);  sum_169 = squeeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:435 in forward, code: x = self.conv_dw(x)
        convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_1247, cat_2, primals_44, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 48, [True, True, False]);  mul_1247 = cat_2 = primals_44 = None
        getitem_421: "f32[512, 48, 112, 112][602112, 1, 5376, 48]cuda:0" = convolution_backward_87[0]
        getitem_422: "f32[48, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_87[1];  convolution_backward_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_89: "f32[512, 24, 112, 112][602112, 1, 5376, 48]cuda:0" = torch.ops.aten.slice.Tensor(getitem_421, 1, 0, 24)
        slice_90: "f32[512, 24, 112, 112][602112, 1, 5376, 48]cuda:0" = torch.ops.aten.slice.Tensor(getitem_421, 1, 24, 48);  getitem_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_6: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
        mul_42: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        unsqueeze_24: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_25: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_48: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        unsqueeze_26: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_27: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_35: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
        relu_4: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.relu.default(add_35);  add_35 = None
        le_37: "b8[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_44: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.where.self(le_37, full_default, slice_90);  le_37 = slice_90 = None
        squeeze_18: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        unsqueeze_1196: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_1197: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, 2);  unsqueeze_1196 = None
        unsqueeze_1198: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1197, 3);  unsqueeze_1197 = None
        sum_170: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
        sub_372: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1198);  convolution_6 = unsqueeze_1198 = None
        mul_1249: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.mul.Tensor(where_44, sub_372)
        sum_171: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1249, [0, 2, 3]);  mul_1249 = None
        mul_1250: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_170, 1.5570192920918366e-07)
        unsqueeze_1199: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1250, 0);  mul_1250 = None
        unsqueeze_1200: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1199, 2);  unsqueeze_1199 = None
        unsqueeze_1201: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, 3);  unsqueeze_1200 = None
        mul_1251: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_171, 1.5570192920918366e-07)
        squeeze_19: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_1252: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_1253: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1251, mul_1252);  mul_1251 = mul_1252 = None
        unsqueeze_1202: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1253, 0);  mul_1253 = None
        unsqueeze_1203: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, 2);  unsqueeze_1202 = None
        unsqueeze_1204: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1203, 3);  unsqueeze_1203 = None
        mul_1254: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, primals_42);  primals_42 = None
        unsqueeze_1205: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1254, 0);  mul_1254 = None
        unsqueeze_1206: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1205, 2);  unsqueeze_1205 = None
        unsqueeze_1207: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 3);  unsqueeze_1206 = None
        mul_1255: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_372, unsqueeze_1204);  sub_372 = unsqueeze_1204 = None
        sub_374: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.sub.Tensor(where_44, mul_1255);  where_44 = mul_1255 = None
        sub_375: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.sub.Tensor(sub_374, unsqueeze_1201);  sub_374 = unsqueeze_1201 = None
        mul_1256: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_375, unsqueeze_1207);  sub_375 = unsqueeze_1207 = None
        mul_1257: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_171, squeeze_19);  sum_171 = squeeze_19 = None
        convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_1256, relu_3, primals_38, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 24, [True, True, False]);  mul_1256 = primals_38 = None
        getitem_424: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = convolution_backward_88[0]
        getitem_425: "f32[24, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_88[1];  convolution_backward_88 = None
        add_473: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.add.Tensor(slice_89, getitem_424);  slice_89 = getitem_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_38: "b8[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_45: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.where.self(le_38, full_default, add_473);  le_38 = add_473 = None
        sum_172: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
        sub_376: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1210);  convolution_5 = unsqueeze_1210 = None
        mul_1258: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.mul.Tensor(where_45, sub_376)
        sum_173: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1258, [0, 2, 3]);  mul_1258 = None
        mul_1259: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_172, 1.5570192920918366e-07)
        unsqueeze_1211: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1259, 0);  mul_1259 = None
        unsqueeze_1212: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1211, 2);  unsqueeze_1211 = None
        unsqueeze_1213: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, 3);  unsqueeze_1212 = None
        mul_1260: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_173, 1.5570192920918366e-07)
        mul_1261: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_1262: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1260, mul_1261);  mul_1260 = mul_1261 = None
        unsqueeze_1214: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1262, 0);  mul_1262 = None
        unsqueeze_1215: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, 2);  unsqueeze_1214 = None
        unsqueeze_1216: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1215, 3);  unsqueeze_1215 = None
        mul_1263: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  primals_36 = None
        unsqueeze_1217: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1263, 0);  mul_1263 = None
        unsqueeze_1218: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1217, 2);  unsqueeze_1217 = None
        unsqueeze_1219: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 3);  unsqueeze_1218 = None
        mul_1264: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_376, unsqueeze_1216);  sub_376 = unsqueeze_1216 = None
        sub_378: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.sub.Tensor(where_45, mul_1264);  where_45 = mul_1264 = None
        sub_379: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.sub.Tensor(sub_378, unsqueeze_1213);  sub_378 = unsqueeze_1213 = None
        mul_1265: "f32[512, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_379, unsqueeze_1219);  sub_379 = unsqueeze_1219 = None
        mul_1266: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_173, squeeze_16);  sum_173 = squeeze_16 = None
        convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_1265, add_25, primals_32, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1265 = add_25 = primals_32 = None
        getitem_427: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = convolution_backward_89[0]
        getitem_428: "f32[24, 16, 1, 1][16, 1, 16, 16]cuda:0" = convolution_backward_89[1];  convolution_backward_89 = None
        add_474: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.add.Tensor(getitem_412, getitem_427);  getitem_412 = getitem_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        clone_16: "f32[512, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.clone.default(add_474, memory_format = torch.contiguous_format)
        copy_32: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.copy.default(add_474, clone_16);  add_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_93: "f32[512, 8, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.slice.Tensor(copy_32, 1, 8, 16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sum_174: "f32[8][1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_93, [0, 2, 3])
        sub_380: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1222);  convolution_4 = unsqueeze_1222 = None
        mul_1267: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.mul.Tensor(slice_93, sub_380)
        sum_175: "f32[8][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1267, [0, 2, 3]);  mul_1267 = None
        mul_1268: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_174, 1.5570192920918366e-07)
        unsqueeze_1223: "f32[1, 8][8, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1268, 0);  mul_1268 = None
        unsqueeze_1224: "f32[1, 8, 1][8, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1223, 2);  unsqueeze_1223 = None
        unsqueeze_1225: "f32[1, 8, 1, 1][8, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, 3);  unsqueeze_1224 = None
        mul_1269: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, 1.5570192920918366e-07)
        mul_1270: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_1271: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1269, mul_1270);  mul_1269 = mul_1270 = None
        unsqueeze_1226: "f32[1, 8][8, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1271, 0);  mul_1271 = None
        unsqueeze_1227: "f32[1, 8, 1][8, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, 2);  unsqueeze_1226 = None
        unsqueeze_1228: "f32[1, 8, 1, 1][8, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1227, 3);  unsqueeze_1227 = None
        mul_1272: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_1229: "f32[1, 8][8, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1272, 0);  mul_1272 = None
        unsqueeze_1230: "f32[1, 8, 1][8, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1229, 2);  unsqueeze_1229 = None
        unsqueeze_1231: "f32[1, 8, 1, 1][8, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, 3);  unsqueeze_1230 = None
        mul_1273: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.mul.Tensor(sub_380, unsqueeze_1228);  sub_380 = unsqueeze_1228 = None
        sub_382: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.sub.Tensor(slice_93, mul_1273);  slice_93 = mul_1273 = None
        sub_383: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.sub.Tensor(sub_382, unsqueeze_1225);  sub_382 = unsqueeze_1225 = None
        mul_1274: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.mul.Tensor(sub_383, unsqueeze_1231);  sub_383 = unsqueeze_1231 = None
        mul_1275: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, squeeze_13);  sum_175 = squeeze_13 = None
        convolution_backward_90 = torch.ops.aten.convolution_backward.default(mul_1274, add_19, primals_26, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1274 = add_19 = primals_26 = None
        getitem_430: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = convolution_backward_90[0]
        getitem_431: "f32[8, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_90[1];  convolution_backward_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_94: "f32[512, 8, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.slice.Tensor(copy_32, 1, 0, 8);  copy_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_475: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.add.Tensor(slice_94, getitem_430);  slice_94 = getitem_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        sum_176: "f32[8][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_475, [0, 2, 3])
        sub_384: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1234);  convolution_3 = unsqueeze_1234 = None
        mul_1276: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.mul.Tensor(add_475, sub_384)
        sum_177: "f32[8][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1276, [0, 2, 3]);  mul_1276 = None
        mul_1277: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_176, 1.5570192920918366e-07)
        unsqueeze_1235: "f32[1, 8][8, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1277, 0);  mul_1277 = None
        unsqueeze_1236: "f32[1, 8, 1][8, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1235, 2);  unsqueeze_1235 = None
        unsqueeze_1237: "f32[1, 8, 1, 1][8, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, 3);  unsqueeze_1236 = None
        mul_1278: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_177, 1.5570192920918366e-07)
        mul_1279: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_1280: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1278, mul_1279);  mul_1278 = mul_1279 = None
        unsqueeze_1238: "f32[1, 8][8, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1280, 0);  mul_1280 = None
        unsqueeze_1239: "f32[1, 8, 1][8, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, 2);  unsqueeze_1238 = None
        unsqueeze_1240: "f32[1, 8, 1, 1][8, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1239, 3);  unsqueeze_1239 = None
        mul_1281: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_1241: "f32[1, 8][8, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1281, 0);  mul_1281 = None
        unsqueeze_1242: "f32[1, 8, 1][8, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1241, 2);  unsqueeze_1241 = None
        unsqueeze_1243: "f32[1, 8, 1, 1][8, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, 3);  unsqueeze_1242 = None
        mul_1282: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.mul.Tensor(sub_384, unsqueeze_1240);  sub_384 = unsqueeze_1240 = None
        sub_386: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.sub.Tensor(add_475, mul_1282);  add_475 = mul_1282 = None
        sub_387: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.sub.Tensor(sub_386, unsqueeze_1237);  sub_386 = unsqueeze_1237 = None
        mul_1283: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.mul.Tensor(sub_387, unsqueeze_1243);  sub_387 = unsqueeze_1243 = None
        mul_1284: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_177, squeeze_10);  sum_177 = squeeze_10 = None
        convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1283, cat, primals_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1283 = cat = primals_20 = None
        getitem_433: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = convolution_backward_91[0]
        getitem_434: "f32[8, 16, 1, 1][16, 1, 16, 16]cuda:0" = convolution_backward_91[1];  convolution_backward_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        slice_95: "f32[512, 8, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.slice.Tensor(getitem_433, 1, 0, 8)
        slice_96: "f32[512, 8, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.slice.Tensor(getitem_433, 1, 8, 16);  getitem_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        sub_2: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_14: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        unsqueeze_8: "f32[8, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[8, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[8, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_11: "f32[8, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        relu_2: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.relu.default(add_14);  add_14 = None
        le_39: "b8[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_46: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.where.self(le_39, full_default, slice_96);  le_39 = slice_96 = None
        squeeze_6: "f32[8][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        unsqueeze_1244: "f32[1, 8][8, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_1245: "f32[1, 8, 1][8, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, 2);  unsqueeze_1244 = None
        unsqueeze_1246: "f32[1, 8, 1, 1][8, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1245, 3);  unsqueeze_1245 = None
        sum_178: "f32[8][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
        sub_388: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1246);  convolution_2 = unsqueeze_1246 = None
        mul_1285: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.mul.Tensor(where_46, sub_388)
        sum_179: "f32[8][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1285, [0, 2, 3]);  mul_1285 = None
        mul_1286: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_178, 1.5570192920918366e-07)
        unsqueeze_1247: "f32[1, 8][8, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1286, 0);  mul_1286 = None
        unsqueeze_1248: "f32[1, 8, 1][8, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1247, 2);  unsqueeze_1247 = None
        unsqueeze_1249: "f32[1, 8, 1, 1][8, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, 3);  unsqueeze_1248 = None
        mul_1287: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_179, 1.5570192920918366e-07)
        squeeze_7: "f32[8][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_1288: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_1289: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1287, mul_1288);  mul_1287 = mul_1288 = None
        unsqueeze_1250: "f32[1, 8][8, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1289, 0);  mul_1289 = None
        unsqueeze_1251: "f32[1, 8, 1][8, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, 2);  unsqueeze_1250 = None
        unsqueeze_1252: "f32[1, 8, 1, 1][8, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1251, 3);  unsqueeze_1251 = None
        mul_1290: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_1253: "f32[1, 8][8, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1290, 0);  mul_1290 = None
        unsqueeze_1254: "f32[1, 8, 1][8, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1253, 2);  unsqueeze_1253 = None
        unsqueeze_1255: "f32[1, 8, 1, 1][8, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 3);  unsqueeze_1254 = None
        mul_1291: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.mul.Tensor(sub_388, unsqueeze_1252);  sub_388 = unsqueeze_1252 = None
        sub_390: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.sub.Tensor(where_46, mul_1291);  where_46 = mul_1291 = None
        sub_391: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.sub.Tensor(sub_390, unsqueeze_1249);  sub_390 = unsqueeze_1249 = None
        mul_1292: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.mul.Tensor(sub_391, unsqueeze_1255);  sub_391 = unsqueeze_1255 = None
        mul_1293: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_179, squeeze_7);  sum_179 = squeeze_7 = None
        convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_1292, relu_1, primals_14, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_1292 = primals_14 = None
        getitem_436: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = convolution_backward_92[0]
        getitem_437: "f32[8, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_92[1];  convolution_backward_92 = None
        add_476: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.add.Tensor(slice_95, getitem_436);  slice_95 = getitem_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        le_40: "b8[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_47: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.where.self(le_40, full_default, add_476);  le_40 = add_476 = None
        sum_180: "f32[8][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
        sub_392: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1258);  convolution_1 = unsqueeze_1258 = None
        mul_1294: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.mul.Tensor(where_47, sub_392)
        sum_181: "f32[8][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1294, [0, 2, 3]);  mul_1294 = None
        mul_1295: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_180, 1.5570192920918366e-07)
        unsqueeze_1259: "f32[1, 8][8, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1295, 0);  mul_1295 = None
        unsqueeze_1260: "f32[1, 8, 1][8, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1259, 2);  unsqueeze_1259 = None
        unsqueeze_1261: "f32[1, 8, 1, 1][8, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, 3);  unsqueeze_1260 = None
        mul_1296: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_181, 1.5570192920918366e-07)
        mul_1297: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_1298: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1296, mul_1297);  mul_1296 = mul_1297 = None
        unsqueeze_1262: "f32[1, 8][8, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1298, 0);  mul_1298 = None
        unsqueeze_1263: "f32[1, 8, 1][8, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, 2);  unsqueeze_1262 = None
        unsqueeze_1264: "f32[1, 8, 1, 1][8, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1263, 3);  unsqueeze_1263 = None
        mul_1299: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_1265: "f32[1, 8][8, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1299, 0);  mul_1299 = None
        unsqueeze_1266: "f32[1, 8, 1][8, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1265, 2);  unsqueeze_1265 = None
        unsqueeze_1267: "f32[1, 8, 1, 1][8, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 3);  unsqueeze_1266 = None
        mul_1300: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.mul.Tensor(sub_392, unsqueeze_1264);  sub_392 = unsqueeze_1264 = None
        sub_394: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.sub.Tensor(where_47, mul_1300);  where_47 = mul_1300 = None
        sub_395: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.sub.Tensor(sub_394, unsqueeze_1261);  sub_394 = unsqueeze_1261 = None
        mul_1301: "f32[512, 8, 112, 112][100352, 1, 896, 8]cuda:0" = torch.ops.aten.mul.Tensor(sub_395, unsqueeze_1267);  sub_395 = unsqueeze_1267 = None
        mul_1302: "f32[8][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_181, squeeze_4);  sum_181 = squeeze_4 = None
        convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_1301, relu, primals_8, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1301 = primals_8 = None
        getitem_439: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = convolution_backward_93[0]
        getitem_440: "f32[8, 16, 1, 1][16, 1, 16, 16]cuda:0" = convolution_backward_93[1];  convolution_backward_93 = None
        add_477: "f32[512, 16, 112, 112][200704, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(clone_16, getitem_439);  clone_16 = getitem_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:823 in forward_features, code: x = self.act1(x)
        le_41: "b8[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_48: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.where.self(le_41, full_default, add_477);  le_41 = full_default = add_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:822 in forward_features, code: x = self.bn1(x)
        sum_182: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
        sub_396: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1270);  convolution = unsqueeze_1270 = None
        mul_1303: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(where_48, sub_396)
        sum_183: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1303, [0, 2, 3]);  mul_1303 = None
        mul_1304: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_182, 1.5570192920918366e-07)
        unsqueeze_1271: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1304, 0);  mul_1304 = None
        unsqueeze_1272: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1271, 2);  unsqueeze_1271 = None
        unsqueeze_1273: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, 3);  unsqueeze_1272 = None
        mul_1305: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_183, 1.5570192920918366e-07)
        mul_1306: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_1307: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1305, mul_1306);  mul_1305 = mul_1306 = None
        unsqueeze_1274: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1307, 0);  mul_1307 = None
        unsqueeze_1275: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, 2);  unsqueeze_1274 = None
        unsqueeze_1276: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1275, 3);  unsqueeze_1275 = None
        mul_1308: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_1277: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1308, 0);  mul_1308 = None
        unsqueeze_1278: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1277, 2);  unsqueeze_1277 = None
        unsqueeze_1279: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 3);  unsqueeze_1278 = None
        mul_1309: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_396, unsqueeze_1276);  sub_396 = unsqueeze_1276 = None
        sub_398: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(where_48, mul_1309);  where_48 = mul_1309 = None
        sub_399: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(sub_398, unsqueeze_1273);  sub_398 = unsqueeze_1273 = None
        mul_1310: "f32[512, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_399, unsqueeze_1279);  sub_399 = unsqueeze_1279 = None
        mul_1311: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_183, squeeze_1);  sum_183 = squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:821 in forward_features, code: x = self.conv_stem(x)
        convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_1310, primals_2, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1310 = primals_2 = primals_1 = None
        getitem_443: "f32[16, 3, 3, 3][27, 1, 9, 3]cuda:0" = convolution_backward_94[1];  convolution_backward_94 = None
        return (getitem_443, None, None, None, None, mul_1311, sum_182, getitem_440, None, None, None, mul_1302, sum_180, getitem_437, None, None, None, mul_1293, sum_178, getitem_434, None, None, None, mul_1284, sum_176, getitem_431, None, None, None, mul_1275, sum_174, getitem_428, None, None, None, mul_1266, sum_172, getitem_425, None, None, None, mul_1257, sum_170, getitem_422, None, None, None, mul_1248, sum_168, getitem_419, None, None, None, mul_1239, sum_166, getitem_416, None, None, None, mul_1230, sum_164, getitem_413, None, None, None, mul_1221, sum_162, getitem_410, None, None, None, mul_1212, sum_160, getitem_407, None, None, None, mul_1203, sum_158, getitem_404, None, None, None, mul_1194, sum_156, getitem_401, None, None, None, mul_1185, sum_154, getitem_398, None, None, None, mul_1176, sum_152, getitem_395, None, None, None, mul_1167, sum_150, getitem_392, None, None, None, mul_1158, sum_148, getitem_389, None, None, None, mul_1149, sum_146, getitem_386, sum_145, getitem_383, sum_144, getitem_380, None, None, None, mul_1137, sum_141, getitem_377, None, None, None, mul_1128, sum_139, getitem_374, None, None, None, mul_1119, sum_137, getitem_371, None, None, None, mul_1110, sum_135, getitem_368, None, None, None, mul_1101, sum_133, getitem_365, None, None, None, mul_1092, sum_131, getitem_362, sum_130, getitem_359, sum_129, getitem_356, None, None, None, mul_1080, sum_126, getitem_353, None, None, None, mul_1071, sum_124, getitem_350, None, None, None, mul_1062, sum_122, getitem_347, None, None, None, mul_1053, sum_120, getitem_344, None, None, None, mul_1044, sum_118, getitem_341, None, None, None, mul_1035, sum_116, getitem_338, None, None, None, mul_1026, sum_114, getitem_335, None, None, None, mul_1017, sum_112, getitem_332, None, None, None, mul_1008, sum_110, getitem_329, None, None, None, mul_999, sum_108, getitem_326, None, None, None, mul_990, sum_106, getitem_323, None, None, None, mul_981, sum_104, getitem_320, None, None, None, mul_972, sum_102, getitem_317, None, None, None, mul_963, sum_100, getitem_314, None, None, None, mul_954, sum_98, getitem_311, None, None, None, mul_945, sum_96, getitem_308, None, None, None, mul_936, sum_94, getitem_305, None, None, None, mul_927, sum_92, getitem_302, None, None, None, mul_918, sum_90, getitem_299, None, None, None, mul_909, sum_88, getitem_296, None, None, None, mul_900, sum_86, getitem_293, None, None, None, mul_891, sum_84, getitem_290, None, None, None, mul_882, sum_82, getitem_287, sum_81, getitem_284, sum_80, getitem_281, None, None, None, mul_870, sum_77, getitem_278, None, None, None, mul_861, sum_75, getitem_275, None, None, None, mul_852, sum_73, getitem_272, None, None, None, mul_843, sum_71, getitem_269, None, None, None, mul_834, sum_69, getitem_266, None, None, None, mul_825, sum_67, getitem_263, sum_66, getitem_260, sum_65, getitem_257, None, None, None, mul_813, sum_62, getitem_254, None, None, None, mul_804, sum_60, getitem_251, None, None, None, mul_795, sum_58, getitem_248, None, None, None, mul_786, sum_56, getitem_245, None, None, None, mul_777, sum_54, getitem_242, sum_53, getitem_239, sum_52, getitem_236, None, None, None, mul_765, sum_49, getitem_233, None, None, None, mul_756, sum_47, getitem_230, None, None, None, mul_747, sum_45, getitem_227, None, None, None, mul_738, sum_43, getitem_224, None, None, None, mul_729, sum_41, getitem_221, None, None, None, mul_720, sum_39, getitem_218, None, None, None, mul_711, sum_37, getitem_215, None, None, None, mul_702, sum_35, getitem_212, None, None, None, mul_693, sum_33, getitem_209, None, None, None, mul_684, sum_31, getitem_206, sum_30, getitem_203, sum_29, getitem_200, None, None, None, mul_672, sum_26, getitem_197, None, None, None, mul_663, sum_24, getitem_194, None, None, None, mul_654, sum_22, getitem_191, None, None, None, mul_645, sum_20, getitem_188, None, None, None, mul_636, sum_18, getitem_185, None, None, None, mul_627, sum_16, getitem_182, None, None, None, mul_618, sum_14, getitem_179, None, None, None, mul_609, sum_12, getitem_176, sum_11, getitem_173, sum_10, getitem_170, None, None, None, mul_597, sum_7, getitem_167, None, None, None, mul_588, sum_5, getitem_164, None, None, None, mul_579, sum_3, getitem_161, sum_2, mm_1, view_2)
