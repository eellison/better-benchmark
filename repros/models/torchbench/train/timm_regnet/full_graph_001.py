import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 3, 3, 3]", primals_2: "f32[32, 3, 224, 224]", primals_6: "f32[32]", primals_8: "f32[224, 32, 1, 1]", primals_12: "f32[224]", primals_14: "f32[224, 112, 3, 3]", primals_18: "f32[224]", primals_19: "f32[224]", primals_20: "f32[8, 224, 1, 1]", primals_22: "f32[224, 8, 1, 1]", primals_24: "f32[224, 224, 1, 1]", primals_28: "f32[224]", primals_30: "f32[224, 32, 1, 1]", primals_34: "f32[224]", primals_36: "f32[224, 224, 1, 1]", primals_40: "f32[224]", primals_42: "f32[224, 112, 3, 3]", primals_46: "f32[224]", primals_47: "f32[224]", primals_48: "f32[56, 224, 1, 1]", primals_50: "f32[224, 56, 1, 1]", primals_52: "f32[224, 224, 1, 1]", primals_56: "f32[224]", primals_58: "f32[448, 224, 1, 1]", primals_62: "f32[448]", primals_64: "f32[448, 112, 3, 3]", primals_68: "f32[448]", primals_69: "f32[448]", primals_70: "f32[56, 448, 1, 1]", primals_72: "f32[448, 56, 1, 1]", primals_74: "f32[448, 448, 1, 1]", primals_78: "f32[448]", primals_80: "f32[448, 224, 1, 1]", primals_84: "f32[448]", primals_86: "f32[448, 448, 1, 1]", primals_90: "f32[448]", primals_92: "f32[448, 112, 3, 3]", primals_96: "f32[448]", primals_97: "f32[448]", primals_98: "f32[112, 448, 1, 1]", primals_100: "f32[448, 112, 1, 1]", primals_102: "f32[448, 448, 1, 1]", primals_106: "f32[448]", primals_108: "f32[448, 448, 1, 1]", primals_112: "f32[448]", primals_114: "f32[448, 112, 3, 3]", primals_118: "f32[448]", primals_119: "f32[448]", primals_120: "f32[112, 448, 1, 1]", primals_122: "f32[448, 112, 1, 1]", primals_124: "f32[448, 448, 1, 1]", primals_128: "f32[448]", primals_130: "f32[448, 448, 1, 1]", primals_134: "f32[448]", primals_136: "f32[448, 112, 3, 3]", primals_140: "f32[448]", primals_141: "f32[448]", primals_142: "f32[112, 448, 1, 1]", primals_144: "f32[448, 112, 1, 1]", primals_146: "f32[448, 448, 1, 1]", primals_150: "f32[448]", primals_152: "f32[448, 448, 1, 1]", primals_156: "f32[448]", primals_158: "f32[448, 112, 3, 3]", primals_162: "f32[448]", primals_163: "f32[448]", primals_164: "f32[112, 448, 1, 1]", primals_166: "f32[448, 112, 1, 1]", primals_168: "f32[448, 448, 1, 1]", primals_172: "f32[448]", primals_174: "f32[896, 448, 1, 1]", primals_178: "f32[896]", primals_180: "f32[896, 112, 3, 3]", primals_184: "f32[896]", primals_185: "f32[896]", primals_186: "f32[112, 896, 1, 1]", primals_188: "f32[896, 112, 1, 1]", primals_190: "f32[896, 896, 1, 1]", primals_194: "f32[896]", primals_196: "f32[896, 448, 1, 1]", primals_200: "f32[896]", primals_202: "f32[896, 896, 1, 1]", primals_206: "f32[896]", primals_208: "f32[896, 112, 3, 3]", primals_212: "f32[896]", primals_213: "f32[896]", primals_214: "f32[224, 896, 1, 1]", primals_216: "f32[896, 224, 1, 1]", primals_218: "f32[896, 896, 1, 1]", primals_222: "f32[896]", primals_224: "f32[896, 896, 1, 1]", primals_228: "f32[896]", primals_230: "f32[896, 112, 3, 3]", primals_234: "f32[896]", primals_235: "f32[896]", primals_236: "f32[224, 896, 1, 1]", primals_238: "f32[896, 224, 1, 1]", primals_240: "f32[896, 896, 1, 1]", primals_244: "f32[896]", primals_246: "f32[896, 896, 1, 1]", primals_250: "f32[896]", primals_252: "f32[896, 112, 3, 3]", primals_256: "f32[896]", primals_257: "f32[896]", primals_258: "f32[224, 896, 1, 1]", primals_260: "f32[896, 224, 1, 1]", primals_262: "f32[896, 896, 1, 1]", primals_266: "f32[896]", primals_268: "f32[896, 896, 1, 1]", primals_272: "f32[896]", primals_274: "f32[896, 112, 3, 3]", primals_278: "f32[896]", primals_279: "f32[896]", primals_280: "f32[224, 896, 1, 1]", primals_282: "f32[896, 224, 1, 1]", primals_284: "f32[896, 896, 1, 1]", primals_288: "f32[896]", primals_290: "f32[896, 896, 1, 1]", primals_294: "f32[896]", primals_296: "f32[896, 112, 3, 3]", primals_300: "f32[896]", primals_301: "f32[896]", primals_302: "f32[224, 896, 1, 1]", primals_304: "f32[896, 224, 1, 1]", primals_306: "f32[896, 896, 1, 1]", primals_310: "f32[896]", primals_312: "f32[896, 896, 1, 1]", primals_316: "f32[896]", primals_318: "f32[896, 112, 3, 3]", primals_322: "f32[896]", primals_323: "f32[896]", primals_324: "f32[224, 896, 1, 1]", primals_326: "f32[896, 224, 1, 1]", primals_328: "f32[896, 896, 1, 1]", primals_332: "f32[896]", primals_334: "f32[896, 896, 1, 1]", primals_338: "f32[896]", primals_340: "f32[896, 112, 3, 3]", primals_344: "f32[896]", primals_345: "f32[896]", primals_346: "f32[224, 896, 1, 1]", primals_348: "f32[896, 224, 1, 1]", primals_350: "f32[896, 896, 1, 1]", primals_354: "f32[896]", primals_356: "f32[896, 896, 1, 1]", primals_360: "f32[896]", primals_362: "f32[896, 112, 3, 3]", primals_366: "f32[896]", primals_367: "f32[896]", primals_368: "f32[224, 896, 1, 1]", primals_370: "f32[896, 224, 1, 1]", primals_372: "f32[896, 896, 1, 1]", primals_376: "f32[896]", primals_378: "f32[896, 896, 1, 1]", primals_382: "f32[896]", primals_384: "f32[896, 112, 3, 3]", primals_388: "f32[896]", primals_389: "f32[896]", primals_390: "f32[224, 896, 1, 1]", primals_392: "f32[896, 224, 1, 1]", primals_394: "f32[896, 896, 1, 1]", primals_398: "f32[896]", primals_400: "f32[896, 896, 1, 1]", primals_404: "f32[896]", primals_406: "f32[896, 112, 3, 3]", primals_410: "f32[896]", primals_411: "f32[896]", primals_412: "f32[224, 896, 1, 1]", primals_414: "f32[896, 224, 1, 1]", primals_416: "f32[896, 896, 1, 1]", primals_420: "f32[896]", primals_422: "f32[2240, 896, 1, 1]", primals_426: "f32[2240]", primals_428: "f32[2240, 112, 3, 3]", primals_432: "f32[2240]", primals_433: "f32[2240]", primals_434: "f32[224, 2240, 1, 1]", primals_436: "f32[2240, 224, 1, 1]", primals_438: "f32[2240, 2240, 1, 1]", primals_442: "f32[2240]", primals_443: "f32[2240]", primals_444: "f32[2240, 896, 1, 1]", primals_448: "f32[2240]", primals_449: "f32[2240]", primals_450: "f32[1000, 2240]", convolution: "f32[32, 32, 112, 112]", squeeze_1: "f32[32]", relu: "f32[32, 32, 112, 112]", convolution_1: "f32[32, 224, 112, 112]", squeeze_4: "f32[224]", relu_1: "f32[32, 224, 112, 112]", convolution_2: "f32[32, 224, 56, 56]", getitem_5: "f32[1, 224, 1, 1]", rsqrt_2: "f32[1, 224, 1, 1]", mean: "f32[32, 224, 1, 1]", relu_3: "f32[32, 8, 1, 1]", convolution_4: "f32[32, 224, 1, 1]", mul_21: "f32[32, 224, 56, 56]", convolution_5: "f32[32, 224, 56, 56]", squeeze_10: "f32[224]", convolution_6: "f32[32, 224, 56, 56]", squeeze_13: "f32[224]", relu_4: "f32[32, 224, 56, 56]", convolution_7: "f32[32, 224, 56, 56]", squeeze_16: "f32[224]", relu_5: "f32[32, 224, 56, 56]", convolution_8: "f32[32, 224, 56, 56]", getitem_13: "f32[1, 224, 1, 1]", rsqrt_6: "f32[1, 224, 1, 1]", mean_1: "f32[32, 224, 1, 1]", relu_7: "f32[32, 56, 1, 1]", convolution_10: "f32[32, 224, 1, 1]", mul_50: "f32[32, 224, 56, 56]", convolution_11: "f32[32, 224, 56, 56]", squeeze_22: "f32[224]", relu_8: "f32[32, 224, 56, 56]", convolution_12: "f32[32, 448, 56, 56]", squeeze_25: "f32[448]", relu_9: "f32[32, 448, 56, 56]", convolution_13: "f32[32, 448, 28, 28]", getitem_19: "f32[1, 448, 1, 1]", rsqrt_9: "f32[1, 448, 1, 1]", mean_2: "f32[32, 448, 1, 1]", relu_11: "f32[32, 56, 1, 1]", convolution_15: "f32[32, 448, 1, 1]", mul_72: "f32[32, 448, 28, 28]", convolution_16: "f32[32, 448, 28, 28]", squeeze_31: "f32[448]", convolution_17: "f32[32, 448, 28, 28]", squeeze_34: "f32[448]", relu_12: "f32[32, 448, 28, 28]", convolution_18: "f32[32, 448, 28, 28]", squeeze_37: "f32[448]", relu_13: "f32[32, 448, 28, 28]", convolution_19: "f32[32, 448, 28, 28]", getitem_27: "f32[1, 448, 1, 1]", rsqrt_13: "f32[1, 448, 1, 1]", mean_3: "f32[32, 448, 1, 1]", relu_15: "f32[32, 112, 1, 1]", convolution_21: "f32[32, 448, 1, 1]", mul_101: "f32[32, 448, 28, 28]", convolution_22: "f32[32, 448, 28, 28]", squeeze_43: "f32[448]", relu_16: "f32[32, 448, 28, 28]", convolution_23: "f32[32, 448, 28, 28]", squeeze_46: "f32[448]", relu_17: "f32[32, 448, 28, 28]", convolution_24: "f32[32, 448, 28, 28]", getitem_33: "f32[1, 448, 1, 1]", rsqrt_16: "f32[1, 448, 1, 1]", mean_4: "f32[32, 448, 1, 1]", relu_19: "f32[32, 112, 1, 1]", convolution_26: "f32[32, 448, 1, 1]", mul_123: "f32[32, 448, 28, 28]", convolution_27: "f32[32, 448, 28, 28]", squeeze_52: "f32[448]", relu_20: "f32[32, 448, 28, 28]", convolution_28: "f32[32, 448, 28, 28]", squeeze_55: "f32[448]", relu_21: "f32[32, 448, 28, 28]", convolution_29: "f32[32, 448, 28, 28]", getitem_39: "f32[1, 448, 1, 1]", rsqrt_19: "f32[1, 448, 1, 1]", mean_5: "f32[32, 448, 1, 1]", relu_23: "f32[32, 112, 1, 1]", convolution_31: "f32[32, 448, 1, 1]", mul_145: "f32[32, 448, 28, 28]", convolution_32: "f32[32, 448, 28, 28]", squeeze_61: "f32[448]", relu_24: "f32[32, 448, 28, 28]", convolution_33: "f32[32, 448, 28, 28]", squeeze_64: "f32[448]", relu_25: "f32[32, 448, 28, 28]", convolution_34: "f32[32, 448, 28, 28]", getitem_45: "f32[1, 448, 1, 1]", rsqrt_22: "f32[1, 448, 1, 1]", mean_6: "f32[32, 448, 1, 1]", relu_27: "f32[32, 112, 1, 1]", convolution_36: "f32[32, 448, 1, 1]", mul_167: "f32[32, 448, 28, 28]", convolution_37: "f32[32, 448, 28, 28]", squeeze_70: "f32[448]", relu_28: "f32[32, 448, 28, 28]", convolution_38: "f32[32, 896, 28, 28]", squeeze_73: "f32[896]", relu_29: "f32[32, 896, 28, 28]", convolution_39: "f32[32, 896, 14, 14]", getitem_51: "f32[1, 896, 1, 1]", rsqrt_25: "f32[1, 896, 1, 1]", mean_7: "f32[32, 896, 1, 1]", relu_31: "f32[32, 112, 1, 1]", convolution_41: "f32[32, 896, 1, 1]", mul_189: "f32[32, 896, 14, 14]", convolution_42: "f32[32, 896, 14, 14]", squeeze_79: "f32[896]", convolution_43: "f32[32, 896, 14, 14]", squeeze_82: "f32[896]", relu_32: "f32[32, 896, 14, 14]", convolution_44: "f32[32, 896, 14, 14]", squeeze_85: "f32[896]", relu_33: "f32[32, 896, 14, 14]", convolution_45: "f32[32, 896, 14, 14]", getitem_59: "f32[1, 896, 1, 1]", rsqrt_29: "f32[1, 896, 1, 1]", mean_8: "f32[32, 896, 1, 1]", relu_35: "f32[32, 224, 1, 1]", convolution_47: "f32[32, 896, 1, 1]", mul_218: "f32[32, 896, 14, 14]", convolution_48: "f32[32, 896, 14, 14]", squeeze_91: "f32[896]", relu_36: "f32[32, 896, 14, 14]", convolution_49: "f32[32, 896, 14, 14]", squeeze_94: "f32[896]", relu_37: "f32[32, 896, 14, 14]", convolution_50: "f32[32, 896, 14, 14]", getitem_65: "f32[1, 896, 1, 1]", rsqrt_32: "f32[1, 896, 1, 1]", mean_9: "f32[32, 896, 1, 1]", relu_39: "f32[32, 224, 1, 1]", convolution_52: "f32[32, 896, 1, 1]", mul_240: "f32[32, 896, 14, 14]", convolution_53: "f32[32, 896, 14, 14]", squeeze_100: "f32[896]", relu_40: "f32[32, 896, 14, 14]", convolution_54: "f32[32, 896, 14, 14]", squeeze_103: "f32[896]", relu_41: "f32[32, 896, 14, 14]", convolution_55: "f32[32, 896, 14, 14]", getitem_71: "f32[1, 896, 1, 1]", rsqrt_35: "f32[1, 896, 1, 1]", mean_10: "f32[32, 896, 1, 1]", relu_43: "f32[32, 224, 1, 1]", convolution_57: "f32[32, 896, 1, 1]", mul_262: "f32[32, 896, 14, 14]", convolution_58: "f32[32, 896, 14, 14]", squeeze_109: "f32[896]", relu_44: "f32[32, 896, 14, 14]", convolution_59: "f32[32, 896, 14, 14]", squeeze_112: "f32[896]", relu_45: "f32[32, 896, 14, 14]", convolution_60: "f32[32, 896, 14, 14]", getitem_77: "f32[1, 896, 1, 1]", rsqrt_38: "f32[1, 896, 1, 1]", mean_11: "f32[32, 896, 1, 1]", relu_47: "f32[32, 224, 1, 1]", convolution_62: "f32[32, 896, 1, 1]", mul_284: "f32[32, 896, 14, 14]", convolution_63: "f32[32, 896, 14, 14]", squeeze_118: "f32[896]", relu_48: "f32[32, 896, 14, 14]", convolution_64: "f32[32, 896, 14, 14]", squeeze_121: "f32[896]", relu_49: "f32[32, 896, 14, 14]", convolution_65: "f32[32, 896, 14, 14]", getitem_83: "f32[1, 896, 1, 1]", rsqrt_41: "f32[1, 896, 1, 1]", mean_12: "f32[32, 896, 1, 1]", relu_51: "f32[32, 224, 1, 1]", convolution_67: "f32[32, 896, 1, 1]", mul_306: "f32[32, 896, 14, 14]", convolution_68: "f32[32, 896, 14, 14]", squeeze_127: "f32[896]", relu_52: "f32[32, 896, 14, 14]", convolution_69: "f32[32, 896, 14, 14]", squeeze_130: "f32[896]", relu_53: "f32[32, 896, 14, 14]", convolution_70: "f32[32, 896, 14, 14]", getitem_89: "f32[1, 896, 1, 1]", rsqrt_44: "f32[1, 896, 1, 1]", mean_13: "f32[32, 896, 1, 1]", relu_55: "f32[32, 224, 1, 1]", convolution_72: "f32[32, 896, 1, 1]", mul_328: "f32[32, 896, 14, 14]", convolution_73: "f32[32, 896, 14, 14]", squeeze_136: "f32[896]", relu_56: "f32[32, 896, 14, 14]", convolution_74: "f32[32, 896, 14, 14]", squeeze_139: "f32[896]", relu_57: "f32[32, 896, 14, 14]", convolution_75: "f32[32, 896, 14, 14]", getitem_95: "f32[1, 896, 1, 1]", rsqrt_47: "f32[1, 896, 1, 1]", mean_14: "f32[32, 896, 1, 1]", relu_59: "f32[32, 224, 1, 1]", convolution_77: "f32[32, 896, 1, 1]", mul_350: "f32[32, 896, 14, 14]", convolution_78: "f32[32, 896, 14, 14]", squeeze_145: "f32[896]", relu_60: "f32[32, 896, 14, 14]", convolution_79: "f32[32, 896, 14, 14]", squeeze_148: "f32[896]", relu_61: "f32[32, 896, 14, 14]", convolution_80: "f32[32, 896, 14, 14]", getitem_101: "f32[1, 896, 1, 1]", rsqrt_50: "f32[1, 896, 1, 1]", mean_15: "f32[32, 896, 1, 1]", relu_63: "f32[32, 224, 1, 1]", convolution_82: "f32[32, 896, 1, 1]", mul_372: "f32[32, 896, 14, 14]", convolution_83: "f32[32, 896, 14, 14]", squeeze_154: "f32[896]", relu_64: "f32[32, 896, 14, 14]", convolution_84: "f32[32, 896, 14, 14]", squeeze_157: "f32[896]", relu_65: "f32[32, 896, 14, 14]", convolution_85: "f32[32, 896, 14, 14]", getitem_107: "f32[1, 896, 1, 1]", rsqrt_53: "f32[1, 896, 1, 1]", mean_16: "f32[32, 896, 1, 1]", relu_67: "f32[32, 224, 1, 1]", convolution_87: "f32[32, 896, 1, 1]", mul_394: "f32[32, 896, 14, 14]", convolution_88: "f32[32, 896, 14, 14]", squeeze_163: "f32[896]", relu_68: "f32[32, 896, 14, 14]", convolution_89: "f32[32, 896, 14, 14]", squeeze_166: "f32[896]", relu_69: "f32[32, 896, 14, 14]", convolution_90: "f32[32, 896, 14, 14]", getitem_113: "f32[1, 896, 1, 1]", rsqrt_56: "f32[1, 896, 1, 1]", mean_17: "f32[32, 896, 1, 1]", relu_71: "f32[32, 224, 1, 1]", convolution_92: "f32[32, 896, 1, 1]", mul_416: "f32[32, 896, 14, 14]", convolution_93: "f32[32, 896, 14, 14]", squeeze_172: "f32[896]", relu_72: "f32[32, 896, 14, 14]", convolution_94: "f32[32, 2240, 14, 14]", squeeze_175: "f32[2240]", relu_73: "f32[32, 2240, 14, 14]", convolution_95: "f32[32, 2240, 7, 7]", getitem_119: "f32[1, 2240, 1, 1]", rsqrt_59: "f32[1, 2240, 1, 1]", mean_18: "f32[32, 2240, 1, 1]", relu_75: "f32[32, 224, 1, 1]", convolution_97: "f32[32, 2240, 1, 1]", mul_438: "f32[32, 2240, 7, 7]", convolution_98: "f32[32, 2240, 7, 7]", getitem_121: "f32[1, 2240, 1, 1]", rsqrt_60: "f32[1, 2240, 1, 1]", convolution_99: "f32[32, 2240, 7, 7]", getitem_123: "f32[1, 2240, 1, 1]", rsqrt_61: "f32[1, 2240, 1, 1]", view: "f32[32, 2240]", unsqueeze_286: "f32[1, 2240, 1, 1]", unsqueeze_298: "f32[1, 896, 1, 1]", unsqueeze_322: "f32[1, 896, 1, 1]", unsqueeze_334: "f32[1, 896, 1, 1]", unsqueeze_358: "f32[1, 896, 1, 1]", unsqueeze_370: "f32[1, 896, 1, 1]", unsqueeze_394: "f32[1, 896, 1, 1]", unsqueeze_406: "f32[1, 896, 1, 1]", unsqueeze_430: "f32[1, 896, 1, 1]", unsqueeze_442: "f32[1, 896, 1, 1]", unsqueeze_466: "f32[1, 896, 1, 1]", unsqueeze_478: "f32[1, 896, 1, 1]", unsqueeze_502: "f32[1, 896, 1, 1]", unsqueeze_514: "f32[1, 896, 1, 1]", unsqueeze_538: "f32[1, 896, 1, 1]", unsqueeze_550: "f32[1, 896, 1, 1]", unsqueeze_574: "f32[1, 896, 1, 1]", unsqueeze_586: "f32[1, 896, 1, 1]", unsqueeze_610: "f32[1, 896, 1, 1]", unsqueeze_622: "f32[1, 896, 1, 1]", unsqueeze_646: "f32[1, 896, 1, 1]", unsqueeze_658: "f32[1, 896, 1, 1]", unsqueeze_670: "f32[1, 896, 1, 1]", unsqueeze_694: "f32[1, 896, 1, 1]", unsqueeze_706: "f32[1, 448, 1, 1]", unsqueeze_730: "f32[1, 448, 1, 1]", unsqueeze_742: "f32[1, 448, 1, 1]", unsqueeze_766: "f32[1, 448, 1, 1]", unsqueeze_778: "f32[1, 448, 1, 1]", unsqueeze_802: "f32[1, 448, 1, 1]", unsqueeze_814: "f32[1, 448, 1, 1]", unsqueeze_838: "f32[1, 448, 1, 1]", unsqueeze_850: "f32[1, 448, 1, 1]", unsqueeze_862: "f32[1, 448, 1, 1]", unsqueeze_886: "f32[1, 448, 1, 1]", unsqueeze_898: "f32[1, 224, 1, 1]", unsqueeze_922: "f32[1, 224, 1, 1]", unsqueeze_934: "f32[1, 224, 1, 1]", unsqueeze_946: "f32[1, 224, 1, 1]", unsqueeze_970: "f32[1, 224, 1, 1]", unsqueeze_982: "f32[1, 32, 1, 1]", tangents_1: "f32[32, 1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute: "f32[2240, 1000]" = torch.ops.aten.permute.default(primals_450, [1, 0]);  primals_450 = None
        permute_1: "f32[1000, 2240]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm: "f32[32, 2240]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "f32[1000, 32]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f32[1000, 2240]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_2: "f32[32, 2240, 1, 1]" = torch.ops.aten.reshape.default(mm, [32, 2240, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        expand: "f32[32, 2240, 7, 7]" = torch.ops.aten.expand.default(view_2, [32, 2240, 7, 7]);  view_2 = None
        div: "f32[32, 2240, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_60: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, getitem_121)
        mul_439: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
        unsqueeze_240: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_442, -1)
        unsqueeze_241: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        mul_445: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(mul_439, unsqueeze_241);  mul_439 = unsqueeze_241 = None
        unsqueeze_242: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_443, -1);  primals_443 = None
        unsqueeze_243: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        add_322: "f32[32, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_445, unsqueeze_243);  mul_445 = unsqueeze_243 = None
        sub_61: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, getitem_123)
        mul_446: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
        unsqueeze_244: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_448, -1)
        unsqueeze_245: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_452: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(mul_446, unsqueeze_245);  mul_446 = unsqueeze_245 = None
        unsqueeze_246: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_449, -1);  primals_449 = None
        unsqueeze_247: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_327: "f32[32, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_452, unsqueeze_247);  mul_452 = unsqueeze_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_328: "f32[32, 2240, 7, 7]" = torch.ops.aten.add.Tensor(add_322, add_327);  add_322 = add_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_76: "f32[32, 2240, 7, 7]" = torch.ops.aten.relu.default(add_328);  add_328 = None
        le: "b8[32, 2240, 7, 7]" = torch.ops.aten.le.Scalar(relu_76, 0);  relu_76 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 2240, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_183: "f32[2240]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
        unsqueeze_248: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
        unsqueeze_249: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
        unsqueeze_250: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None
        sum_2: "f32[2240]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_62: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_250);  convolution_99 = unsqueeze_250 = None
        mul_453: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_62)
        sum_3: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_453, [0, 2, 3]);  mul_453 = None
        mul_454: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_2, 0.0006377551020408163)
        unsqueeze_251: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_454, 0);  mul_454 = None
        unsqueeze_252: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
        unsqueeze_253: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
        mul_455: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_3, 0.0006377551020408163)
        squeeze_184: "f32[2240]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
        mul_456: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
        mul_457: "f32[2240]" = torch.ops.aten.mul.Tensor(mul_455, mul_456);  mul_455 = mul_456 = None
        unsqueeze_254: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_457, 0);  mul_457 = None
        unsqueeze_255: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
        unsqueeze_256: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
        mul_458: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_448);  primals_448 = None
        unsqueeze_257: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
        unsqueeze_258: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
        unsqueeze_259: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
        mul_459: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_256);  sub_62 = unsqueeze_256 = None
        sub_64: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_459);  mul_459 = None
        sub_65: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(sub_64, unsqueeze_253);  sub_64 = unsqueeze_253 = None
        mul_460: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_259);  sub_65 = unsqueeze_259 = None
        mul_461: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_184);  sum_3 = squeeze_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_460, relu_72, primals_444, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_460 = primals_444 = None
        getitem_124: "f32[32, 896, 14, 14]" = convolution_backward[0]
        getitem_125: "f32[2240, 896, 1, 1]" = convolution_backward[1];  convolution_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_180: "f32[2240]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        unsqueeze_260: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
        unsqueeze_261: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
        unsqueeze_262: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
        sum_4: "f32[2240]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_66: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_262);  convolution_98 = unsqueeze_262 = None
        mul_462: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_66)
        sum_5: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_462, [0, 2, 3]);  mul_462 = None
        mul_463: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_4, 0.0006377551020408163)
        unsqueeze_263: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_463, 0);  mul_463 = None
        unsqueeze_264: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
        unsqueeze_265: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
        mul_464: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_5, 0.0006377551020408163)
        squeeze_181: "f32[2240]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
        mul_465: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
        mul_466: "f32[2240]" = torch.ops.aten.mul.Tensor(mul_464, mul_465);  mul_464 = mul_465 = None
        unsqueeze_266: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_466, 0);  mul_466 = None
        unsqueeze_267: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
        unsqueeze_268: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
        mul_467: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_442);  primals_442 = None
        unsqueeze_269: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_467, 0);  mul_467 = None
        unsqueeze_270: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
        unsqueeze_271: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
        mul_468: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_268);  sub_66 = unsqueeze_268 = None
        sub_68: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_468);  where = mul_468 = None
        sub_69: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_265);  sub_68 = unsqueeze_265 = None
        mul_469: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_271);  sub_69 = unsqueeze_271 = None
        mul_470: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_181);  sum_5 = squeeze_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_469, mul_438, primals_438, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_469 = mul_438 = primals_438 = None
        getitem_127: "f32[32, 2240, 7, 7]" = convolution_backward_1[0]
        getitem_128: "f32[2240, 2240, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_59: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_95, getitem_119)
        mul_431: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
        unsqueeze_236: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_432, -1)
        unsqueeze_237: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_437: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(mul_431, unsqueeze_237);  mul_431 = unsqueeze_237 = None
        unsqueeze_238: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_433, -1);  primals_433 = None
        unsqueeze_239: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_317: "f32[32, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_437, unsqueeze_239);  mul_437 = unsqueeze_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_74: "f32[32, 2240, 7, 7]" = torch.ops.aten.relu.default(add_317);  add_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_471: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_127, relu_74)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_18: "f32[32, 2240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_97);  convolution_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_472: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_127, sigmoid_18);  getitem_127 = None
        sum_6: "f32[32, 2240, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_471, [2, 3], True);  mul_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_70: "f32[32, 2240, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_18)
        mul_473: "f32[32, 2240, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_18, sub_70);  sigmoid_18 = sub_70 = None
        mul_474: "f32[32, 2240, 1, 1]" = torch.ops.aten.mul.Tensor(sum_6, mul_473);  sum_6 = mul_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_7: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_474, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_474, relu_75, primals_436, [2240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_474 = primals_436 = None
        getitem_130: "f32[32, 224, 1, 1]" = convolution_backward_2[0]
        getitem_131: "f32[2240, 224, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_1: "b8[32, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_75, 0);  relu_75 = None
        where_1: "f32[32, 224, 1, 1]" = torch.ops.aten.where.self(le_1, full_default, getitem_130);  le_1 = getitem_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_8: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_1, mean_18, primals_434, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_1 = mean_18 = primals_434 = None
        getitem_133: "f32[32, 2240, 1, 1]" = convolution_backward_3[0]
        getitem_134: "f32[224, 2240, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_1: "f32[32, 2240, 7, 7]" = torch.ops.aten.expand.default(getitem_133, [32, 2240, 7, 7]);  getitem_133 = None
        div_1: "f32[32, 2240, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
        add_329: "f32[32, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_472, div_1);  mul_472 = div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_2: "b8[32, 2240, 7, 7]" = torch.ops.aten.le.Scalar(relu_74, 0);  relu_74 = None
        where_2: "f32[32, 2240, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, add_329);  le_2 = add_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_177: "f32[2240]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
        unsqueeze_272: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
        unsqueeze_273: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
        unsqueeze_274: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
        sum_9: "f32[2240]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        sub_71: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_274);  convolution_95 = unsqueeze_274 = None
        mul_475: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_71)
        sum_10: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_475, [0, 2, 3]);  mul_475 = None
        mul_476: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_9, 0.0006377551020408163)
        unsqueeze_275: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
        unsqueeze_276: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
        unsqueeze_277: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
        mul_477: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_10, 0.0006377551020408163)
        squeeze_178: "f32[2240]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
        mul_478: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
        mul_479: "f32[2240]" = torch.ops.aten.mul.Tensor(mul_477, mul_478);  mul_477 = mul_478 = None
        unsqueeze_278: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_479, 0);  mul_479 = None
        unsqueeze_279: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
        unsqueeze_280: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
        mul_480: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_432);  primals_432 = None
        unsqueeze_281: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
        unsqueeze_282: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
        unsqueeze_283: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
        mul_481: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_280);  sub_71 = unsqueeze_280 = None
        sub_73: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_481);  where_2 = mul_481 = None
        sub_74: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(sub_73, unsqueeze_277);  sub_73 = unsqueeze_277 = None
        mul_482: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_283);  sub_74 = unsqueeze_283 = None
        mul_483: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_178);  sum_10 = squeeze_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_482, relu_73, primals_428, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 20, [True, True, False]);  mul_482 = primals_428 = None
        getitem_136: "f32[32, 2240, 14, 14]" = convolution_backward_4[0]
        getitem_137: "f32[2240, 112, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_3: "b8[32, 2240, 14, 14]" = torch.ops.aten.le.Scalar(relu_73, 0);  relu_73 = None
        where_3: "f32[32, 2240, 14, 14]" = torch.ops.aten.where.self(le_3, full_default, getitem_136);  le_3 = getitem_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_11: "f32[2240]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        sub_75: "f32[32, 2240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_286);  convolution_94 = unsqueeze_286 = None
        mul_484: "f32[32, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(where_3, sub_75)
        sum_12: "f32[2240]" = torch.ops.aten.sum.dim_IntList(mul_484, [0, 2, 3]);  mul_484 = None
        mul_485: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_11, 0.00015943877551020407)
        unsqueeze_287: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
        unsqueeze_288: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
        unsqueeze_289: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
        mul_486: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_12, 0.00015943877551020407)
        mul_487: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
        mul_488: "f32[2240]" = torch.ops.aten.mul.Tensor(mul_486, mul_487);  mul_486 = mul_487 = None
        unsqueeze_290: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_488, 0);  mul_488 = None
        unsqueeze_291: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
        unsqueeze_292: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
        mul_489: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_426);  primals_426 = None
        unsqueeze_293: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(mul_489, 0);  mul_489 = None
        unsqueeze_294: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
        unsqueeze_295: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
        mul_490: "f32[32, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_292);  sub_75 = unsqueeze_292 = None
        sub_77: "f32[32, 2240, 14, 14]" = torch.ops.aten.sub.Tensor(where_3, mul_490);  where_3 = mul_490 = None
        sub_78: "f32[32, 2240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_77, unsqueeze_289);  sub_77 = unsqueeze_289 = None
        mul_491: "f32[32, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_295);  sub_78 = unsqueeze_295 = None
        mul_492: "f32[2240]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_175);  sum_12 = squeeze_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_491, relu_72, primals_422, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_491 = primals_422 = None
        getitem_139: "f32[32, 896, 14, 14]" = convolution_backward_5[0]
        getitem_140: "f32[2240, 896, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
        add_330: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(getitem_124, getitem_139);  getitem_124 = getitem_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_4: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_72, 0);  relu_72 = None
        where_4: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_4, full_default, add_330);  le_4 = add_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_13: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_79: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_298);  convolution_93 = unsqueeze_298 = None
        mul_493: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, sub_79)
        sum_14: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_493, [0, 2, 3]);  mul_493 = None
        mul_494: "f32[896]" = torch.ops.aten.mul.Tensor(sum_13, 0.00015943877551020407)
        unsqueeze_299: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
        unsqueeze_300: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
        unsqueeze_301: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
        mul_495: "f32[896]" = torch.ops.aten.mul.Tensor(sum_14, 0.00015943877551020407)
        mul_496: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
        mul_497: "f32[896]" = torch.ops.aten.mul.Tensor(mul_495, mul_496);  mul_495 = mul_496 = None
        unsqueeze_302: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_497, 0);  mul_497 = None
        unsqueeze_303: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
        unsqueeze_304: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
        mul_498: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_420);  primals_420 = None
        unsqueeze_305: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_498, 0);  mul_498 = None
        unsqueeze_306: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
        unsqueeze_307: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
        mul_499: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_304);  sub_79 = unsqueeze_304 = None
        sub_81: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_4, mul_499);  mul_499 = None
        sub_82: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_301);  sub_81 = unsqueeze_301 = None
        mul_500: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_307);  sub_82 = unsqueeze_307 = None
        mul_501: "f32[896]" = torch.ops.aten.mul.Tensor(sum_14, squeeze_172);  sum_14 = squeeze_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_500, mul_416, primals_416, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_500 = mul_416 = primals_416 = None
        getitem_142: "f32[32, 896, 14, 14]" = convolution_backward_6[0]
        getitem_143: "f32[896, 896, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_56: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, getitem_113)
        mul_409: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
        unsqueeze_224: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_410, -1)
        unsqueeze_225: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        mul_415: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_409, unsqueeze_225);  mul_409 = unsqueeze_225 = None
        unsqueeze_226: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_411, -1);  primals_411 = None
        unsqueeze_227: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        add_301: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_415, unsqueeze_227);  mul_415 = unsqueeze_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_70: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_301);  add_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_502: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_142, relu_70)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_17: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_92);  convolution_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_503: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_142, sigmoid_17);  getitem_142 = None
        sum_15: "f32[32, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_502, [2, 3], True);  mul_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_83: "f32[32, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_17)
        mul_504: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_17, sub_83);  sigmoid_17 = sub_83 = None
        mul_505: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_15, mul_504);  sum_15 = mul_504 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_16: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_505, [0, 2, 3])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_505, relu_71, primals_414, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_505 = primals_414 = None
        getitem_145: "f32[32, 224, 1, 1]" = convolution_backward_7[0]
        getitem_146: "f32[896, 224, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_5: "b8[32, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_71, 0);  relu_71 = None
        where_5: "f32[32, 224, 1, 1]" = torch.ops.aten.where.self(le_5, full_default, getitem_145);  le_5 = getitem_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_17: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_5, mean_17, primals_412, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = mean_17 = primals_412 = None
        getitem_148: "f32[32, 896, 1, 1]" = convolution_backward_8[0]
        getitem_149: "f32[224, 896, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_2: "f32[32, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_148, [32, 896, 14, 14]);  getitem_148 = None
        div_2: "f32[32, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_2, 196);  expand_2 = None
        add_331: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_503, div_2);  mul_503 = div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_6: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_70, 0);  relu_70 = None
        where_6: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_6, full_default, add_331);  le_6 = add_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_168: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
        unsqueeze_308: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
        unsqueeze_309: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
        unsqueeze_310: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
        sum_18: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_84: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_310);  convolution_90 = unsqueeze_310 = None
        mul_506: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, sub_84)
        sum_19: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_506, [0, 2, 3]);  mul_506 = None
        mul_507: "f32[896]" = torch.ops.aten.mul.Tensor(sum_18, 0.00015943877551020407)
        unsqueeze_311: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_507, 0);  mul_507 = None
        unsqueeze_312: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
        unsqueeze_313: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
        mul_508: "f32[896]" = torch.ops.aten.mul.Tensor(sum_19, 0.00015943877551020407)
        squeeze_169: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
        mul_509: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
        mul_510: "f32[896]" = torch.ops.aten.mul.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
        unsqueeze_314: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_510, 0);  mul_510 = None
        unsqueeze_315: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
        unsqueeze_316: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
        mul_511: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_410);  primals_410 = None
        unsqueeze_317: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
        unsqueeze_318: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
        unsqueeze_319: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
        mul_512: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_316);  sub_84 = unsqueeze_316 = None
        sub_86: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_6, mul_512);  where_6 = mul_512 = None
        sub_87: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_313);  sub_86 = unsqueeze_313 = None
        mul_513: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_319);  sub_87 = unsqueeze_319 = None
        mul_514: "f32[896]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_169);  sum_19 = squeeze_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_513, relu_69, primals_406, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_513 = primals_406 = None
        getitem_151: "f32[32, 896, 14, 14]" = convolution_backward_9[0]
        getitem_152: "f32[896, 112, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_7: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_69, 0);  relu_69 = None
        where_7: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_7, full_default, getitem_151);  le_7 = getitem_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_20: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        sub_88: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_322);  convolution_89 = unsqueeze_322 = None
        mul_515: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, sub_88)
        sum_21: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_515, [0, 2, 3]);  mul_515 = None
        mul_516: "f32[896]" = torch.ops.aten.mul.Tensor(sum_20, 0.00015943877551020407)
        unsqueeze_323: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
        unsqueeze_324: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
        unsqueeze_325: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
        mul_517: "f32[896]" = torch.ops.aten.mul.Tensor(sum_21, 0.00015943877551020407)
        mul_518: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
        mul_519: "f32[896]" = torch.ops.aten.mul.Tensor(mul_517, mul_518);  mul_517 = mul_518 = None
        unsqueeze_326: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_519, 0);  mul_519 = None
        unsqueeze_327: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
        unsqueeze_328: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
        mul_520: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_404);  primals_404 = None
        unsqueeze_329: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
        unsqueeze_330: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
        unsqueeze_331: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
        mul_521: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_328);  sub_88 = unsqueeze_328 = None
        sub_90: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_7, mul_521);  where_7 = mul_521 = None
        sub_91: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_90, unsqueeze_325);  sub_90 = unsqueeze_325 = None
        mul_522: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_331);  sub_91 = unsqueeze_331 = None
        mul_523: "f32[896]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_166);  sum_21 = squeeze_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_522, relu_68, primals_400, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_522 = primals_400 = None
        getitem_154: "f32[32, 896, 14, 14]" = convolution_backward_10[0]
        getitem_155: "f32[896, 896, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
        add_332: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_4, getitem_154);  where_4 = getitem_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_8: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_68, 0);  relu_68 = None
        where_8: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_8, full_default, add_332);  le_8 = add_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_22: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        sub_92: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_334);  convolution_88 = unsqueeze_334 = None
        mul_524: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_92)
        sum_23: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_524, [0, 2, 3]);  mul_524 = None
        mul_525: "f32[896]" = torch.ops.aten.mul.Tensor(sum_22, 0.00015943877551020407)
        unsqueeze_335: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_525, 0);  mul_525 = None
        unsqueeze_336: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
        unsqueeze_337: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
        mul_526: "f32[896]" = torch.ops.aten.mul.Tensor(sum_23, 0.00015943877551020407)
        mul_527: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
        mul_528: "f32[896]" = torch.ops.aten.mul.Tensor(mul_526, mul_527);  mul_526 = mul_527 = None
        unsqueeze_338: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_528, 0);  mul_528 = None
        unsqueeze_339: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
        unsqueeze_340: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
        mul_529: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_398);  primals_398 = None
        unsqueeze_341: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_529, 0);  mul_529 = None
        unsqueeze_342: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
        unsqueeze_343: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
        mul_530: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_340);  sub_92 = unsqueeze_340 = None
        sub_94: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_8, mul_530);  mul_530 = None
        sub_95: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_94, unsqueeze_337);  sub_94 = unsqueeze_337 = None
        mul_531: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_343);  sub_95 = unsqueeze_343 = None
        mul_532: "f32[896]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_163);  sum_23 = squeeze_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_531, mul_394, primals_394, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_531 = mul_394 = primals_394 = None
        getitem_157: "f32[32, 896, 14, 14]" = convolution_backward_11[0]
        getitem_158: "f32[896, 896, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_53: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, getitem_107)
        mul_387: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
        unsqueeze_212: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_388, -1)
        unsqueeze_213: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_393: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_387, unsqueeze_213);  mul_387 = unsqueeze_213 = None
        unsqueeze_214: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_389, -1);  primals_389 = None
        unsqueeze_215: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_285: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_393, unsqueeze_215);  mul_393 = unsqueeze_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_66: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_285);  add_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_533: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_157, relu_66)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_16: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_87);  convolution_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_534: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_157, sigmoid_16);  getitem_157 = None
        sum_24: "f32[32, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_533, [2, 3], True);  mul_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_96: "f32[32, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_16)
        mul_535: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_16, sub_96);  sigmoid_16 = sub_96 = None
        mul_536: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_24, mul_535);  sum_24 = mul_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_25: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 2, 3])
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_536, relu_67, primals_392, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_536 = primals_392 = None
        getitem_160: "f32[32, 224, 1, 1]" = convolution_backward_12[0]
        getitem_161: "f32[896, 224, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_9: "b8[32, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_67, 0);  relu_67 = None
        where_9: "f32[32, 224, 1, 1]" = torch.ops.aten.where.self(le_9, full_default, getitem_160);  le_9 = getitem_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_26: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_9, mean_16, primals_390, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_9 = mean_16 = primals_390 = None
        getitem_163: "f32[32, 896, 1, 1]" = convolution_backward_13[0]
        getitem_164: "f32[224, 896, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_3: "f32[32, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_163, [32, 896, 14, 14]);  getitem_163 = None
        div_3: "f32[32, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_3, 196);  expand_3 = None
        add_333: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_534, div_3);  mul_534 = div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_10: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_66, 0);  relu_66 = None
        where_10: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_10, full_default, add_333);  le_10 = add_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_159: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        unsqueeze_344: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
        unsqueeze_345: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
        unsqueeze_346: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
        sum_27: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_97: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_346);  convolution_85 = unsqueeze_346 = None
        mul_537: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_97)
        sum_28: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_537, [0, 2, 3]);  mul_537 = None
        mul_538: "f32[896]" = torch.ops.aten.mul.Tensor(sum_27, 0.00015943877551020407)
        unsqueeze_347: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_538, 0);  mul_538 = None
        unsqueeze_348: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
        unsqueeze_349: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
        mul_539: "f32[896]" = torch.ops.aten.mul.Tensor(sum_28, 0.00015943877551020407)
        squeeze_160: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
        mul_540: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
        mul_541: "f32[896]" = torch.ops.aten.mul.Tensor(mul_539, mul_540);  mul_539 = mul_540 = None
        unsqueeze_350: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
        unsqueeze_351: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
        unsqueeze_352: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
        mul_542: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_388);  primals_388 = None
        unsqueeze_353: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_542, 0);  mul_542 = None
        unsqueeze_354: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
        unsqueeze_355: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
        mul_543: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_352);  sub_97 = unsqueeze_352 = None
        sub_99: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_543);  where_10 = mul_543 = None
        sub_100: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_349);  sub_99 = unsqueeze_349 = None
        mul_544: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_355);  sub_100 = unsqueeze_355 = None
        mul_545: "f32[896]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_160);  sum_28 = squeeze_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_544, relu_65, primals_384, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_544 = primals_384 = None
        getitem_166: "f32[32, 896, 14, 14]" = convolution_backward_14[0]
        getitem_167: "f32[896, 112, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_11: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_65, 0);  relu_65 = None
        where_11: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_11, full_default, getitem_166);  le_11 = getitem_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_29: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        sub_101: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_358);  convolution_84 = unsqueeze_358 = None
        mul_546: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_101)
        sum_30: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_546, [0, 2, 3]);  mul_546 = None
        mul_547: "f32[896]" = torch.ops.aten.mul.Tensor(sum_29, 0.00015943877551020407)
        unsqueeze_359: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_547, 0);  mul_547 = None
        unsqueeze_360: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
        unsqueeze_361: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
        mul_548: "f32[896]" = torch.ops.aten.mul.Tensor(sum_30, 0.00015943877551020407)
        mul_549: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
        mul_550: "f32[896]" = torch.ops.aten.mul.Tensor(mul_548, mul_549);  mul_548 = mul_549 = None
        unsqueeze_362: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_550, 0);  mul_550 = None
        unsqueeze_363: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
        unsqueeze_364: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
        mul_551: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_382);  primals_382 = None
        unsqueeze_365: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
        unsqueeze_366: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
        unsqueeze_367: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
        mul_552: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_364);  sub_101 = unsqueeze_364 = None
        sub_103: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_552);  where_11 = mul_552 = None
        sub_104: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_361);  sub_103 = unsqueeze_361 = None
        mul_553: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_367);  sub_104 = unsqueeze_367 = None
        mul_554: "f32[896]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_157);  sum_30 = squeeze_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_553, relu_64, primals_378, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_553 = primals_378 = None
        getitem_169: "f32[32, 896, 14, 14]" = convolution_backward_15[0]
        getitem_170: "f32[896, 896, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
        add_334: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_8, getitem_169);  where_8 = getitem_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_12: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_64, 0);  relu_64 = None
        where_12: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_12, full_default, add_334);  le_12 = add_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_31: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        sub_105: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_370);  convolution_83 = unsqueeze_370 = None
        mul_555: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_105)
        sum_32: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_555, [0, 2, 3]);  mul_555 = None
        mul_556: "f32[896]" = torch.ops.aten.mul.Tensor(sum_31, 0.00015943877551020407)
        unsqueeze_371: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_556, 0);  mul_556 = None
        unsqueeze_372: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
        unsqueeze_373: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
        mul_557: "f32[896]" = torch.ops.aten.mul.Tensor(sum_32, 0.00015943877551020407)
        mul_558: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_559: "f32[896]" = torch.ops.aten.mul.Tensor(mul_557, mul_558);  mul_557 = mul_558 = None
        unsqueeze_374: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_559, 0);  mul_559 = None
        unsqueeze_375: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
        unsqueeze_376: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
        mul_560: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_376);  primals_376 = None
        unsqueeze_377: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_560, 0);  mul_560 = None
        unsqueeze_378: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
        unsqueeze_379: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
        mul_561: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_376);  sub_105 = unsqueeze_376 = None
        sub_107: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_561);  mul_561 = None
        sub_108: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_373);  sub_107 = unsqueeze_373 = None
        mul_562: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_379);  sub_108 = unsqueeze_379 = None
        mul_563: "f32[896]" = torch.ops.aten.mul.Tensor(sum_32, squeeze_154);  sum_32 = squeeze_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_562, mul_372, primals_372, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_562 = mul_372 = primals_372 = None
        getitem_172: "f32[32, 896, 14, 14]" = convolution_backward_16[0]
        getitem_173: "f32[896, 896, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_50: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, getitem_101)
        mul_365: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        unsqueeze_200: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_366, -1)
        unsqueeze_201: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        mul_371: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_365, unsqueeze_201);  mul_365 = unsqueeze_201 = None
        unsqueeze_202: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_367, -1);  primals_367 = None
        unsqueeze_203: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        add_269: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_371, unsqueeze_203);  mul_371 = unsqueeze_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_62: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_269);  add_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_564: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_172, relu_62)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_15: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_82);  convolution_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_565: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_172, sigmoid_15);  getitem_172 = None
        sum_33: "f32[32, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_564, [2, 3], True);  mul_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_109: "f32[32, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_15)
        mul_566: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_15, sub_109);  sigmoid_15 = sub_109 = None
        mul_567: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_33, mul_566);  sum_33 = mul_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_34: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_567, [0, 2, 3])
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_567, relu_63, primals_370, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_567 = primals_370 = None
        getitem_175: "f32[32, 224, 1, 1]" = convolution_backward_17[0]
        getitem_176: "f32[896, 224, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_13: "b8[32, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_63, 0);  relu_63 = None
        where_13: "f32[32, 224, 1, 1]" = torch.ops.aten.where.self(le_13, full_default, getitem_175);  le_13 = getitem_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_35: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_13, mean_15, primals_368, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_13 = mean_15 = primals_368 = None
        getitem_178: "f32[32, 896, 1, 1]" = convolution_backward_18[0]
        getitem_179: "f32[224, 896, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_4: "f32[32, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_178, [32, 896, 14, 14]);  getitem_178 = None
        div_4: "f32[32, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_4, 196);  expand_4 = None
        add_335: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_565, div_4);  mul_565 = div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_14: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_62, 0);  relu_62 = None
        where_14: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, add_335);  le_14 = add_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_150: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        unsqueeze_380: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
        unsqueeze_381: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
        unsqueeze_382: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
        sum_36: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        sub_110: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_382);  convolution_80 = unsqueeze_382 = None
        mul_568: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_110)
        sum_37: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_568, [0, 2, 3]);  mul_568 = None
        mul_569: "f32[896]" = torch.ops.aten.mul.Tensor(sum_36, 0.00015943877551020407)
        unsqueeze_383: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_569, 0);  mul_569 = None
        unsqueeze_384: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
        unsqueeze_385: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
        mul_570: "f32[896]" = torch.ops.aten.mul.Tensor(sum_37, 0.00015943877551020407)
        squeeze_151: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_571: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_572: "f32[896]" = torch.ops.aten.mul.Tensor(mul_570, mul_571);  mul_570 = mul_571 = None
        unsqueeze_386: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
        unsqueeze_387: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
        unsqueeze_388: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
        mul_573: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_366);  primals_366 = None
        unsqueeze_389: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_573, 0);  mul_573 = None
        unsqueeze_390: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
        unsqueeze_391: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
        mul_574: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_388);  sub_110 = unsqueeze_388 = None
        sub_112: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_574);  where_14 = mul_574 = None
        sub_113: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_112, unsqueeze_385);  sub_112 = unsqueeze_385 = None
        mul_575: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_391);  sub_113 = unsqueeze_391 = None
        mul_576: "f32[896]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_151);  sum_37 = squeeze_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_575, relu_61, primals_362, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_575 = primals_362 = None
        getitem_181: "f32[32, 896, 14, 14]" = convolution_backward_19[0]
        getitem_182: "f32[896, 112, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_15: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_61, 0);  relu_61 = None
        where_15: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_15, full_default, getitem_181);  le_15 = getitem_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_38: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        sub_114: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_394);  convolution_79 = unsqueeze_394 = None
        mul_577: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_114)
        sum_39: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_577, [0, 2, 3]);  mul_577 = None
        mul_578: "f32[896]" = torch.ops.aten.mul.Tensor(sum_38, 0.00015943877551020407)
        unsqueeze_395: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_578, 0);  mul_578 = None
        unsqueeze_396: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
        unsqueeze_397: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
        mul_579: "f32[896]" = torch.ops.aten.mul.Tensor(sum_39, 0.00015943877551020407)
        mul_580: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_581: "f32[896]" = torch.ops.aten.mul.Tensor(mul_579, mul_580);  mul_579 = mul_580 = None
        unsqueeze_398: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_581, 0);  mul_581 = None
        unsqueeze_399: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
        unsqueeze_400: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
        mul_582: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_360);  primals_360 = None
        unsqueeze_401: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
        unsqueeze_402: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
        unsqueeze_403: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
        mul_583: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_400);  sub_114 = unsqueeze_400 = None
        sub_116: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_583);  where_15 = mul_583 = None
        sub_117: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_116, unsqueeze_397);  sub_116 = unsqueeze_397 = None
        mul_584: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_403);  sub_117 = unsqueeze_403 = None
        mul_585: "f32[896]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_148);  sum_39 = squeeze_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_584, relu_60, primals_356, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_584 = primals_356 = None
        getitem_184: "f32[32, 896, 14, 14]" = convolution_backward_20[0]
        getitem_185: "f32[896, 896, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
        add_336: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_12, getitem_184);  where_12 = getitem_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_16: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None
        where_16: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_16, full_default, add_336);  le_16 = add_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_40: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        sub_118: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_406);  convolution_78 = unsqueeze_406 = None
        mul_586: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_118)
        sum_41: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_586, [0, 2, 3]);  mul_586 = None
        mul_587: "f32[896]" = torch.ops.aten.mul.Tensor(sum_40, 0.00015943877551020407)
        unsqueeze_407: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_587, 0);  mul_587 = None
        unsqueeze_408: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
        unsqueeze_409: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
        mul_588: "f32[896]" = torch.ops.aten.mul.Tensor(sum_41, 0.00015943877551020407)
        mul_589: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_590: "f32[896]" = torch.ops.aten.mul.Tensor(mul_588, mul_589);  mul_588 = mul_589 = None
        unsqueeze_410: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_590, 0);  mul_590 = None
        unsqueeze_411: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
        unsqueeze_412: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
        mul_591: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_354);  primals_354 = None
        unsqueeze_413: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_591, 0);  mul_591 = None
        unsqueeze_414: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
        unsqueeze_415: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
        mul_592: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_412);  sub_118 = unsqueeze_412 = None
        sub_120: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_592);  mul_592 = None
        sub_121: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_120, unsqueeze_409);  sub_120 = unsqueeze_409 = None
        mul_593: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_415);  sub_121 = unsqueeze_415 = None
        mul_594: "f32[896]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_145);  sum_41 = squeeze_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_593, mul_350, primals_350, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_593 = mul_350 = primals_350 = None
        getitem_187: "f32[32, 896, 14, 14]" = convolution_backward_21[0]
        getitem_188: "f32[896, 896, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_47: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_95)
        mul_343: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        unsqueeze_188: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_344, -1)
        unsqueeze_189: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_349: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_189);  mul_343 = unsqueeze_189 = None
        unsqueeze_190: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_345, -1);  primals_345 = None
        unsqueeze_191: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_253: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_191);  mul_349 = unsqueeze_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_58: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_253);  add_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_595: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_187, relu_58)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_14: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_77);  convolution_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_596: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_187, sigmoid_14);  getitem_187 = None
        sum_42: "f32[32, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_595, [2, 3], True);  mul_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_122: "f32[32, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_14)
        mul_597: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_14, sub_122);  sigmoid_14 = sub_122 = None
        mul_598: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_42, mul_597);  sum_42 = mul_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_43: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_598, [0, 2, 3])
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_598, relu_59, primals_348, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_598 = primals_348 = None
        getitem_190: "f32[32, 224, 1, 1]" = convolution_backward_22[0]
        getitem_191: "f32[896, 224, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_17: "b8[32, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_59, 0);  relu_59 = None
        where_17: "f32[32, 224, 1, 1]" = torch.ops.aten.where.self(le_17, full_default, getitem_190);  le_17 = getitem_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_44: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(where_17, mean_14, primals_346, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_17 = mean_14 = primals_346 = None
        getitem_193: "f32[32, 896, 1, 1]" = convolution_backward_23[0]
        getitem_194: "f32[224, 896, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_5: "f32[32, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_193, [32, 896, 14, 14]);  getitem_193 = None
        div_5: "f32[32, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_5, 196);  expand_5 = None
        add_337: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_596, div_5);  mul_596 = div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_18: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_58, 0);  relu_58 = None
        where_18: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_18, full_default, add_337);  le_18 = add_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_141: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        unsqueeze_416: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
        unsqueeze_417: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
        unsqueeze_418: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
        sum_45: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
        sub_123: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_418);  convolution_75 = unsqueeze_418 = None
        mul_599: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_123)
        sum_46: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 2, 3]);  mul_599 = None
        mul_600: "f32[896]" = torch.ops.aten.mul.Tensor(sum_45, 0.00015943877551020407)
        unsqueeze_419: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
        unsqueeze_420: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
        unsqueeze_421: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
        mul_601: "f32[896]" = torch.ops.aten.mul.Tensor(sum_46, 0.00015943877551020407)
        squeeze_142: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_602: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_603: "f32[896]" = torch.ops.aten.mul.Tensor(mul_601, mul_602);  mul_601 = mul_602 = None
        unsqueeze_422: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
        unsqueeze_423: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
        unsqueeze_424: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
        mul_604: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_344);  primals_344 = None
        unsqueeze_425: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
        unsqueeze_426: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
        unsqueeze_427: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
        mul_605: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_424);  sub_123 = unsqueeze_424 = None
        sub_125: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_18, mul_605);  where_18 = mul_605 = None
        sub_126: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_125, unsqueeze_421);  sub_125 = unsqueeze_421 = None
        mul_606: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_427);  sub_126 = unsqueeze_427 = None
        mul_607: "f32[896]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_142);  sum_46 = squeeze_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_606, relu_57, primals_340, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_606 = primals_340 = None
        getitem_196: "f32[32, 896, 14, 14]" = convolution_backward_24[0]
        getitem_197: "f32[896, 112, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_19: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_57, 0);  relu_57 = None
        where_19: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_19, full_default, getitem_196);  le_19 = getitem_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_47: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
        sub_127: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_430);  convolution_74 = unsqueeze_430 = None
        mul_608: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_127)
        sum_48: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 2, 3]);  mul_608 = None
        mul_609: "f32[896]" = torch.ops.aten.mul.Tensor(sum_47, 0.00015943877551020407)
        unsqueeze_431: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
        unsqueeze_432: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
        unsqueeze_433: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
        mul_610: "f32[896]" = torch.ops.aten.mul.Tensor(sum_48, 0.00015943877551020407)
        mul_611: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_612: "f32[896]" = torch.ops.aten.mul.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
        unsqueeze_434: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
        unsqueeze_435: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
        unsqueeze_436: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
        mul_613: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_338);  primals_338 = None
        unsqueeze_437: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
        unsqueeze_438: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
        unsqueeze_439: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
        mul_614: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_436);  sub_127 = unsqueeze_436 = None
        sub_129: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_614);  where_19 = mul_614 = None
        sub_130: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_129, unsqueeze_433);  sub_129 = unsqueeze_433 = None
        mul_615: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_439);  sub_130 = unsqueeze_439 = None
        mul_616: "f32[896]" = torch.ops.aten.mul.Tensor(sum_48, squeeze_139);  sum_48 = squeeze_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_615, relu_56, primals_334, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_615 = primals_334 = None
        getitem_199: "f32[32, 896, 14, 14]" = convolution_backward_25[0]
        getitem_200: "f32[896, 896, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
        add_338: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_16, getitem_199);  where_16 = getitem_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_20: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_56, 0);  relu_56 = None
        where_20: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_20, full_default, add_338);  le_20 = add_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_49: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
        sub_131: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_442);  convolution_73 = unsqueeze_442 = None
        mul_617: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_131)
        sum_50: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_617, [0, 2, 3]);  mul_617 = None
        mul_618: "f32[896]" = torch.ops.aten.mul.Tensor(sum_49, 0.00015943877551020407)
        unsqueeze_443: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_618, 0);  mul_618 = None
        unsqueeze_444: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
        unsqueeze_445: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
        mul_619: "f32[896]" = torch.ops.aten.mul.Tensor(sum_50, 0.00015943877551020407)
        mul_620: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_621: "f32[896]" = torch.ops.aten.mul.Tensor(mul_619, mul_620);  mul_619 = mul_620 = None
        unsqueeze_446: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
        unsqueeze_447: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
        unsqueeze_448: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
        mul_622: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_332);  primals_332 = None
        unsqueeze_449: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_622, 0);  mul_622 = None
        unsqueeze_450: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
        unsqueeze_451: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
        mul_623: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_448);  sub_131 = unsqueeze_448 = None
        sub_133: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_20, mul_623);  mul_623 = None
        sub_134: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_133, unsqueeze_445);  sub_133 = unsqueeze_445 = None
        mul_624: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_451);  sub_134 = unsqueeze_451 = None
        mul_625: "f32[896]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_136);  sum_50 = squeeze_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_624, mul_328, primals_328, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_624 = mul_328 = primals_328 = None
        getitem_202: "f32[32, 896, 14, 14]" = convolution_backward_26[0]
        getitem_203: "f32[896, 896, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_44: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_89)
        mul_321: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        unsqueeze_176: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_322, -1)
        unsqueeze_177: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        mul_327: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_321, unsqueeze_177);  mul_321 = unsqueeze_177 = None
        unsqueeze_178: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_323, -1);  primals_323 = None
        unsqueeze_179: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        add_237: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_327, unsqueeze_179);  mul_327 = unsqueeze_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_54: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_237);  add_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_626: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_202, relu_54)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_13: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_72);  convolution_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_627: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_202, sigmoid_13);  getitem_202 = None
        sum_51: "f32[32, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_626, [2, 3], True);  mul_626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_135: "f32[32, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_13)
        mul_628: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_13, sub_135);  sigmoid_13 = sub_135 = None
        mul_629: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_51, mul_628);  sum_51 = mul_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_52: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_629, [0, 2, 3])
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_629, relu_55, primals_326, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_629 = primals_326 = None
        getitem_205: "f32[32, 224, 1, 1]" = convolution_backward_27[0]
        getitem_206: "f32[896, 224, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_21: "b8[32, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_55, 0);  relu_55 = None
        where_21: "f32[32, 224, 1, 1]" = torch.ops.aten.where.self(le_21, full_default, getitem_205);  le_21 = getitem_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_53: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(where_21, mean_13, primals_324, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_21 = mean_13 = primals_324 = None
        getitem_208: "f32[32, 896, 1, 1]" = convolution_backward_28[0]
        getitem_209: "f32[224, 896, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_6: "f32[32, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_208, [32, 896, 14, 14]);  getitem_208 = None
        div_6: "f32[32, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_6, 196);  expand_6 = None
        add_339: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_627, div_6);  mul_627 = div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_22: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_54, 0);  relu_54 = None
        where_22: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_22, full_default, add_339);  le_22 = add_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_132: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        unsqueeze_452: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
        unsqueeze_453: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
        unsqueeze_454: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
        sum_54: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
        sub_136: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_454);  convolution_70 = unsqueeze_454 = None
        mul_630: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_136)
        sum_55: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_630, [0, 2, 3]);  mul_630 = None
        mul_631: "f32[896]" = torch.ops.aten.mul.Tensor(sum_54, 0.00015943877551020407)
        unsqueeze_455: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
        unsqueeze_456: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
        unsqueeze_457: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
        mul_632: "f32[896]" = torch.ops.aten.mul.Tensor(sum_55, 0.00015943877551020407)
        squeeze_133: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_633: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_634: "f32[896]" = torch.ops.aten.mul.Tensor(mul_632, mul_633);  mul_632 = mul_633 = None
        unsqueeze_458: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_634, 0);  mul_634 = None
        unsqueeze_459: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
        unsqueeze_460: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
        mul_635: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_322);  primals_322 = None
        unsqueeze_461: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
        unsqueeze_462: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
        unsqueeze_463: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
        mul_636: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_460);  sub_136 = unsqueeze_460 = None
        sub_138: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_22, mul_636);  where_22 = mul_636 = None
        sub_139: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_457);  sub_138 = unsqueeze_457 = None
        mul_637: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_463);  sub_139 = unsqueeze_463 = None
        mul_638: "f32[896]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_133);  sum_55 = squeeze_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_637, relu_53, primals_318, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_637 = primals_318 = None
        getitem_211: "f32[32, 896, 14, 14]" = convolution_backward_29[0]
        getitem_212: "f32[896, 112, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_23: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_53, 0);  relu_53 = None
        where_23: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_23, full_default, getitem_211);  le_23 = getitem_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_56: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
        sub_140: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_466);  convolution_69 = unsqueeze_466 = None
        mul_639: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_140)
        sum_57: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_639, [0, 2, 3]);  mul_639 = None
        mul_640: "f32[896]" = torch.ops.aten.mul.Tensor(sum_56, 0.00015943877551020407)
        unsqueeze_467: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_640, 0);  mul_640 = None
        unsqueeze_468: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
        unsqueeze_469: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
        mul_641: "f32[896]" = torch.ops.aten.mul.Tensor(sum_57, 0.00015943877551020407)
        mul_642: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_643: "f32[896]" = torch.ops.aten.mul.Tensor(mul_641, mul_642);  mul_641 = mul_642 = None
        unsqueeze_470: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_643, 0);  mul_643 = None
        unsqueeze_471: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
        unsqueeze_472: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
        mul_644: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_316);  primals_316 = None
        unsqueeze_473: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_644, 0);  mul_644 = None
        unsqueeze_474: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
        unsqueeze_475: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
        mul_645: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_472);  sub_140 = unsqueeze_472 = None
        sub_142: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_23, mul_645);  where_23 = mul_645 = None
        sub_143: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_469);  sub_142 = unsqueeze_469 = None
        mul_646: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_475);  sub_143 = unsqueeze_475 = None
        mul_647: "f32[896]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_130);  sum_57 = squeeze_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_646, relu_52, primals_312, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_646 = primals_312 = None
        getitem_214: "f32[32, 896, 14, 14]" = convolution_backward_30[0]
        getitem_215: "f32[896, 896, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
        add_340: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_20, getitem_214);  where_20 = getitem_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_24: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_52, 0);  relu_52 = None
        where_24: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_24, full_default, add_340);  le_24 = add_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_58: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
        sub_144: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_478);  convolution_68 = unsqueeze_478 = None
        mul_648: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_144)
        sum_59: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_648, [0, 2, 3]);  mul_648 = None
        mul_649: "f32[896]" = torch.ops.aten.mul.Tensor(sum_58, 0.00015943877551020407)
        unsqueeze_479: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_649, 0);  mul_649 = None
        unsqueeze_480: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
        unsqueeze_481: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
        mul_650: "f32[896]" = torch.ops.aten.mul.Tensor(sum_59, 0.00015943877551020407)
        mul_651: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_652: "f32[896]" = torch.ops.aten.mul.Tensor(mul_650, mul_651);  mul_650 = mul_651 = None
        unsqueeze_482: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_652, 0);  mul_652 = None
        unsqueeze_483: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
        unsqueeze_484: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
        mul_653: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_310);  primals_310 = None
        unsqueeze_485: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_653, 0);  mul_653 = None
        unsqueeze_486: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
        unsqueeze_487: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
        mul_654: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_484);  sub_144 = unsqueeze_484 = None
        sub_146: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_24, mul_654);  mul_654 = None
        sub_147: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_481);  sub_146 = unsqueeze_481 = None
        mul_655: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_487);  sub_147 = unsqueeze_487 = None
        mul_656: "f32[896]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_127);  sum_59 = squeeze_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_655, mul_306, primals_306, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_655 = mul_306 = primals_306 = None
        getitem_217: "f32[32, 896, 14, 14]" = convolution_backward_31[0]
        getitem_218: "f32[896, 896, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_41: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_83)
        mul_299: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        unsqueeze_164: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_300, -1)
        unsqueeze_165: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_305: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_299, unsqueeze_165);  mul_299 = unsqueeze_165 = None
        unsqueeze_166: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_301, -1);  primals_301 = None
        unsqueeze_167: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_221: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_305, unsqueeze_167);  mul_305 = unsqueeze_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_50: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_221);  add_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_657: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_217, relu_50)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_12: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67);  convolution_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_658: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_217, sigmoid_12);  getitem_217 = None
        sum_60: "f32[32, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_657, [2, 3], True);  mul_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_148: "f32[32, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_12)
        mul_659: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_12, sub_148);  sigmoid_12 = sub_148 = None
        mul_660: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_60, mul_659);  sum_60 = mul_659 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_61: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_660, [0, 2, 3])
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_660, relu_51, primals_304, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_660 = primals_304 = None
        getitem_220: "f32[32, 224, 1, 1]" = convolution_backward_32[0]
        getitem_221: "f32[896, 224, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_25: "b8[32, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_51, 0);  relu_51 = None
        where_25: "f32[32, 224, 1, 1]" = torch.ops.aten.where.self(le_25, full_default, getitem_220);  le_25 = getitem_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_62: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_25, mean_12, primals_302, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_25 = mean_12 = primals_302 = None
        getitem_223: "f32[32, 896, 1, 1]" = convolution_backward_33[0]
        getitem_224: "f32[224, 896, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_7: "f32[32, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_223, [32, 896, 14, 14]);  getitem_223 = None
        div_7: "f32[32, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_7, 196);  expand_7 = None
        add_341: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_658, div_7);  mul_658 = div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_26: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
        where_26: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_26, full_default, add_341);  le_26 = add_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_123: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        unsqueeze_488: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
        unsqueeze_489: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
        unsqueeze_490: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
        sum_63: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
        sub_149: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_490);  convolution_65 = unsqueeze_490 = None
        mul_661: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, sub_149)
        sum_64: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 2, 3]);  mul_661 = None
        mul_662: "f32[896]" = torch.ops.aten.mul.Tensor(sum_63, 0.00015943877551020407)
        unsqueeze_491: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_662, 0);  mul_662 = None
        unsqueeze_492: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
        unsqueeze_493: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
        mul_663: "f32[896]" = torch.ops.aten.mul.Tensor(sum_64, 0.00015943877551020407)
        squeeze_124: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_664: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_665: "f32[896]" = torch.ops.aten.mul.Tensor(mul_663, mul_664);  mul_663 = mul_664 = None
        unsqueeze_494: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
        unsqueeze_495: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
        unsqueeze_496: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
        mul_666: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_300);  primals_300 = None
        unsqueeze_497: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
        unsqueeze_498: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
        unsqueeze_499: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
        mul_667: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_496);  sub_149 = unsqueeze_496 = None
        sub_151: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_26, mul_667);  where_26 = mul_667 = None
        sub_152: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_493);  sub_151 = unsqueeze_493 = None
        mul_668: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_499);  sub_152 = unsqueeze_499 = None
        mul_669: "f32[896]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_124);  sum_64 = squeeze_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_668, relu_49, primals_296, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_668 = primals_296 = None
        getitem_226: "f32[32, 896, 14, 14]" = convolution_backward_34[0]
        getitem_227: "f32[896, 112, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_27: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
        where_27: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_27, full_default, getitem_226);  le_27 = getitem_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_65: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
        sub_153: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_502);  convolution_64 = unsqueeze_502 = None
        mul_670: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_153)
        sum_66: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_670, [0, 2, 3]);  mul_670 = None
        mul_671: "f32[896]" = torch.ops.aten.mul.Tensor(sum_65, 0.00015943877551020407)
        unsqueeze_503: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
        unsqueeze_504: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
        unsqueeze_505: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
        mul_672: "f32[896]" = torch.ops.aten.mul.Tensor(sum_66, 0.00015943877551020407)
        mul_673: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_674: "f32[896]" = torch.ops.aten.mul.Tensor(mul_672, mul_673);  mul_672 = mul_673 = None
        unsqueeze_506: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
        unsqueeze_507: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
        unsqueeze_508: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
        mul_675: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_294);  primals_294 = None
        unsqueeze_509: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
        unsqueeze_510: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
        unsqueeze_511: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
        mul_676: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_508);  sub_153 = unsqueeze_508 = None
        sub_155: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_27, mul_676);  where_27 = mul_676 = None
        sub_156: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_505);  sub_155 = unsqueeze_505 = None
        mul_677: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_511);  sub_156 = unsqueeze_511 = None
        mul_678: "f32[896]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_121);  sum_66 = squeeze_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_677, relu_48, primals_290, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_677 = primals_290 = None
        getitem_229: "f32[32, 896, 14, 14]" = convolution_backward_35[0]
        getitem_230: "f32[896, 896, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
        add_342: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_24, getitem_229);  where_24 = getitem_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_28: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
        where_28: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_28, full_default, add_342);  le_28 = add_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_67: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
        sub_157: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_514);  convolution_63 = unsqueeze_514 = None
        mul_679: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, sub_157)
        sum_68: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_679, [0, 2, 3]);  mul_679 = None
        mul_680: "f32[896]" = torch.ops.aten.mul.Tensor(sum_67, 0.00015943877551020407)
        unsqueeze_515: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
        unsqueeze_516: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
        unsqueeze_517: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
        mul_681: "f32[896]" = torch.ops.aten.mul.Tensor(sum_68, 0.00015943877551020407)
        mul_682: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_683: "f32[896]" = torch.ops.aten.mul.Tensor(mul_681, mul_682);  mul_681 = mul_682 = None
        unsqueeze_518: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
        unsqueeze_519: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
        unsqueeze_520: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
        mul_684: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_288);  primals_288 = None
        unsqueeze_521: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
        unsqueeze_522: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
        unsqueeze_523: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
        mul_685: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_520);  sub_157 = unsqueeze_520 = None
        sub_159: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_28, mul_685);  mul_685 = None
        sub_160: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_517);  sub_159 = unsqueeze_517 = None
        mul_686: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_523);  sub_160 = unsqueeze_523 = None
        mul_687: "f32[896]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_118);  sum_68 = squeeze_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_686, mul_284, primals_284, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_686 = mul_284 = primals_284 = None
        getitem_232: "f32[32, 896, 14, 14]" = convolution_backward_36[0]
        getitem_233: "f32[896, 896, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_38: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_77)
        mul_277: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        unsqueeze_152: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_278, -1)
        unsqueeze_153: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        mul_283: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_153);  mul_277 = unsqueeze_153 = None
        unsqueeze_154: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_279, -1);  primals_279 = None
        unsqueeze_155: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        add_205: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_283, unsqueeze_155);  mul_283 = unsqueeze_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_46: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_205);  add_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_688: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_232, relu_46)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_11: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_62);  convolution_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_689: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_232, sigmoid_11);  getitem_232 = None
        sum_69: "f32[32, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_688, [2, 3], True);  mul_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_161: "f32[32, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_11)
        mul_690: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_11, sub_161);  sigmoid_11 = sub_161 = None
        mul_691: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_69, mul_690);  sum_69 = mul_690 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_70: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 2, 3])
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_691, relu_47, primals_282, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_691 = primals_282 = None
        getitem_235: "f32[32, 224, 1, 1]" = convolution_backward_37[0]
        getitem_236: "f32[896, 224, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_29: "b8[32, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
        where_29: "f32[32, 224, 1, 1]" = torch.ops.aten.where.self(le_29, full_default, getitem_235);  le_29 = getitem_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_71: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(where_29, mean_11, primals_280, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_29 = mean_11 = primals_280 = None
        getitem_238: "f32[32, 896, 1, 1]" = convolution_backward_38[0]
        getitem_239: "f32[224, 896, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_8: "f32[32, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_238, [32, 896, 14, 14]);  getitem_238 = None
        div_8: "f32[32, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
        add_343: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_689, div_8);  mul_689 = div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_30: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
        where_30: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_30, full_default, add_343);  le_30 = add_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_114: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        unsqueeze_524: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
        unsqueeze_525: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
        unsqueeze_526: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
        sum_72: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
        sub_162: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_526);  convolution_60 = unsqueeze_526 = None
        mul_692: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, sub_162)
        sum_73: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_692, [0, 2, 3]);  mul_692 = None
        mul_693: "f32[896]" = torch.ops.aten.mul.Tensor(sum_72, 0.00015943877551020407)
        unsqueeze_527: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
        unsqueeze_528: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
        unsqueeze_529: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
        mul_694: "f32[896]" = torch.ops.aten.mul.Tensor(sum_73, 0.00015943877551020407)
        squeeze_115: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_695: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_696: "f32[896]" = torch.ops.aten.mul.Tensor(mul_694, mul_695);  mul_694 = mul_695 = None
        unsqueeze_530: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_696, 0);  mul_696 = None
        unsqueeze_531: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
        unsqueeze_532: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
        mul_697: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_278);  primals_278 = None
        unsqueeze_533: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_697, 0);  mul_697 = None
        unsqueeze_534: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
        unsqueeze_535: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
        mul_698: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_532);  sub_162 = unsqueeze_532 = None
        sub_164: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_30, mul_698);  where_30 = mul_698 = None
        sub_165: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_164, unsqueeze_529);  sub_164 = unsqueeze_529 = None
        mul_699: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_535);  sub_165 = unsqueeze_535 = None
        mul_700: "f32[896]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_115);  sum_73 = squeeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_699, relu_45, primals_274, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_699 = primals_274 = None
        getitem_241: "f32[32, 896, 14, 14]" = convolution_backward_39[0]
        getitem_242: "f32[896, 112, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_31: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
        where_31: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_31, full_default, getitem_241);  le_31 = getitem_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_74: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
        sub_166: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_538);  convolution_59 = unsqueeze_538 = None
        mul_701: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_166)
        sum_75: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_701, [0, 2, 3]);  mul_701 = None
        mul_702: "f32[896]" = torch.ops.aten.mul.Tensor(sum_74, 0.00015943877551020407)
        unsqueeze_539: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
        unsqueeze_540: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
        unsqueeze_541: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
        mul_703: "f32[896]" = torch.ops.aten.mul.Tensor(sum_75, 0.00015943877551020407)
        mul_704: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_705: "f32[896]" = torch.ops.aten.mul.Tensor(mul_703, mul_704);  mul_703 = mul_704 = None
        unsqueeze_542: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_705, 0);  mul_705 = None
        unsqueeze_543: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
        unsqueeze_544: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
        mul_706: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_272);  primals_272 = None
        unsqueeze_545: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_706, 0);  mul_706 = None
        unsqueeze_546: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
        unsqueeze_547: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
        mul_707: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_544);  sub_166 = unsqueeze_544 = None
        sub_168: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_31, mul_707);  where_31 = mul_707 = None
        sub_169: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_168, unsqueeze_541);  sub_168 = unsqueeze_541 = None
        mul_708: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_547);  sub_169 = unsqueeze_547 = None
        mul_709: "f32[896]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_112);  sum_75 = squeeze_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_708, relu_44, primals_268, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_708 = primals_268 = None
        getitem_244: "f32[32, 896, 14, 14]" = convolution_backward_40[0]
        getitem_245: "f32[896, 896, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
        add_344: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_28, getitem_244);  where_28 = getitem_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_32: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
        where_32: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_32, full_default, add_344);  le_32 = add_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_76: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
        sub_170: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_550);  convolution_58 = unsqueeze_550 = None
        mul_710: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_32, sub_170)
        sum_77: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_710, [0, 2, 3]);  mul_710 = None
        mul_711: "f32[896]" = torch.ops.aten.mul.Tensor(sum_76, 0.00015943877551020407)
        unsqueeze_551: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
        unsqueeze_552: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
        unsqueeze_553: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
        mul_712: "f32[896]" = torch.ops.aten.mul.Tensor(sum_77, 0.00015943877551020407)
        mul_713: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_714: "f32[896]" = torch.ops.aten.mul.Tensor(mul_712, mul_713);  mul_712 = mul_713 = None
        unsqueeze_554: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_714, 0);  mul_714 = None
        unsqueeze_555: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
        unsqueeze_556: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
        mul_715: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_266);  primals_266 = None
        unsqueeze_557: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_715, 0);  mul_715 = None
        unsqueeze_558: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
        unsqueeze_559: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
        mul_716: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_556);  sub_170 = unsqueeze_556 = None
        sub_172: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_32, mul_716);  mul_716 = None
        sub_173: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_172, unsqueeze_553);  sub_172 = unsqueeze_553 = None
        mul_717: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_559);  sub_173 = unsqueeze_559 = None
        mul_718: "f32[896]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_109);  sum_77 = squeeze_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_717, mul_262, primals_262, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_717 = mul_262 = primals_262 = None
        getitem_247: "f32[32, 896, 14, 14]" = convolution_backward_41[0]
        getitem_248: "f32[896, 896, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_35: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_71)
        mul_255: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        unsqueeze_140: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_256, -1)
        unsqueeze_141: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_261: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_255, unsqueeze_141);  mul_255 = unsqueeze_141 = None
        unsqueeze_142: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_257, -1);  primals_257 = None
        unsqueeze_143: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_189: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_261, unsqueeze_143);  mul_261 = unsqueeze_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_42: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_189);  add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_719: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_247, relu_42)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_10: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_57);  convolution_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_720: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_247, sigmoid_10);  getitem_247 = None
        sum_78: "f32[32, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_719, [2, 3], True);  mul_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_174: "f32[32, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_10)
        mul_721: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_10, sub_174);  sigmoid_10 = sub_174 = None
        mul_722: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_78, mul_721);  sum_78 = mul_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_79: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_722, [0, 2, 3])
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_722, relu_43, primals_260, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_722 = primals_260 = None
        getitem_250: "f32[32, 224, 1, 1]" = convolution_backward_42[0]
        getitem_251: "f32[896, 224, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_33: "b8[32, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
        where_33: "f32[32, 224, 1, 1]" = torch.ops.aten.where.self(le_33, full_default, getitem_250);  le_33 = getitem_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_80: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(where_33, mean_10, primals_258, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_33 = mean_10 = primals_258 = None
        getitem_253: "f32[32, 896, 1, 1]" = convolution_backward_43[0]
        getitem_254: "f32[224, 896, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_9: "f32[32, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_253, [32, 896, 14, 14]);  getitem_253 = None
        div_9: "f32[32, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
        add_345: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_720, div_9);  mul_720 = div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_34: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
        where_34: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_34, full_default, add_345);  le_34 = add_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_105: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        unsqueeze_560: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
        unsqueeze_561: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
        unsqueeze_562: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
        sum_81: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
        sub_175: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_562);  convolution_55 = unsqueeze_562 = None
        mul_723: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, sub_175)
        sum_82: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_723, [0, 2, 3]);  mul_723 = None
        mul_724: "f32[896]" = torch.ops.aten.mul.Tensor(sum_81, 0.00015943877551020407)
        unsqueeze_563: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_724, 0);  mul_724 = None
        unsqueeze_564: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 2);  unsqueeze_563 = None
        unsqueeze_565: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 3);  unsqueeze_564 = None
        mul_725: "f32[896]" = torch.ops.aten.mul.Tensor(sum_82, 0.00015943877551020407)
        squeeze_106: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_726: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_727: "f32[896]" = torch.ops.aten.mul.Tensor(mul_725, mul_726);  mul_725 = mul_726 = None
        unsqueeze_566: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_727, 0);  mul_727 = None
        unsqueeze_567: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
        unsqueeze_568: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
        mul_728: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_256);  primals_256 = None
        unsqueeze_569: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_728, 0);  mul_728 = None
        unsqueeze_570: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
        unsqueeze_571: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
        mul_729: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_568);  sub_175 = unsqueeze_568 = None
        sub_177: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_34, mul_729);  where_34 = mul_729 = None
        sub_178: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_177, unsqueeze_565);  sub_177 = unsqueeze_565 = None
        mul_730: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_571);  sub_178 = unsqueeze_571 = None
        mul_731: "f32[896]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_106);  sum_82 = squeeze_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_730, relu_41, primals_252, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_730 = primals_252 = None
        getitem_256: "f32[32, 896, 14, 14]" = convolution_backward_44[0]
        getitem_257: "f32[896, 112, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_35: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
        where_35: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_35, full_default, getitem_256);  le_35 = getitem_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_83: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
        sub_179: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_574);  convolution_54 = unsqueeze_574 = None
        mul_732: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_179)
        sum_84: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_732, [0, 2, 3]);  mul_732 = None
        mul_733: "f32[896]" = torch.ops.aten.mul.Tensor(sum_83, 0.00015943877551020407)
        unsqueeze_575: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_733, 0);  mul_733 = None
        unsqueeze_576: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
        unsqueeze_577: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
        mul_734: "f32[896]" = torch.ops.aten.mul.Tensor(sum_84, 0.00015943877551020407)
        mul_735: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_736: "f32[896]" = torch.ops.aten.mul.Tensor(mul_734, mul_735);  mul_734 = mul_735 = None
        unsqueeze_578: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
        unsqueeze_579: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
        unsqueeze_580: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
        mul_737: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_250);  primals_250 = None
        unsqueeze_581: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
        unsqueeze_582: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
        unsqueeze_583: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
        mul_738: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_580);  sub_179 = unsqueeze_580 = None
        sub_181: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_35, mul_738);  where_35 = mul_738 = None
        sub_182: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_181, unsqueeze_577);  sub_181 = unsqueeze_577 = None
        mul_739: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_583);  sub_182 = unsqueeze_583 = None
        mul_740: "f32[896]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_103);  sum_84 = squeeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_739, relu_40, primals_246, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_739 = primals_246 = None
        getitem_259: "f32[32, 896, 14, 14]" = convolution_backward_45[0]
        getitem_260: "f32[896, 896, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
        add_346: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_32, getitem_259);  where_32 = getitem_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_36: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
        where_36: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_36, full_default, add_346);  le_36 = add_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_85: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
        sub_183: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_586);  convolution_53 = unsqueeze_586 = None
        mul_741: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, sub_183)
        sum_86: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_741, [0, 2, 3]);  mul_741 = None
        mul_742: "f32[896]" = torch.ops.aten.mul.Tensor(sum_85, 0.00015943877551020407)
        unsqueeze_587: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_742, 0);  mul_742 = None
        unsqueeze_588: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
        unsqueeze_589: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
        mul_743: "f32[896]" = torch.ops.aten.mul.Tensor(sum_86, 0.00015943877551020407)
        mul_744: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_745: "f32[896]" = torch.ops.aten.mul.Tensor(mul_743, mul_744);  mul_743 = mul_744 = None
        unsqueeze_590: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_745, 0);  mul_745 = None
        unsqueeze_591: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
        unsqueeze_592: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
        mul_746: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_244);  primals_244 = None
        unsqueeze_593: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
        unsqueeze_594: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
        unsqueeze_595: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
        mul_747: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_592);  sub_183 = unsqueeze_592 = None
        sub_185: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_36, mul_747);  mul_747 = None
        sub_186: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_185, unsqueeze_589);  sub_185 = unsqueeze_589 = None
        mul_748: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_595);  sub_186 = unsqueeze_595 = None
        mul_749: "f32[896]" = torch.ops.aten.mul.Tensor(sum_86, squeeze_100);  sum_86 = squeeze_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_748, mul_240, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_748 = mul_240 = primals_240 = None
        getitem_262: "f32[32, 896, 14, 14]" = convolution_backward_46[0]
        getitem_263: "f32[896, 896, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_32: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_65)
        mul_233: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        unsqueeze_128: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_234, -1)
        unsqueeze_129: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        mul_239: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_233, unsqueeze_129);  mul_233 = unsqueeze_129 = None
        unsqueeze_130: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_235, -1);  primals_235 = None
        unsqueeze_131: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        add_173: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_131);  mul_239 = unsqueeze_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_38: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_173);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_750: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_262, relu_38)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_9: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_52);  convolution_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_751: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_262, sigmoid_9);  getitem_262 = None
        sum_87: "f32[32, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_750, [2, 3], True);  mul_750 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_187: "f32[32, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_9)
        mul_752: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_9, sub_187);  sigmoid_9 = sub_187 = None
        mul_753: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_87, mul_752);  sum_87 = mul_752 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_88: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_753, [0, 2, 3])
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_753, relu_39, primals_238, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_753 = primals_238 = None
        getitem_265: "f32[32, 224, 1, 1]" = convolution_backward_47[0]
        getitem_266: "f32[896, 224, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_37: "b8[32, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
        where_37: "f32[32, 224, 1, 1]" = torch.ops.aten.where.self(le_37, full_default, getitem_265);  le_37 = getitem_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_89: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(where_37, mean_9, primals_236, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_37 = mean_9 = primals_236 = None
        getitem_268: "f32[32, 896, 1, 1]" = convolution_backward_48[0]
        getitem_269: "f32[224, 896, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_10: "f32[32, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_268, [32, 896, 14, 14]);  getitem_268 = None
        div_10: "f32[32, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_10, 196);  expand_10 = None
        add_347: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_751, div_10);  mul_751 = div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_38: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
        where_38: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_38, full_default, add_347);  le_38 = add_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_96: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        unsqueeze_596: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
        unsqueeze_597: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
        unsqueeze_598: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
        sum_90: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
        sub_188: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_598);  convolution_50 = unsqueeze_598 = None
        mul_754: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, sub_188)
        sum_91: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_754, [0, 2, 3]);  mul_754 = None
        mul_755: "f32[896]" = torch.ops.aten.mul.Tensor(sum_90, 0.00015943877551020407)
        unsqueeze_599: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
        unsqueeze_600: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
        unsqueeze_601: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
        mul_756: "f32[896]" = torch.ops.aten.mul.Tensor(sum_91, 0.00015943877551020407)
        squeeze_97: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_757: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_758: "f32[896]" = torch.ops.aten.mul.Tensor(mul_756, mul_757);  mul_756 = mul_757 = None
        unsqueeze_602: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
        unsqueeze_603: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
        unsqueeze_604: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
        mul_759: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_234);  primals_234 = None
        unsqueeze_605: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_759, 0);  mul_759 = None
        unsqueeze_606: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
        unsqueeze_607: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
        mul_760: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_604);  sub_188 = unsqueeze_604 = None
        sub_190: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_38, mul_760);  where_38 = mul_760 = None
        sub_191: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_601);  sub_190 = unsqueeze_601 = None
        mul_761: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_607);  sub_191 = unsqueeze_607 = None
        mul_762: "f32[896]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_97);  sum_91 = squeeze_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_761, relu_37, primals_230, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_761 = primals_230 = None
        getitem_271: "f32[32, 896, 14, 14]" = convolution_backward_49[0]
        getitem_272: "f32[896, 112, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_39: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
        where_39: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_39, full_default, getitem_271);  le_39 = getitem_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_92: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
        sub_192: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_610);  convolution_49 = unsqueeze_610 = None
        mul_763: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_192)
        sum_93: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_763, [0, 2, 3]);  mul_763 = None
        mul_764: "f32[896]" = torch.ops.aten.mul.Tensor(sum_92, 0.00015943877551020407)
        unsqueeze_611: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
        unsqueeze_612: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 2);  unsqueeze_611 = None
        unsqueeze_613: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 3);  unsqueeze_612 = None
        mul_765: "f32[896]" = torch.ops.aten.mul.Tensor(sum_93, 0.00015943877551020407)
        mul_766: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_767: "f32[896]" = torch.ops.aten.mul.Tensor(mul_765, mul_766);  mul_765 = mul_766 = None
        unsqueeze_614: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_767, 0);  mul_767 = None
        unsqueeze_615: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 2);  unsqueeze_614 = None
        unsqueeze_616: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 3);  unsqueeze_615 = None
        mul_768: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_228);  primals_228 = None
        unsqueeze_617: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_768, 0);  mul_768 = None
        unsqueeze_618: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
        unsqueeze_619: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
        mul_769: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_616);  sub_192 = unsqueeze_616 = None
        sub_194: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_39, mul_769);  where_39 = mul_769 = None
        sub_195: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_613);  sub_194 = unsqueeze_613 = None
        mul_770: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_619);  sub_195 = unsqueeze_619 = None
        mul_771: "f32[896]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_94);  sum_93 = squeeze_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_770, relu_36, primals_224, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_770 = primals_224 = None
        getitem_274: "f32[32, 896, 14, 14]" = convolution_backward_50[0]
        getitem_275: "f32[896, 896, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
        add_348: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_36, getitem_274);  where_36 = getitem_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_40: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
        where_40: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_40, full_default, add_348);  le_40 = add_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_94: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
        sub_196: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_622);  convolution_48 = unsqueeze_622 = None
        mul_772: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_196)
        sum_95: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_772, [0, 2, 3]);  mul_772 = None
        mul_773: "f32[896]" = torch.ops.aten.mul.Tensor(sum_94, 0.00015943877551020407)
        unsqueeze_623: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
        unsqueeze_624: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
        unsqueeze_625: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
        mul_774: "f32[896]" = torch.ops.aten.mul.Tensor(sum_95, 0.00015943877551020407)
        mul_775: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_776: "f32[896]" = torch.ops.aten.mul.Tensor(mul_774, mul_775);  mul_774 = mul_775 = None
        unsqueeze_626: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
        unsqueeze_627: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
        unsqueeze_628: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
        mul_777: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_222);  primals_222 = None
        unsqueeze_629: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_777, 0);  mul_777 = None
        unsqueeze_630: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
        unsqueeze_631: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
        mul_778: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_628);  sub_196 = unsqueeze_628 = None
        sub_198: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_40, mul_778);  mul_778 = None
        sub_199: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_625);  sub_198 = unsqueeze_625 = None
        mul_779: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_631);  sub_199 = unsqueeze_631 = None
        mul_780: "f32[896]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_91);  sum_95 = squeeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_779, mul_218, primals_218, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_779 = mul_218 = primals_218 = None
        getitem_277: "f32[32, 896, 14, 14]" = convolution_backward_51[0]
        getitem_278: "f32[896, 896, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_29: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_59)
        mul_211: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        unsqueeze_116: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_212, -1)
        unsqueeze_117: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_217: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_117);  mul_211 = unsqueeze_117 = None
        unsqueeze_118: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_213, -1);  primals_213 = None
        unsqueeze_119: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_157: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_217, unsqueeze_119);  mul_217 = unsqueeze_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_34: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_157);  add_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_781: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_277, relu_34)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_8: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_47);  convolution_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_782: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_277, sigmoid_8);  getitem_277 = None
        sum_96: "f32[32, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_781, [2, 3], True);  mul_781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_200: "f32[32, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_8)
        mul_783: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_8, sub_200);  sigmoid_8 = sub_200 = None
        mul_784: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_96, mul_783);  sum_96 = mul_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_97: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_784, [0, 2, 3])
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_784, relu_35, primals_216, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_784 = primals_216 = None
        getitem_280: "f32[32, 224, 1, 1]" = convolution_backward_52[0]
        getitem_281: "f32[896, 224, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_41: "b8[32, 224, 1, 1]" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        where_41: "f32[32, 224, 1, 1]" = torch.ops.aten.where.self(le_41, full_default, getitem_280);  le_41 = getitem_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_98: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(where_41, mean_8, primals_214, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_41 = mean_8 = primals_214 = None
        getitem_283: "f32[32, 896, 1, 1]" = convolution_backward_53[0]
        getitem_284: "f32[224, 896, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_11: "f32[32, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_283, [32, 896, 14, 14]);  getitem_283 = None
        div_11: "f32[32, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_11, 196);  expand_11 = None
        add_349: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_782, div_11);  mul_782 = div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_42: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        where_42: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_42, full_default, add_349);  le_42 = add_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_87: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        unsqueeze_632: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_633: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
        unsqueeze_634: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
        sum_99: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
        sub_201: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_634);  convolution_45 = unsqueeze_634 = None
        mul_785: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, sub_201)
        sum_100: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_785, [0, 2, 3]);  mul_785 = None
        mul_786: "f32[896]" = torch.ops.aten.mul.Tensor(sum_99, 0.00015943877551020407)
        unsqueeze_635: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
        unsqueeze_636: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
        unsqueeze_637: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
        mul_787: "f32[896]" = torch.ops.aten.mul.Tensor(sum_100, 0.00015943877551020407)
        squeeze_88: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_788: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_789: "f32[896]" = torch.ops.aten.mul.Tensor(mul_787, mul_788);  mul_787 = mul_788 = None
        unsqueeze_638: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_789, 0);  mul_789 = None
        unsqueeze_639: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
        unsqueeze_640: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
        mul_790: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_212);  primals_212 = None
        unsqueeze_641: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_790, 0);  mul_790 = None
        unsqueeze_642: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
        unsqueeze_643: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
        mul_791: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_640);  sub_201 = unsqueeze_640 = None
        sub_203: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_42, mul_791);  where_42 = mul_791 = None
        sub_204: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_637);  sub_203 = unsqueeze_637 = None
        mul_792: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_643);  sub_204 = unsqueeze_643 = None
        mul_793: "f32[896]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_88);  sum_100 = squeeze_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_792, relu_33, primals_208, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_792 = primals_208 = None
        getitem_286: "f32[32, 896, 14, 14]" = convolution_backward_54[0]
        getitem_287: "f32[896, 112, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_43: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        where_43: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_43, full_default, getitem_286);  le_43 = getitem_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_101: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
        sub_205: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_646);  convolution_44 = unsqueeze_646 = None
        mul_794: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_205)
        sum_102: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_794, [0, 2, 3]);  mul_794 = None
        mul_795: "f32[896]" = torch.ops.aten.mul.Tensor(sum_101, 0.00015943877551020407)
        unsqueeze_647: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_795, 0);  mul_795 = None
        unsqueeze_648: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
        unsqueeze_649: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
        mul_796: "f32[896]" = torch.ops.aten.mul.Tensor(sum_102, 0.00015943877551020407)
        mul_797: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_798: "f32[896]" = torch.ops.aten.mul.Tensor(mul_796, mul_797);  mul_796 = mul_797 = None
        unsqueeze_650: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
        unsqueeze_651: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
        unsqueeze_652: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
        mul_799: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_206);  primals_206 = None
        unsqueeze_653: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
        unsqueeze_654: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
        unsqueeze_655: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
        mul_800: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_652);  sub_205 = unsqueeze_652 = None
        sub_207: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_43, mul_800);  where_43 = mul_800 = None
        sub_208: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_649);  sub_207 = unsqueeze_649 = None
        mul_801: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_655);  sub_208 = unsqueeze_655 = None
        mul_802: "f32[896]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_85);  sum_102 = squeeze_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_801, relu_32, primals_202, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_801 = primals_202 = None
        getitem_289: "f32[32, 896, 14, 14]" = convolution_backward_55[0]
        getitem_290: "f32[896, 896, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
        add_350: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(where_40, getitem_289);  where_40 = getitem_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_44: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
        where_44: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_44, full_default, add_350);  le_44 = add_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_103: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
        sub_209: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_658);  convolution_43 = unsqueeze_658 = None
        mul_803: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, sub_209)
        sum_104: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_803, [0, 2, 3]);  mul_803 = None
        mul_804: "f32[896]" = torch.ops.aten.mul.Tensor(sum_103, 0.00015943877551020407)
        unsqueeze_659: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_804, 0);  mul_804 = None
        unsqueeze_660: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 2);  unsqueeze_659 = None
        unsqueeze_661: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 3);  unsqueeze_660 = None
        mul_805: "f32[896]" = torch.ops.aten.mul.Tensor(sum_104, 0.00015943877551020407)
        mul_806: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_807: "f32[896]" = torch.ops.aten.mul.Tensor(mul_805, mul_806);  mul_805 = mul_806 = None
        unsqueeze_662: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
        unsqueeze_663: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 2);  unsqueeze_662 = None
        unsqueeze_664: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 3);  unsqueeze_663 = None
        mul_808: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_200);  primals_200 = None
        unsqueeze_665: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
        unsqueeze_666: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
        unsqueeze_667: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
        mul_809: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_664);  sub_209 = unsqueeze_664 = None
        sub_211: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_44, mul_809);  mul_809 = None
        sub_212: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_661);  sub_211 = unsqueeze_661 = None
        mul_810: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_667);  sub_212 = unsqueeze_667 = None
        mul_811: "f32[896]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_82);  sum_104 = squeeze_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_810, relu_28, primals_196, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_810 = primals_196 = None
        getitem_292: "f32[32, 448, 28, 28]" = convolution_backward_56[0]
        getitem_293: "f32[896, 448, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_105: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
        sub_213: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_670);  convolution_42 = unsqueeze_670 = None
        mul_812: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, sub_213)
        sum_106: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_812, [0, 2, 3]);  mul_812 = None
        mul_813: "f32[896]" = torch.ops.aten.mul.Tensor(sum_105, 0.00015943877551020407)
        unsqueeze_671: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_813, 0);  mul_813 = None
        unsqueeze_672: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 2);  unsqueeze_671 = None
        unsqueeze_673: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 3);  unsqueeze_672 = None
        mul_814: "f32[896]" = torch.ops.aten.mul.Tensor(sum_106, 0.00015943877551020407)
        mul_815: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_816: "f32[896]" = torch.ops.aten.mul.Tensor(mul_814, mul_815);  mul_814 = mul_815 = None
        unsqueeze_674: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
        unsqueeze_675: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 2);  unsqueeze_674 = None
        unsqueeze_676: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 3);  unsqueeze_675 = None
        mul_817: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_194);  primals_194 = None
        unsqueeze_677: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_817, 0);  mul_817 = None
        unsqueeze_678: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
        unsqueeze_679: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
        mul_818: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_676);  sub_213 = unsqueeze_676 = None
        sub_215: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_44, mul_818);  where_44 = mul_818 = None
        sub_216: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_673);  sub_215 = unsqueeze_673 = None
        mul_819: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_679);  sub_216 = unsqueeze_679 = None
        mul_820: "f32[896]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_79);  sum_106 = squeeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_819, mul_189, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_819 = mul_189 = primals_190 = None
        getitem_295: "f32[32, 896, 14, 14]" = convolution_backward_57[0]
        getitem_296: "f32[896, 896, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_25: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_51)
        mul_182: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        unsqueeze_100: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_184, -1)
        unsqueeze_101: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_188: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_101);  mul_182 = unsqueeze_101 = None
        unsqueeze_102: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_185, -1);  primals_185 = None
        unsqueeze_103: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_136: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_103);  mul_188 = unsqueeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_30: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_136);  add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_821: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_295, relu_30)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_7: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_41);  convolution_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_822: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_295, sigmoid_7);  getitem_295 = None
        sum_107: "f32[32, 896, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_821, [2, 3], True);  mul_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_217: "f32[32, 896, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_7)
        mul_823: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_7, sub_217);  sigmoid_7 = sub_217 = None
        mul_824: "f32[32, 896, 1, 1]" = torch.ops.aten.mul.Tensor(sum_107, mul_823);  sum_107 = mul_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_108: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_824, [0, 2, 3])
        convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_824, relu_31, primals_188, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_824 = primals_188 = None
        getitem_298: "f32[32, 112, 1, 1]" = convolution_backward_58[0]
        getitem_299: "f32[896, 112, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_45: "b8[32, 112, 1, 1]" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        where_45: "f32[32, 112, 1, 1]" = torch.ops.aten.where.self(le_45, full_default, getitem_298);  le_45 = getitem_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_109: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
        convolution_backward_59 = torch.ops.aten.convolution_backward.default(where_45, mean_7, primals_186, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_45 = mean_7 = primals_186 = None
        getitem_301: "f32[32, 896, 1, 1]" = convolution_backward_59[0]
        getitem_302: "f32[112, 896, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_12: "f32[32, 896, 14, 14]" = torch.ops.aten.expand.default(getitem_301, [32, 896, 14, 14]);  getitem_301 = None
        div_12: "f32[32, 896, 14, 14]" = torch.ops.aten.div.Scalar(expand_12, 196);  expand_12 = None
        add_351: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_822, div_12);  mul_822 = div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_46: "b8[32, 896, 14, 14]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        where_46: "f32[32, 896, 14, 14]" = torch.ops.aten.where.self(le_46, full_default, add_351);  le_46 = add_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_75: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        unsqueeze_680: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_681: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
        unsqueeze_682: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
        sum_110: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
        sub_218: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_682);  convolution_39 = unsqueeze_682 = None
        mul_825: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(where_46, sub_218)
        sum_111: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_825, [0, 2, 3]);  mul_825 = None
        mul_826: "f32[896]" = torch.ops.aten.mul.Tensor(sum_110, 0.00015943877551020407)
        unsqueeze_683: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_826, 0);  mul_826 = None
        unsqueeze_684: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 2);  unsqueeze_683 = None
        unsqueeze_685: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 3);  unsqueeze_684 = None
        mul_827: "f32[896]" = torch.ops.aten.mul.Tensor(sum_111, 0.00015943877551020407)
        squeeze_76: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_828: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_829: "f32[896]" = torch.ops.aten.mul.Tensor(mul_827, mul_828);  mul_827 = mul_828 = None
        unsqueeze_686: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_829, 0);  mul_829 = None
        unsqueeze_687: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 2);  unsqueeze_686 = None
        unsqueeze_688: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 3);  unsqueeze_687 = None
        mul_830: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_184);  primals_184 = None
        unsqueeze_689: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_830, 0);  mul_830 = None
        unsqueeze_690: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
        unsqueeze_691: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
        mul_831: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_688);  sub_218 = unsqueeze_688 = None
        sub_220: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(where_46, mul_831);  where_46 = mul_831 = None
        sub_221: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(sub_220, unsqueeze_685);  sub_220 = unsqueeze_685 = None
        mul_832: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_691);  sub_221 = unsqueeze_691 = None
        mul_833: "f32[896]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_76);  sum_111 = squeeze_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_832, relu_29, primals_180, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_832 = primals_180 = None
        getitem_304: "f32[32, 896, 28, 28]" = convolution_backward_60[0]
        getitem_305: "f32[896, 112, 3, 3]" = convolution_backward_60[1];  convolution_backward_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_47: "b8[32, 896, 28, 28]" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        where_47: "f32[32, 896, 28, 28]" = torch.ops.aten.where.self(le_47, full_default, getitem_304);  le_47 = getitem_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_112: "f32[896]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
        sub_222: "f32[32, 896, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_694);  convolution_38 = unsqueeze_694 = None
        mul_834: "f32[32, 896, 28, 28]" = torch.ops.aten.mul.Tensor(where_47, sub_222)
        sum_113: "f32[896]" = torch.ops.aten.sum.dim_IntList(mul_834, [0, 2, 3]);  mul_834 = None
        mul_835: "f32[896]" = torch.ops.aten.mul.Tensor(sum_112, 3.985969387755102e-05)
        unsqueeze_695: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_835, 0);  mul_835 = None
        unsqueeze_696: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 2);  unsqueeze_695 = None
        unsqueeze_697: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 3);  unsqueeze_696 = None
        mul_836: "f32[896]" = torch.ops.aten.mul.Tensor(sum_113, 3.985969387755102e-05)
        mul_837: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_838: "f32[896]" = torch.ops.aten.mul.Tensor(mul_836, mul_837);  mul_836 = mul_837 = None
        unsqueeze_698: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_838, 0);  mul_838 = None
        unsqueeze_699: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 2);  unsqueeze_698 = None
        unsqueeze_700: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 3);  unsqueeze_699 = None
        mul_839: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_178);  primals_178 = None
        unsqueeze_701: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(mul_839, 0);  mul_839 = None
        unsqueeze_702: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
        unsqueeze_703: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
        mul_840: "f32[32, 896, 28, 28]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_700);  sub_222 = unsqueeze_700 = None
        sub_224: "f32[32, 896, 28, 28]" = torch.ops.aten.sub.Tensor(where_47, mul_840);  where_47 = mul_840 = None
        sub_225: "f32[32, 896, 28, 28]" = torch.ops.aten.sub.Tensor(sub_224, unsqueeze_697);  sub_224 = unsqueeze_697 = None
        mul_841: "f32[32, 896, 28, 28]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_703);  sub_225 = unsqueeze_703 = None
        mul_842: "f32[896]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_73);  sum_113 = squeeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_841, relu_28, primals_174, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_841 = primals_174 = None
        getitem_307: "f32[32, 448, 28, 28]" = convolution_backward_61[0]
        getitem_308: "f32[896, 448, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
        add_352: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(getitem_292, getitem_307);  getitem_292 = getitem_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_48: "b8[32, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
        where_48: "f32[32, 448, 28, 28]" = torch.ops.aten.where.self(le_48, full_default, add_352);  le_48 = add_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_114: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
        sub_226: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_706);  convolution_37 = unsqueeze_706 = None
        mul_843: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_48, sub_226)
        sum_115: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_843, [0, 2, 3]);  mul_843 = None
        mul_844: "f32[448]" = torch.ops.aten.mul.Tensor(sum_114, 3.985969387755102e-05)
        unsqueeze_707: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_844, 0);  mul_844 = None
        unsqueeze_708: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 2);  unsqueeze_707 = None
        unsqueeze_709: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 3);  unsqueeze_708 = None
        mul_845: "f32[448]" = torch.ops.aten.mul.Tensor(sum_115, 3.985969387755102e-05)
        mul_846: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_847: "f32[448]" = torch.ops.aten.mul.Tensor(mul_845, mul_846);  mul_845 = mul_846 = None
        unsqueeze_710: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_847, 0);  mul_847 = None
        unsqueeze_711: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 2);  unsqueeze_710 = None
        unsqueeze_712: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 3);  unsqueeze_711 = None
        mul_848: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_172);  primals_172 = None
        unsqueeze_713: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_848, 0);  mul_848 = None
        unsqueeze_714: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
        unsqueeze_715: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
        mul_849: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_712);  sub_226 = unsqueeze_712 = None
        sub_228: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(where_48, mul_849);  mul_849 = None
        sub_229: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(sub_228, unsqueeze_709);  sub_228 = unsqueeze_709 = None
        mul_850: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_715);  sub_229 = unsqueeze_715 = None
        mul_851: "f32[448]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_70);  sum_115 = squeeze_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_850, mul_167, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_850 = mul_167 = primals_168 = None
        getitem_310: "f32[32, 448, 28, 28]" = convolution_backward_62[0]
        getitem_311: "f32[448, 448, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_22: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_45)
        mul_160: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        unsqueeze_88: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1)
        unsqueeze_89: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_166: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_89);  mul_160 = unsqueeze_89 = None
        unsqueeze_90: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_163, -1);  primals_163 = None
        unsqueeze_91: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_120: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_166, unsqueeze_91);  mul_166 = unsqueeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_26: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_120);  add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_852: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_310, relu_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_6: "f32[32, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36);  convolution_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_853: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_310, sigmoid_6);  getitem_310 = None
        sum_116: "f32[32, 448, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_852, [2, 3], True);  mul_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_230: "f32[32, 448, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_6)
        mul_854: "f32[32, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_6, sub_230);  sigmoid_6 = sub_230 = None
        mul_855: "f32[32, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sum_116, mul_854);  sum_116 = mul_854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_117: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_855, [0, 2, 3])
        convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_855, relu_27, primals_166, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_855 = primals_166 = None
        getitem_313: "f32[32, 112, 1, 1]" = convolution_backward_63[0]
        getitem_314: "f32[448, 112, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_49: "b8[32, 112, 1, 1]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        where_49: "f32[32, 112, 1, 1]" = torch.ops.aten.where.self(le_49, full_default, getitem_313);  le_49 = getitem_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_118: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
        convolution_backward_64 = torch.ops.aten.convolution_backward.default(where_49, mean_6, primals_164, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_49 = mean_6 = primals_164 = None
        getitem_316: "f32[32, 448, 1, 1]" = convolution_backward_64[0]
        getitem_317: "f32[112, 448, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_13: "f32[32, 448, 28, 28]" = torch.ops.aten.expand.default(getitem_316, [32, 448, 28, 28]);  getitem_316 = None
        div_13: "f32[32, 448, 28, 28]" = torch.ops.aten.div.Scalar(expand_13, 784);  expand_13 = None
        add_353: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_853, div_13);  mul_853 = div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_50: "b8[32, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        where_50: "f32[32, 448, 28, 28]" = torch.ops.aten.where.self(le_50, full_default, add_353);  le_50 = add_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_66: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        unsqueeze_716: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_717: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
        unsqueeze_718: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None
        sum_119: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
        sub_231: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_718);  convolution_34 = unsqueeze_718 = None
        mul_856: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_50, sub_231)
        sum_120: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_856, [0, 2, 3]);  mul_856 = None
        mul_857: "f32[448]" = torch.ops.aten.mul.Tensor(sum_119, 3.985969387755102e-05)
        unsqueeze_719: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_857, 0);  mul_857 = None
        unsqueeze_720: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 2);  unsqueeze_719 = None
        unsqueeze_721: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 3);  unsqueeze_720 = None
        mul_858: "f32[448]" = torch.ops.aten.mul.Tensor(sum_120, 3.985969387755102e-05)
        squeeze_67: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_859: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_860: "f32[448]" = torch.ops.aten.mul.Tensor(mul_858, mul_859);  mul_858 = mul_859 = None
        unsqueeze_722: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_860, 0);  mul_860 = None
        unsqueeze_723: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 2);  unsqueeze_722 = None
        unsqueeze_724: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 3);  unsqueeze_723 = None
        mul_861: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_162);  primals_162 = None
        unsqueeze_725: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_861, 0);  mul_861 = None
        unsqueeze_726: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
        unsqueeze_727: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
        mul_862: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_724);  sub_231 = unsqueeze_724 = None
        sub_233: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(where_50, mul_862);  where_50 = mul_862 = None
        sub_234: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(sub_233, unsqueeze_721);  sub_233 = unsqueeze_721 = None
        mul_863: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_727);  sub_234 = unsqueeze_727 = None
        mul_864: "f32[448]" = torch.ops.aten.mul.Tensor(sum_120, squeeze_67);  sum_120 = squeeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_863, relu_25, primals_158, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_863 = primals_158 = None
        getitem_319: "f32[32, 448, 28, 28]" = convolution_backward_65[0]
        getitem_320: "f32[448, 112, 3, 3]" = convolution_backward_65[1];  convolution_backward_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_51: "b8[32, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        where_51: "f32[32, 448, 28, 28]" = torch.ops.aten.where.self(le_51, full_default, getitem_319);  le_51 = getitem_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_121: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
        sub_235: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_730);  convolution_33 = unsqueeze_730 = None
        mul_865: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_51, sub_235)
        sum_122: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_865, [0, 2, 3]);  mul_865 = None
        mul_866: "f32[448]" = torch.ops.aten.mul.Tensor(sum_121, 3.985969387755102e-05)
        unsqueeze_731: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_866, 0);  mul_866 = None
        unsqueeze_732: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 2);  unsqueeze_731 = None
        unsqueeze_733: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 3);  unsqueeze_732 = None
        mul_867: "f32[448]" = torch.ops.aten.mul.Tensor(sum_122, 3.985969387755102e-05)
        mul_868: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_869: "f32[448]" = torch.ops.aten.mul.Tensor(mul_867, mul_868);  mul_867 = mul_868 = None
        unsqueeze_734: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_869, 0);  mul_869 = None
        unsqueeze_735: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 2);  unsqueeze_734 = None
        unsqueeze_736: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 3);  unsqueeze_735 = None
        mul_870: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_156);  primals_156 = None
        unsqueeze_737: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_870, 0);  mul_870 = None
        unsqueeze_738: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
        unsqueeze_739: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
        mul_871: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_736);  sub_235 = unsqueeze_736 = None
        sub_237: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(where_51, mul_871);  where_51 = mul_871 = None
        sub_238: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(sub_237, unsqueeze_733);  sub_237 = unsqueeze_733 = None
        mul_872: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_739);  sub_238 = unsqueeze_739 = None
        mul_873: "f32[448]" = torch.ops.aten.mul.Tensor(sum_122, squeeze_64);  sum_122 = squeeze_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_872, relu_24, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_872 = primals_152 = None
        getitem_322: "f32[32, 448, 28, 28]" = convolution_backward_66[0]
        getitem_323: "f32[448, 448, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
        add_354: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(where_48, getitem_322);  where_48 = getitem_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_52: "b8[32, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        where_52: "f32[32, 448, 28, 28]" = torch.ops.aten.where.self(le_52, full_default, add_354);  le_52 = add_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_123: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
        sub_239: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_742);  convolution_32 = unsqueeze_742 = None
        mul_874: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_52, sub_239)
        sum_124: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_874, [0, 2, 3]);  mul_874 = None
        mul_875: "f32[448]" = torch.ops.aten.mul.Tensor(sum_123, 3.985969387755102e-05)
        unsqueeze_743: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
        unsqueeze_744: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 2);  unsqueeze_743 = None
        unsqueeze_745: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 3);  unsqueeze_744 = None
        mul_876: "f32[448]" = torch.ops.aten.mul.Tensor(sum_124, 3.985969387755102e-05)
        mul_877: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_878: "f32[448]" = torch.ops.aten.mul.Tensor(mul_876, mul_877);  mul_876 = mul_877 = None
        unsqueeze_746: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_878, 0);  mul_878 = None
        unsqueeze_747: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 2);  unsqueeze_746 = None
        unsqueeze_748: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 3);  unsqueeze_747 = None
        mul_879: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_150);  primals_150 = None
        unsqueeze_749: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_879, 0);  mul_879 = None
        unsqueeze_750: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
        unsqueeze_751: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
        mul_880: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_748);  sub_239 = unsqueeze_748 = None
        sub_241: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(where_52, mul_880);  mul_880 = None
        sub_242: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(sub_241, unsqueeze_745);  sub_241 = unsqueeze_745 = None
        mul_881: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_751);  sub_242 = unsqueeze_751 = None
        mul_882: "f32[448]" = torch.ops.aten.mul.Tensor(sum_124, squeeze_61);  sum_124 = squeeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_881, mul_145, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_881 = mul_145 = primals_146 = None
        getitem_325: "f32[32, 448, 28, 28]" = convolution_backward_67[0]
        getitem_326: "f32[448, 448, 1, 1]" = convolution_backward_67[1];  convolution_backward_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_19: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_39)
        mul_138: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        unsqueeze_76: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
        unsqueeze_77: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_144: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_77);  mul_138 = unsqueeze_77 = None
        unsqueeze_78: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
        unsqueeze_79: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_104: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_79);  mul_144 = unsqueeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_22: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_104);  add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_883: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_325, relu_22)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_5: "f32[32, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_31);  convolution_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_884: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_325, sigmoid_5);  getitem_325 = None
        sum_125: "f32[32, 448, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_883, [2, 3], True);  mul_883 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_243: "f32[32, 448, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_5)
        mul_885: "f32[32, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_5, sub_243);  sigmoid_5 = sub_243 = None
        mul_886: "f32[32, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sum_125, mul_885);  sum_125 = mul_885 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_126: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_886, [0, 2, 3])
        convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_886, relu_23, primals_144, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_886 = primals_144 = None
        getitem_328: "f32[32, 112, 1, 1]" = convolution_backward_68[0]
        getitem_329: "f32[448, 112, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_53: "b8[32, 112, 1, 1]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        where_53: "f32[32, 112, 1, 1]" = torch.ops.aten.where.self(le_53, full_default, getitem_328);  le_53 = getitem_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_127: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
        convolution_backward_69 = torch.ops.aten.convolution_backward.default(where_53, mean_5, primals_142, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_53 = mean_5 = primals_142 = None
        getitem_331: "f32[32, 448, 1, 1]" = convolution_backward_69[0]
        getitem_332: "f32[112, 448, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_14: "f32[32, 448, 28, 28]" = torch.ops.aten.expand.default(getitem_331, [32, 448, 28, 28]);  getitem_331 = None
        div_14: "f32[32, 448, 28, 28]" = torch.ops.aten.div.Scalar(expand_14, 784);  expand_14 = None
        add_355: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_884, div_14);  mul_884 = div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_54: "b8[32, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_54: "f32[32, 448, 28, 28]" = torch.ops.aten.where.self(le_54, full_default, add_355);  le_54 = add_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_57: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        unsqueeze_752: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_753: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
        unsqueeze_754: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None
        sum_128: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
        sub_244: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_754);  convolution_29 = unsqueeze_754 = None
        mul_887: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_54, sub_244)
        sum_129: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_887, [0, 2, 3]);  mul_887 = None
        mul_888: "f32[448]" = torch.ops.aten.mul.Tensor(sum_128, 3.985969387755102e-05)
        unsqueeze_755: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_888, 0);  mul_888 = None
        unsqueeze_756: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 2);  unsqueeze_755 = None
        unsqueeze_757: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 3);  unsqueeze_756 = None
        mul_889: "f32[448]" = torch.ops.aten.mul.Tensor(sum_129, 3.985969387755102e-05)
        squeeze_58: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_890: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_891: "f32[448]" = torch.ops.aten.mul.Tensor(mul_889, mul_890);  mul_889 = mul_890 = None
        unsqueeze_758: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_891, 0);  mul_891 = None
        unsqueeze_759: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 2);  unsqueeze_758 = None
        unsqueeze_760: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 3);  unsqueeze_759 = None
        mul_892: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_140);  primals_140 = None
        unsqueeze_761: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_892, 0);  mul_892 = None
        unsqueeze_762: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
        unsqueeze_763: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
        mul_893: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_760);  sub_244 = unsqueeze_760 = None
        sub_246: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(where_54, mul_893);  where_54 = mul_893 = None
        sub_247: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_757);  sub_246 = unsqueeze_757 = None
        mul_894: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_763);  sub_247 = unsqueeze_763 = None
        mul_895: "f32[448]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_58);  sum_129 = squeeze_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_894, relu_21, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_894 = primals_136 = None
        getitem_334: "f32[32, 448, 28, 28]" = convolution_backward_70[0]
        getitem_335: "f32[448, 112, 3, 3]" = convolution_backward_70[1];  convolution_backward_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_55: "b8[32, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        where_55: "f32[32, 448, 28, 28]" = torch.ops.aten.where.self(le_55, full_default, getitem_334);  le_55 = getitem_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_130: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
        sub_248: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_766);  convolution_28 = unsqueeze_766 = None
        mul_896: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_55, sub_248)
        sum_131: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_896, [0, 2, 3]);  mul_896 = None
        mul_897: "f32[448]" = torch.ops.aten.mul.Tensor(sum_130, 3.985969387755102e-05)
        unsqueeze_767: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_897, 0);  mul_897 = None
        unsqueeze_768: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 2);  unsqueeze_767 = None
        unsqueeze_769: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 3);  unsqueeze_768 = None
        mul_898: "f32[448]" = torch.ops.aten.mul.Tensor(sum_131, 3.985969387755102e-05)
        mul_899: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_900: "f32[448]" = torch.ops.aten.mul.Tensor(mul_898, mul_899);  mul_898 = mul_899 = None
        unsqueeze_770: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_900, 0);  mul_900 = None
        unsqueeze_771: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 2);  unsqueeze_770 = None
        unsqueeze_772: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 3);  unsqueeze_771 = None
        mul_901: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_134);  primals_134 = None
        unsqueeze_773: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_901, 0);  mul_901 = None
        unsqueeze_774: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
        unsqueeze_775: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
        mul_902: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_772);  sub_248 = unsqueeze_772 = None
        sub_250: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(where_55, mul_902);  where_55 = mul_902 = None
        sub_251: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_769);  sub_250 = unsqueeze_769 = None
        mul_903: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_775);  sub_251 = unsqueeze_775 = None
        mul_904: "f32[448]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_55);  sum_131 = squeeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_903, relu_20, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_903 = primals_130 = None
        getitem_337: "f32[32, 448, 28, 28]" = convolution_backward_71[0]
        getitem_338: "f32[448, 448, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
        add_356: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(where_52, getitem_337);  where_52 = getitem_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_56: "b8[32, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_56: "f32[32, 448, 28, 28]" = torch.ops.aten.where.self(le_56, full_default, add_356);  le_56 = add_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_132: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
        sub_252: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_778);  convolution_27 = unsqueeze_778 = None
        mul_905: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_56, sub_252)
        sum_133: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_905, [0, 2, 3]);  mul_905 = None
        mul_906: "f32[448]" = torch.ops.aten.mul.Tensor(sum_132, 3.985969387755102e-05)
        unsqueeze_779: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_906, 0);  mul_906 = None
        unsqueeze_780: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 2);  unsqueeze_779 = None
        unsqueeze_781: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 3);  unsqueeze_780 = None
        mul_907: "f32[448]" = torch.ops.aten.mul.Tensor(sum_133, 3.985969387755102e-05)
        mul_908: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_909: "f32[448]" = torch.ops.aten.mul.Tensor(mul_907, mul_908);  mul_907 = mul_908 = None
        unsqueeze_782: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_909, 0);  mul_909 = None
        unsqueeze_783: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 2);  unsqueeze_782 = None
        unsqueeze_784: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 3);  unsqueeze_783 = None
        mul_910: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_128);  primals_128 = None
        unsqueeze_785: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_910, 0);  mul_910 = None
        unsqueeze_786: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
        unsqueeze_787: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
        mul_911: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_784);  sub_252 = unsqueeze_784 = None
        sub_254: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(where_56, mul_911);  mul_911 = None
        sub_255: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_781);  sub_254 = unsqueeze_781 = None
        mul_912: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_787);  sub_255 = unsqueeze_787 = None
        mul_913: "f32[448]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_52);  sum_133 = squeeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_912, mul_123, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_912 = mul_123 = primals_124 = None
        getitem_340: "f32[32, 448, 28, 28]" = convolution_backward_72[0]
        getitem_341: "f32[448, 448, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_16: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_33)
        mul_116: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        unsqueeze_64: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1)
        unsqueeze_65: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_122: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_116, unsqueeze_65);  mul_116 = unsqueeze_65 = None
        unsqueeze_66: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1);  primals_119 = None
        unsqueeze_67: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_88: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_67);  mul_122 = unsqueeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_18: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_88);  add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_914: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_340, relu_18)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_4: "f32[32, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_26);  convolution_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_915: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_340, sigmoid_4);  getitem_340 = None
        sum_134: "f32[32, 448, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_914, [2, 3], True);  mul_914 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_256: "f32[32, 448, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_4)
        mul_916: "f32[32, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_4, sub_256);  sigmoid_4 = sub_256 = None
        mul_917: "f32[32, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sum_134, mul_916);  sum_134 = mul_916 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_135: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_917, [0, 2, 3])
        convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_917, relu_19, primals_122, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_917 = primals_122 = None
        getitem_343: "f32[32, 112, 1, 1]" = convolution_backward_73[0]
        getitem_344: "f32[448, 112, 1, 1]" = convolution_backward_73[1];  convolution_backward_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_57: "b8[32, 112, 1, 1]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_57: "f32[32, 112, 1, 1]" = torch.ops.aten.where.self(le_57, full_default, getitem_343);  le_57 = getitem_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_136: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
        convolution_backward_74 = torch.ops.aten.convolution_backward.default(where_57, mean_4, primals_120, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_57 = mean_4 = primals_120 = None
        getitem_346: "f32[32, 448, 1, 1]" = convolution_backward_74[0]
        getitem_347: "f32[112, 448, 1, 1]" = convolution_backward_74[1];  convolution_backward_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_15: "f32[32, 448, 28, 28]" = torch.ops.aten.expand.default(getitem_346, [32, 448, 28, 28]);  getitem_346 = None
        div_15: "f32[32, 448, 28, 28]" = torch.ops.aten.div.Scalar(expand_15, 784);  expand_15 = None
        add_357: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_915, div_15);  mul_915 = div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_58: "b8[32, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        where_58: "f32[32, 448, 28, 28]" = torch.ops.aten.where.self(le_58, full_default, add_357);  le_58 = add_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_48: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        unsqueeze_788: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_789: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
        unsqueeze_790: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None
        sum_137: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
        sub_257: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_790);  convolution_24 = unsqueeze_790 = None
        mul_918: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_58, sub_257)
        sum_138: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_918, [0, 2, 3]);  mul_918 = None
        mul_919: "f32[448]" = torch.ops.aten.mul.Tensor(sum_137, 3.985969387755102e-05)
        unsqueeze_791: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_919, 0);  mul_919 = None
        unsqueeze_792: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 2);  unsqueeze_791 = None
        unsqueeze_793: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 3);  unsqueeze_792 = None
        mul_920: "f32[448]" = torch.ops.aten.mul.Tensor(sum_138, 3.985969387755102e-05)
        squeeze_49: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_921: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_922: "f32[448]" = torch.ops.aten.mul.Tensor(mul_920, mul_921);  mul_920 = mul_921 = None
        unsqueeze_794: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_922, 0);  mul_922 = None
        unsqueeze_795: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 2);  unsqueeze_794 = None
        unsqueeze_796: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 3);  unsqueeze_795 = None
        mul_923: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_118);  primals_118 = None
        unsqueeze_797: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_923, 0);  mul_923 = None
        unsqueeze_798: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
        unsqueeze_799: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
        mul_924: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_796);  sub_257 = unsqueeze_796 = None
        sub_259: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(where_58, mul_924);  where_58 = mul_924 = None
        sub_260: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_793);  sub_259 = unsqueeze_793 = None
        mul_925: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_799);  sub_260 = unsqueeze_799 = None
        mul_926: "f32[448]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_49);  sum_138 = squeeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_925, relu_17, primals_114, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_925 = primals_114 = None
        getitem_349: "f32[32, 448, 28, 28]" = convolution_backward_75[0]
        getitem_350: "f32[448, 112, 3, 3]" = convolution_backward_75[1];  convolution_backward_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_59: "b8[32, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_59: "f32[32, 448, 28, 28]" = torch.ops.aten.where.self(le_59, full_default, getitem_349);  le_59 = getitem_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_139: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
        sub_261: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_802);  convolution_23 = unsqueeze_802 = None
        mul_927: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_59, sub_261)
        sum_140: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_927, [0, 2, 3]);  mul_927 = None
        mul_928: "f32[448]" = torch.ops.aten.mul.Tensor(sum_139, 3.985969387755102e-05)
        unsqueeze_803: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_928, 0);  mul_928 = None
        unsqueeze_804: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 2);  unsqueeze_803 = None
        unsqueeze_805: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 3);  unsqueeze_804 = None
        mul_929: "f32[448]" = torch.ops.aten.mul.Tensor(sum_140, 3.985969387755102e-05)
        mul_930: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_931: "f32[448]" = torch.ops.aten.mul.Tensor(mul_929, mul_930);  mul_929 = mul_930 = None
        unsqueeze_806: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_931, 0);  mul_931 = None
        unsqueeze_807: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 2);  unsqueeze_806 = None
        unsqueeze_808: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 3);  unsqueeze_807 = None
        mul_932: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_112);  primals_112 = None
        unsqueeze_809: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
        unsqueeze_810: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
        unsqueeze_811: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
        mul_933: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_808);  sub_261 = unsqueeze_808 = None
        sub_263: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(where_59, mul_933);  where_59 = mul_933 = None
        sub_264: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_805);  sub_263 = unsqueeze_805 = None
        mul_934: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_811);  sub_264 = unsqueeze_811 = None
        mul_935: "f32[448]" = torch.ops.aten.mul.Tensor(sum_140, squeeze_46);  sum_140 = squeeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_934, relu_16, primals_108, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_934 = primals_108 = None
        getitem_352: "f32[32, 448, 28, 28]" = convolution_backward_76[0]
        getitem_353: "f32[448, 448, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
        add_358: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(where_56, getitem_352);  where_56 = getitem_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_60: "b8[32, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_60: "f32[32, 448, 28, 28]" = torch.ops.aten.where.self(le_60, full_default, add_358);  le_60 = add_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_141: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
        sub_265: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_814);  convolution_22 = unsqueeze_814 = None
        mul_936: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_60, sub_265)
        sum_142: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_936, [0, 2, 3]);  mul_936 = None
        mul_937: "f32[448]" = torch.ops.aten.mul.Tensor(sum_141, 3.985969387755102e-05)
        unsqueeze_815: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_937, 0);  mul_937 = None
        unsqueeze_816: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 2);  unsqueeze_815 = None
        unsqueeze_817: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 3);  unsqueeze_816 = None
        mul_938: "f32[448]" = torch.ops.aten.mul.Tensor(sum_142, 3.985969387755102e-05)
        mul_939: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_940: "f32[448]" = torch.ops.aten.mul.Tensor(mul_938, mul_939);  mul_938 = mul_939 = None
        unsqueeze_818: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_940, 0);  mul_940 = None
        unsqueeze_819: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 2);  unsqueeze_818 = None
        unsqueeze_820: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 3);  unsqueeze_819 = None
        mul_941: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_106);  primals_106 = None
        unsqueeze_821: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_941, 0);  mul_941 = None
        unsqueeze_822: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
        unsqueeze_823: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
        mul_942: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_820);  sub_265 = unsqueeze_820 = None
        sub_267: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(where_60, mul_942);  mul_942 = None
        sub_268: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_817);  sub_267 = unsqueeze_817 = None
        mul_943: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_823);  sub_268 = unsqueeze_823 = None
        mul_944: "f32[448]" = torch.ops.aten.mul.Tensor(sum_142, squeeze_43);  sum_142 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_943, mul_101, primals_102, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_943 = mul_101 = primals_102 = None
        getitem_355: "f32[32, 448, 28, 28]" = convolution_backward_77[0]
        getitem_356: "f32[448, 448, 1, 1]" = convolution_backward_77[1];  convolution_backward_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_13: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_27)
        mul_94: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        unsqueeze_52: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1)
        unsqueeze_53: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_100: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_53);  mul_94 = unsqueeze_53 = None
        unsqueeze_54: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1);  primals_97 = None
        unsqueeze_55: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_72: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_100, unsqueeze_55);  mul_100 = unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_14: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_72);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_945: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_355, relu_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_3: "f32[32, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_21);  convolution_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_946: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_355, sigmoid_3);  getitem_355 = None
        sum_143: "f32[32, 448, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_945, [2, 3], True);  mul_945 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_269: "f32[32, 448, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_3)
        mul_947: "f32[32, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_3, sub_269);  sigmoid_3 = sub_269 = None
        mul_948: "f32[32, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sum_143, mul_947);  sum_143 = mul_947 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_144: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_948, [0, 2, 3])
        convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_948, relu_15, primals_100, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_948 = primals_100 = None
        getitem_358: "f32[32, 112, 1, 1]" = convolution_backward_78[0]
        getitem_359: "f32[448, 112, 1, 1]" = convolution_backward_78[1];  convolution_backward_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_61: "b8[32, 112, 1, 1]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_61: "f32[32, 112, 1, 1]" = torch.ops.aten.where.self(le_61, full_default, getitem_358);  le_61 = getitem_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_145: "f32[112]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
        convolution_backward_79 = torch.ops.aten.convolution_backward.default(where_61, mean_3, primals_98, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_61 = mean_3 = primals_98 = None
        getitem_361: "f32[32, 448, 1, 1]" = convolution_backward_79[0]
        getitem_362: "f32[112, 448, 1, 1]" = convolution_backward_79[1];  convolution_backward_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_16: "f32[32, 448, 28, 28]" = torch.ops.aten.expand.default(getitem_361, [32, 448, 28, 28]);  getitem_361 = None
        div_16: "f32[32, 448, 28, 28]" = torch.ops.aten.div.Scalar(expand_16, 784);  expand_16 = None
        add_359: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_946, div_16);  mul_946 = div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_62: "b8[32, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_62: "f32[32, 448, 28, 28]" = torch.ops.aten.where.self(le_62, full_default, add_359);  le_62 = add_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_39: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        unsqueeze_824: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_825: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
        unsqueeze_826: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None
        sum_146: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
        sub_270: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_826);  convolution_19 = unsqueeze_826 = None
        mul_949: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_62, sub_270)
        sum_147: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_949, [0, 2, 3]);  mul_949 = None
        mul_950: "f32[448]" = torch.ops.aten.mul.Tensor(sum_146, 3.985969387755102e-05)
        unsqueeze_827: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_950, 0);  mul_950 = None
        unsqueeze_828: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 2);  unsqueeze_827 = None
        unsqueeze_829: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 3);  unsqueeze_828 = None
        mul_951: "f32[448]" = torch.ops.aten.mul.Tensor(sum_147, 3.985969387755102e-05)
        squeeze_40: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_952: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_953: "f32[448]" = torch.ops.aten.mul.Tensor(mul_951, mul_952);  mul_951 = mul_952 = None
        unsqueeze_830: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_953, 0);  mul_953 = None
        unsqueeze_831: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 2);  unsqueeze_830 = None
        unsqueeze_832: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 3);  unsqueeze_831 = None
        mul_954: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_96);  primals_96 = None
        unsqueeze_833: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
        unsqueeze_834: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
        unsqueeze_835: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
        mul_955: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_832);  sub_270 = unsqueeze_832 = None
        sub_272: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(where_62, mul_955);  where_62 = mul_955 = None
        sub_273: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(sub_272, unsqueeze_829);  sub_272 = unsqueeze_829 = None
        mul_956: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_835);  sub_273 = unsqueeze_835 = None
        mul_957: "f32[448]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_40);  sum_147 = squeeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_956, relu_13, primals_92, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_956 = primals_92 = None
        getitem_364: "f32[32, 448, 28, 28]" = convolution_backward_80[0]
        getitem_365: "f32[448, 112, 3, 3]" = convolution_backward_80[1];  convolution_backward_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_63: "b8[32, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_63: "f32[32, 448, 28, 28]" = torch.ops.aten.where.self(le_63, full_default, getitem_364);  le_63 = getitem_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_148: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
        sub_274: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_838);  convolution_18 = unsqueeze_838 = None
        mul_958: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_63, sub_274)
        sum_149: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_958, [0, 2, 3]);  mul_958 = None
        mul_959: "f32[448]" = torch.ops.aten.mul.Tensor(sum_148, 3.985969387755102e-05)
        unsqueeze_839: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_959, 0);  mul_959 = None
        unsqueeze_840: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 2);  unsqueeze_839 = None
        unsqueeze_841: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 3);  unsqueeze_840 = None
        mul_960: "f32[448]" = torch.ops.aten.mul.Tensor(sum_149, 3.985969387755102e-05)
        mul_961: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_962: "f32[448]" = torch.ops.aten.mul.Tensor(mul_960, mul_961);  mul_960 = mul_961 = None
        unsqueeze_842: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_962, 0);  mul_962 = None
        unsqueeze_843: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 2);  unsqueeze_842 = None
        unsqueeze_844: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 3);  unsqueeze_843 = None
        mul_963: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_90);  primals_90 = None
        unsqueeze_845: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
        unsqueeze_846: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
        unsqueeze_847: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
        mul_964: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_844);  sub_274 = unsqueeze_844 = None
        sub_276: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(where_63, mul_964);  where_63 = mul_964 = None
        sub_277: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(sub_276, unsqueeze_841);  sub_276 = unsqueeze_841 = None
        mul_965: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_847);  sub_277 = unsqueeze_847 = None
        mul_966: "f32[448]" = torch.ops.aten.mul.Tensor(sum_149, squeeze_37);  sum_149 = squeeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_965, relu_12, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_965 = primals_86 = None
        getitem_367: "f32[32, 448, 28, 28]" = convolution_backward_81[0]
        getitem_368: "f32[448, 448, 1, 1]" = convolution_backward_81[1];  convolution_backward_81 = None
        add_360: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(where_60, getitem_367);  where_60 = getitem_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_64: "b8[32, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_64: "f32[32, 448, 28, 28]" = torch.ops.aten.where.self(le_64, full_default, add_360);  le_64 = add_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_150: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
        sub_278: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_850);  convolution_17 = unsqueeze_850 = None
        mul_967: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_64, sub_278)
        sum_151: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_967, [0, 2, 3]);  mul_967 = None
        mul_968: "f32[448]" = torch.ops.aten.mul.Tensor(sum_150, 3.985969387755102e-05)
        unsqueeze_851: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_968, 0);  mul_968 = None
        unsqueeze_852: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 2);  unsqueeze_851 = None
        unsqueeze_853: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 3);  unsqueeze_852 = None
        mul_969: "f32[448]" = torch.ops.aten.mul.Tensor(sum_151, 3.985969387755102e-05)
        mul_970: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_971: "f32[448]" = torch.ops.aten.mul.Tensor(mul_969, mul_970);  mul_969 = mul_970 = None
        unsqueeze_854: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_971, 0);  mul_971 = None
        unsqueeze_855: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 2);  unsqueeze_854 = None
        unsqueeze_856: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 3);  unsqueeze_855 = None
        mul_972: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_84);  primals_84 = None
        unsqueeze_857: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
        unsqueeze_858: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
        unsqueeze_859: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
        mul_973: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_856);  sub_278 = unsqueeze_856 = None
        sub_280: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(where_64, mul_973);  mul_973 = None
        sub_281: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(sub_280, unsqueeze_853);  sub_280 = unsqueeze_853 = None
        mul_974: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_859);  sub_281 = unsqueeze_859 = None
        mul_975: "f32[448]" = torch.ops.aten.mul.Tensor(sum_151, squeeze_34);  sum_151 = squeeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_974, relu_8, primals_80, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_974 = primals_80 = None
        getitem_370: "f32[32, 224, 56, 56]" = convolution_backward_82[0]
        getitem_371: "f32[448, 224, 1, 1]" = convolution_backward_82[1];  convolution_backward_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_152: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
        sub_282: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_862);  convolution_16 = unsqueeze_862 = None
        mul_976: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_64, sub_282)
        sum_153: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_976, [0, 2, 3]);  mul_976 = None
        mul_977: "f32[448]" = torch.ops.aten.mul.Tensor(sum_152, 3.985969387755102e-05)
        unsqueeze_863: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_977, 0);  mul_977 = None
        unsqueeze_864: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 2);  unsqueeze_863 = None
        unsqueeze_865: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 3);  unsqueeze_864 = None
        mul_978: "f32[448]" = torch.ops.aten.mul.Tensor(sum_153, 3.985969387755102e-05)
        mul_979: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_980: "f32[448]" = torch.ops.aten.mul.Tensor(mul_978, mul_979);  mul_978 = mul_979 = None
        unsqueeze_866: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_980, 0);  mul_980 = None
        unsqueeze_867: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 2);  unsqueeze_866 = None
        unsqueeze_868: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 3);  unsqueeze_867 = None
        mul_981: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_78);  primals_78 = None
        unsqueeze_869: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_981, 0);  mul_981 = None
        unsqueeze_870: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
        unsqueeze_871: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
        mul_982: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_868);  sub_282 = unsqueeze_868 = None
        sub_284: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(where_64, mul_982);  where_64 = mul_982 = None
        sub_285: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(sub_284, unsqueeze_865);  sub_284 = unsqueeze_865 = None
        mul_983: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_871);  sub_285 = unsqueeze_871 = None
        mul_984: "f32[448]" = torch.ops.aten.mul.Tensor(sum_153, squeeze_31);  sum_153 = squeeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_983, mul_72, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_983 = mul_72 = primals_74 = None
        getitem_373: "f32[32, 448, 28, 28]" = convolution_backward_83[0]
        getitem_374: "f32[448, 448, 1, 1]" = convolution_backward_83[1];  convolution_backward_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_9: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_19)
        mul_65: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        unsqueeze_36: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
        unsqueeze_37: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_71: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_65, unsqueeze_37);  mul_65 = unsqueeze_37 = None
        unsqueeze_38: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
        unsqueeze_39: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_51: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_39);  mul_71 = unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_10: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_51);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_985: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_373, relu_10)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_2: "f32[32, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_15);  convolution_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_986: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_373, sigmoid_2);  getitem_373 = None
        sum_154: "f32[32, 448, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_985, [2, 3], True);  mul_985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_286: "f32[32, 448, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_2)
        mul_987: "f32[32, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_2, sub_286);  sigmoid_2 = sub_286 = None
        mul_988: "f32[32, 448, 1, 1]" = torch.ops.aten.mul.Tensor(sum_154, mul_987);  sum_154 = mul_987 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_155: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_988, [0, 2, 3])
        convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_988, relu_11, primals_72, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_988 = primals_72 = None
        getitem_376: "f32[32, 56, 1, 1]" = convolution_backward_84[0]
        getitem_377: "f32[448, 56, 1, 1]" = convolution_backward_84[1];  convolution_backward_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_65: "b8[32, 56, 1, 1]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_65: "f32[32, 56, 1, 1]" = torch.ops.aten.where.self(le_65, full_default, getitem_376);  le_65 = getitem_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_156: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
        convolution_backward_85 = torch.ops.aten.convolution_backward.default(where_65, mean_2, primals_70, [56], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_65 = mean_2 = primals_70 = None
        getitem_379: "f32[32, 448, 1, 1]" = convolution_backward_85[0]
        getitem_380: "f32[56, 448, 1, 1]" = convolution_backward_85[1];  convolution_backward_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_17: "f32[32, 448, 28, 28]" = torch.ops.aten.expand.default(getitem_379, [32, 448, 28, 28]);  getitem_379 = None
        div_17: "f32[32, 448, 28, 28]" = torch.ops.aten.div.Scalar(expand_17, 784);  expand_17 = None
        add_361: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_986, div_17);  mul_986 = div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_66: "b8[32, 448, 28, 28]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_66: "f32[32, 448, 28, 28]" = torch.ops.aten.where.self(le_66, full_default, add_361);  le_66 = add_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_27: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        unsqueeze_872: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_873: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 2);  unsqueeze_872 = None
        unsqueeze_874: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 3);  unsqueeze_873 = None
        sum_157: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
        sub_287: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_874);  convolution_13 = unsqueeze_874 = None
        mul_989: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(where_66, sub_287)
        sum_158: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_989, [0, 2, 3]);  mul_989 = None
        mul_990: "f32[448]" = torch.ops.aten.mul.Tensor(sum_157, 3.985969387755102e-05)
        unsqueeze_875: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_990, 0);  mul_990 = None
        unsqueeze_876: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 2);  unsqueeze_875 = None
        unsqueeze_877: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 3);  unsqueeze_876 = None
        mul_991: "f32[448]" = torch.ops.aten.mul.Tensor(sum_158, 3.985969387755102e-05)
        squeeze_28: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_992: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_993: "f32[448]" = torch.ops.aten.mul.Tensor(mul_991, mul_992);  mul_991 = mul_992 = None
        unsqueeze_878: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_993, 0);  mul_993 = None
        unsqueeze_879: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 2);  unsqueeze_878 = None
        unsqueeze_880: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 3);  unsqueeze_879 = None
        mul_994: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_68);  primals_68 = None
        unsqueeze_881: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_994, 0);  mul_994 = None
        unsqueeze_882: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
        unsqueeze_883: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
        mul_995: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_880);  sub_287 = unsqueeze_880 = None
        sub_289: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(where_66, mul_995);  where_66 = mul_995 = None
        sub_290: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(sub_289, unsqueeze_877);  sub_289 = unsqueeze_877 = None
        mul_996: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_883);  sub_290 = unsqueeze_883 = None
        mul_997: "f32[448]" = torch.ops.aten.mul.Tensor(sum_158, squeeze_28);  sum_158 = squeeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_996, relu_9, primals_64, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False]);  mul_996 = primals_64 = None
        getitem_382: "f32[32, 448, 56, 56]" = convolution_backward_86[0]
        getitem_383: "f32[448, 112, 3, 3]" = convolution_backward_86[1];  convolution_backward_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_67: "b8[32, 448, 56, 56]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_67: "f32[32, 448, 56, 56]" = torch.ops.aten.where.self(le_67, full_default, getitem_382);  le_67 = getitem_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_159: "f32[448]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
        sub_291: "f32[32, 448, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_886);  convolution_12 = unsqueeze_886 = None
        mul_998: "f32[32, 448, 56, 56]" = torch.ops.aten.mul.Tensor(where_67, sub_291)
        sum_160: "f32[448]" = torch.ops.aten.sum.dim_IntList(mul_998, [0, 2, 3]);  mul_998 = None
        mul_999: "f32[448]" = torch.ops.aten.mul.Tensor(sum_159, 9.964923469387754e-06)
        unsqueeze_887: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_999, 0);  mul_999 = None
        unsqueeze_888: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 2);  unsqueeze_887 = None
        unsqueeze_889: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 3);  unsqueeze_888 = None
        mul_1000: "f32[448]" = torch.ops.aten.mul.Tensor(sum_160, 9.964923469387754e-06)
        mul_1001: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_1002: "f32[448]" = torch.ops.aten.mul.Tensor(mul_1000, mul_1001);  mul_1000 = mul_1001 = None
        unsqueeze_890: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1002, 0);  mul_1002 = None
        unsqueeze_891: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 2);  unsqueeze_890 = None
        unsqueeze_892: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 3);  unsqueeze_891 = None
        mul_1003: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_62);  primals_62 = None
        unsqueeze_893: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(mul_1003, 0);  mul_1003 = None
        unsqueeze_894: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
        unsqueeze_895: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
        mul_1004: "f32[32, 448, 56, 56]" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_892);  sub_291 = unsqueeze_892 = None
        sub_293: "f32[32, 448, 56, 56]" = torch.ops.aten.sub.Tensor(where_67, mul_1004);  where_67 = mul_1004 = None
        sub_294: "f32[32, 448, 56, 56]" = torch.ops.aten.sub.Tensor(sub_293, unsqueeze_889);  sub_293 = unsqueeze_889 = None
        mul_1005: "f32[32, 448, 56, 56]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_895);  sub_294 = unsqueeze_895 = None
        mul_1006: "f32[448]" = torch.ops.aten.mul.Tensor(sum_160, squeeze_25);  sum_160 = squeeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_1005, relu_8, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1005 = primals_58 = None
        getitem_385: "f32[32, 224, 56, 56]" = convolution_backward_87[0]
        getitem_386: "f32[448, 224, 1, 1]" = convolution_backward_87[1];  convolution_backward_87 = None
        add_362: "f32[32, 224, 56, 56]" = torch.ops.aten.add.Tensor(getitem_370, getitem_385);  getitem_370 = getitem_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_68: "b8[32, 224, 56, 56]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_68: "f32[32, 224, 56, 56]" = torch.ops.aten.where.self(le_68, full_default, add_362);  le_68 = add_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_161: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
        sub_295: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_898);  convolution_11 = unsqueeze_898 = None
        mul_1007: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_68, sub_295)
        sum_162: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_1007, [0, 2, 3]);  mul_1007 = None
        mul_1008: "f32[224]" = torch.ops.aten.mul.Tensor(sum_161, 9.964923469387754e-06)
        unsqueeze_899: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1008, 0);  mul_1008 = None
        unsqueeze_900: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 2);  unsqueeze_899 = None
        unsqueeze_901: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 3);  unsqueeze_900 = None
        mul_1009: "f32[224]" = torch.ops.aten.mul.Tensor(sum_162, 9.964923469387754e-06)
        mul_1010: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_1011: "f32[224]" = torch.ops.aten.mul.Tensor(mul_1009, mul_1010);  mul_1009 = mul_1010 = None
        unsqueeze_902: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1011, 0);  mul_1011 = None
        unsqueeze_903: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 2);  unsqueeze_902 = None
        unsqueeze_904: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 3);  unsqueeze_903 = None
        mul_1012: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_56);  primals_56 = None
        unsqueeze_905: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1012, 0);  mul_1012 = None
        unsqueeze_906: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
        unsqueeze_907: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
        mul_1013: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_904);  sub_295 = unsqueeze_904 = None
        sub_297: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(where_68, mul_1013);  mul_1013 = None
        sub_298: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(sub_297, unsqueeze_901);  sub_297 = unsqueeze_901 = None
        mul_1014: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_907);  sub_298 = unsqueeze_907 = None
        mul_1015: "f32[224]" = torch.ops.aten.mul.Tensor(sum_162, squeeze_22);  sum_162 = squeeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_1014, mul_50, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1014 = mul_50 = primals_52 = None
        getitem_388: "f32[32, 224, 56, 56]" = convolution_backward_88[0]
        getitem_389: "f32[224, 224, 1, 1]" = convolution_backward_88[1];  convolution_backward_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_6: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_13)
        mul_43: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        unsqueeze_24: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1)
        unsqueeze_25: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_49: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_25);  mul_43 = unsqueeze_25 = None
        unsqueeze_26: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1);  primals_47 = None
        unsqueeze_27: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_35: "f32[32, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_49, unsqueeze_27);  mul_49 = unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_6: "f32[32, 224, 56, 56]" = torch.ops.aten.relu.default(add_35);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_1016: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_388, relu_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_1: "f32[32, 224, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_1017: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_388, sigmoid_1);  getitem_388 = None
        sum_163: "f32[32, 224, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1016, [2, 3], True);  mul_1016 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_299: "f32[32, 224, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_1)
        mul_1018: "f32[32, 224, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_1, sub_299);  sigmoid_1 = sub_299 = None
        mul_1019: "f32[32, 224, 1, 1]" = torch.ops.aten.mul.Tensor(sum_163, mul_1018);  sum_163 = mul_1018 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_164: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_1019, [0, 2, 3])
        convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_1019, relu_7, primals_50, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1019 = primals_50 = None
        getitem_391: "f32[32, 56, 1, 1]" = convolution_backward_89[0]
        getitem_392: "f32[224, 56, 1, 1]" = convolution_backward_89[1];  convolution_backward_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_69: "b8[32, 56, 1, 1]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_69: "f32[32, 56, 1, 1]" = torch.ops.aten.where.self(le_69, full_default, getitem_391);  le_69 = getitem_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_165: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
        convolution_backward_90 = torch.ops.aten.convolution_backward.default(where_69, mean_1, primals_48, [56], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_69 = mean_1 = primals_48 = None
        getitem_394: "f32[32, 224, 1, 1]" = convolution_backward_90[0]
        getitem_395: "f32[56, 224, 1, 1]" = convolution_backward_90[1];  convolution_backward_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_18: "f32[32, 224, 56, 56]" = torch.ops.aten.expand.default(getitem_394, [32, 224, 56, 56]);  getitem_394 = None
        div_18: "f32[32, 224, 56, 56]" = torch.ops.aten.div.Scalar(expand_18, 3136);  expand_18 = None
        add_363: "f32[32, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_1017, div_18);  mul_1017 = div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_70: "b8[32, 224, 56, 56]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_70: "f32[32, 224, 56, 56]" = torch.ops.aten.where.self(le_70, full_default, add_363);  le_70 = add_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_18: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        unsqueeze_908: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_909: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 2);  unsqueeze_908 = None
        unsqueeze_910: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 3);  unsqueeze_909 = None
        sum_166: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
        sub_300: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_910);  convolution_8 = unsqueeze_910 = None
        mul_1020: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_70, sub_300)
        sum_167: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_1020, [0, 2, 3]);  mul_1020 = None
        mul_1021: "f32[224]" = torch.ops.aten.mul.Tensor(sum_166, 9.964923469387754e-06)
        unsqueeze_911: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1021, 0);  mul_1021 = None
        unsqueeze_912: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 2);  unsqueeze_911 = None
        unsqueeze_913: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 3);  unsqueeze_912 = None
        mul_1022: "f32[224]" = torch.ops.aten.mul.Tensor(sum_167, 9.964923469387754e-06)
        squeeze_19: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_1023: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_1024: "f32[224]" = torch.ops.aten.mul.Tensor(mul_1022, mul_1023);  mul_1022 = mul_1023 = None
        unsqueeze_914: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1024, 0);  mul_1024 = None
        unsqueeze_915: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 2);  unsqueeze_914 = None
        unsqueeze_916: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 3);  unsqueeze_915 = None
        mul_1025: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_46);  primals_46 = None
        unsqueeze_917: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1025, 0);  mul_1025 = None
        unsqueeze_918: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
        unsqueeze_919: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
        mul_1026: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_916);  sub_300 = unsqueeze_916 = None
        sub_302: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(where_70, mul_1026);  where_70 = mul_1026 = None
        sub_303: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(sub_302, unsqueeze_913);  sub_302 = unsqueeze_913 = None
        mul_1027: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_919);  sub_303 = unsqueeze_919 = None
        mul_1028: "f32[224]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_19);  sum_167 = squeeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1027, relu_5, primals_42, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1027 = primals_42 = None
        getitem_397: "f32[32, 224, 56, 56]" = convolution_backward_91[0]
        getitem_398: "f32[224, 112, 3, 3]" = convolution_backward_91[1];  convolution_backward_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_71: "b8[32, 224, 56, 56]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_71: "f32[32, 224, 56, 56]" = torch.ops.aten.where.self(le_71, full_default, getitem_397);  le_71 = getitem_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_168: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
        sub_304: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_922);  convolution_7 = unsqueeze_922 = None
        mul_1029: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_71, sub_304)
        sum_169: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_1029, [0, 2, 3]);  mul_1029 = None
        mul_1030: "f32[224]" = torch.ops.aten.mul.Tensor(sum_168, 9.964923469387754e-06)
        unsqueeze_923: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1030, 0);  mul_1030 = None
        unsqueeze_924: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 2);  unsqueeze_923 = None
        unsqueeze_925: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 3);  unsqueeze_924 = None
        mul_1031: "f32[224]" = torch.ops.aten.mul.Tensor(sum_169, 9.964923469387754e-06)
        mul_1032: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_1033: "f32[224]" = torch.ops.aten.mul.Tensor(mul_1031, mul_1032);  mul_1031 = mul_1032 = None
        unsqueeze_926: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1033, 0);  mul_1033 = None
        unsqueeze_927: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 2);  unsqueeze_926 = None
        unsqueeze_928: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 3);  unsqueeze_927 = None
        mul_1034: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_40);  primals_40 = None
        unsqueeze_929: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1034, 0);  mul_1034 = None
        unsqueeze_930: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 2);  unsqueeze_929 = None
        unsqueeze_931: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 3);  unsqueeze_930 = None
        mul_1035: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_928);  sub_304 = unsqueeze_928 = None
        sub_306: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(where_71, mul_1035);  where_71 = mul_1035 = None
        sub_307: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(sub_306, unsqueeze_925);  sub_306 = unsqueeze_925 = None
        mul_1036: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_931);  sub_307 = unsqueeze_931 = None
        mul_1037: "f32[224]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_16);  sum_169 = squeeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_1036, relu_4, primals_36, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1036 = primals_36 = None
        getitem_400: "f32[32, 224, 56, 56]" = convolution_backward_92[0]
        getitem_401: "f32[224, 224, 1, 1]" = convolution_backward_92[1];  convolution_backward_92 = None
        add_364: "f32[32, 224, 56, 56]" = torch.ops.aten.add.Tensor(where_68, getitem_400);  where_68 = getitem_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        le_72: "b8[32, 224, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_72: "f32[32, 224, 56, 56]" = torch.ops.aten.where.self(le_72, full_default, add_364);  le_72 = add_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_170: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
        sub_308: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_934);  convolution_6 = unsqueeze_934 = None
        mul_1038: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_72, sub_308)
        sum_171: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_1038, [0, 2, 3]);  mul_1038 = None
        mul_1039: "f32[224]" = torch.ops.aten.mul.Tensor(sum_170, 9.964923469387754e-06)
        unsqueeze_935: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1039, 0);  mul_1039 = None
        unsqueeze_936: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 2);  unsqueeze_935 = None
        unsqueeze_937: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, 3);  unsqueeze_936 = None
        mul_1040: "f32[224]" = torch.ops.aten.mul.Tensor(sum_171, 9.964923469387754e-06)
        mul_1041: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_1042: "f32[224]" = torch.ops.aten.mul.Tensor(mul_1040, mul_1041);  mul_1040 = mul_1041 = None
        unsqueeze_938: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1042, 0);  mul_1042 = None
        unsqueeze_939: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 2);  unsqueeze_938 = None
        unsqueeze_940: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 3);  unsqueeze_939 = None
        mul_1043: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_34);  primals_34 = None
        unsqueeze_941: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1043, 0);  mul_1043 = None
        unsqueeze_942: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 2);  unsqueeze_941 = None
        unsqueeze_943: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 3);  unsqueeze_942 = None
        mul_1044: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_940);  sub_308 = unsqueeze_940 = None
        sub_310: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(where_72, mul_1044);  mul_1044 = None
        sub_311: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(sub_310, unsqueeze_937);  sub_310 = unsqueeze_937 = None
        mul_1045: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_943);  sub_311 = unsqueeze_943 = None
        mul_1046: "f32[224]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_13);  sum_171 = squeeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_1045, relu, primals_30, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1045 = primals_30 = None
        getitem_403: "f32[32, 32, 112, 112]" = convolution_backward_93[0]
        getitem_404: "f32[224, 32, 1, 1]" = convolution_backward_93[1];  convolution_backward_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_172: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
        sub_312: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_946);  convolution_5 = unsqueeze_946 = None
        mul_1047: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_72, sub_312)
        sum_173: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_1047, [0, 2, 3]);  mul_1047 = None
        mul_1048: "f32[224]" = torch.ops.aten.mul.Tensor(sum_172, 9.964923469387754e-06)
        unsqueeze_947: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1048, 0);  mul_1048 = None
        unsqueeze_948: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 2);  unsqueeze_947 = None
        unsqueeze_949: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 3);  unsqueeze_948 = None
        mul_1049: "f32[224]" = torch.ops.aten.mul.Tensor(sum_173, 9.964923469387754e-06)
        mul_1050: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_1051: "f32[224]" = torch.ops.aten.mul.Tensor(mul_1049, mul_1050);  mul_1049 = mul_1050 = None
        unsqueeze_950: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1051, 0);  mul_1051 = None
        unsqueeze_951: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 2);  unsqueeze_950 = None
        unsqueeze_952: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 3);  unsqueeze_951 = None
        mul_1052: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_28);  primals_28 = None
        unsqueeze_953: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1052, 0);  mul_1052 = None
        unsqueeze_954: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 2);  unsqueeze_953 = None
        unsqueeze_955: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 3);  unsqueeze_954 = None
        mul_1053: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_952);  sub_312 = unsqueeze_952 = None
        sub_314: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(where_72, mul_1053);  where_72 = mul_1053 = None
        sub_315: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(sub_314, unsqueeze_949);  sub_314 = unsqueeze_949 = None
        mul_1054: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_955);  sub_315 = unsqueeze_955 = None
        mul_1055: "f32[224]" = torch.ops.aten.mul.Tensor(sum_173, squeeze_10);  sum_173 = squeeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_1054, mul_21, primals_24, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1054 = mul_21 = primals_24 = None
        getitem_406: "f32[32, 224, 56, 56]" = convolution_backward_94[0]
        getitem_407: "f32[224, 224, 1, 1]" = convolution_backward_94[1];  convolution_backward_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_2: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_14: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        unsqueeze_8: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_11: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[32, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_2: "f32[32, 224, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_1056: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_406, relu_2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid: "f32[32, 224, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_4);  convolution_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_1057: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_406, sigmoid);  getitem_406 = None
        sum_174: "f32[32, 224, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1056, [2, 3], True);  mul_1056 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_316: "f32[32, 224, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid)
        mul_1058: "f32[32, 224, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid, sub_316);  sigmoid = sub_316 = None
        mul_1059: "f32[32, 224, 1, 1]" = torch.ops.aten.mul.Tensor(sum_174, mul_1058);  sum_174 = mul_1058 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_175: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_1059, [0, 2, 3])
        convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_1059, relu_3, primals_22, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1059 = primals_22 = None
        getitem_409: "f32[32, 8, 1, 1]" = convolution_backward_95[0]
        getitem_410: "f32[224, 8, 1, 1]" = convolution_backward_95[1];  convolution_backward_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_73: "b8[32, 8, 1, 1]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_73: "f32[32, 8, 1, 1]" = torch.ops.aten.where.self(le_73, full_default, getitem_409);  le_73 = getitem_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_176: "f32[8]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
        convolution_backward_96 = torch.ops.aten.convolution_backward.default(where_73, mean, primals_20, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_73 = mean = primals_20 = None
        getitem_412: "f32[32, 224, 1, 1]" = convolution_backward_96[0]
        getitem_413: "f32[8, 224, 1, 1]" = convolution_backward_96[1];  convolution_backward_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_19: "f32[32, 224, 56, 56]" = torch.ops.aten.expand.default(getitem_412, [32, 224, 56, 56]);  getitem_412 = None
        div_19: "f32[32, 224, 56, 56]" = torch.ops.aten.div.Scalar(expand_19, 3136);  expand_19 = None
        add_365: "f32[32, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_1057, div_19);  mul_1057 = div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_74: "b8[32, 224, 56, 56]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_74: "f32[32, 224, 56, 56]" = torch.ops.aten.where.self(le_74, full_default, add_365);  le_74 = add_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_6: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        unsqueeze_956: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_957: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 2);  unsqueeze_956 = None
        unsqueeze_958: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 3);  unsqueeze_957 = None
        sum_177: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
        sub_317: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_958);  convolution_2 = unsqueeze_958 = None
        mul_1060: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(where_74, sub_317)
        sum_178: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_1060, [0, 2, 3]);  mul_1060 = None
        mul_1061: "f32[224]" = torch.ops.aten.mul.Tensor(sum_177, 9.964923469387754e-06)
        unsqueeze_959: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1061, 0);  mul_1061 = None
        unsqueeze_960: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 2);  unsqueeze_959 = None
        unsqueeze_961: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 3);  unsqueeze_960 = None
        mul_1062: "f32[224]" = torch.ops.aten.mul.Tensor(sum_178, 9.964923469387754e-06)
        squeeze_7: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_1063: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_1064: "f32[224]" = torch.ops.aten.mul.Tensor(mul_1062, mul_1063);  mul_1062 = mul_1063 = None
        unsqueeze_962: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1064, 0);  mul_1064 = None
        unsqueeze_963: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 2);  unsqueeze_962 = None
        unsqueeze_964: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 3);  unsqueeze_963 = None
        mul_1065: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_965: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1065, 0);  mul_1065 = None
        unsqueeze_966: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 2);  unsqueeze_965 = None
        unsqueeze_967: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 3);  unsqueeze_966 = None
        mul_1066: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_964);  sub_317 = unsqueeze_964 = None
        sub_319: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(where_74, mul_1066);  where_74 = mul_1066 = None
        sub_320: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(sub_319, unsqueeze_961);  sub_319 = unsqueeze_961 = None
        mul_1067: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_967);  sub_320 = unsqueeze_967 = None
        mul_1068: "f32[224]" = torch.ops.aten.mul.Tensor(sum_178, squeeze_7);  sum_178 = squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_97 = torch.ops.aten.convolution_backward.default(mul_1067, relu_1, primals_14, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_1067 = primals_14 = None
        getitem_415: "f32[32, 224, 112, 112]" = convolution_backward_97[0]
        getitem_416: "f32[224, 112, 3, 3]" = convolution_backward_97[1];  convolution_backward_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_75: "b8[32, 224, 112, 112]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_75: "f32[32, 224, 112, 112]" = torch.ops.aten.where.self(le_75, full_default, getitem_415);  le_75 = getitem_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_179: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
        sub_321: "f32[32, 224, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_970);  convolution_1 = unsqueeze_970 = None
        mul_1069: "f32[32, 224, 112, 112]" = torch.ops.aten.mul.Tensor(where_75, sub_321)
        sum_180: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_1069, [0, 2, 3]);  mul_1069 = None
        mul_1070: "f32[224]" = torch.ops.aten.mul.Tensor(sum_179, 2.4912308673469386e-06)
        unsqueeze_971: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1070, 0);  mul_1070 = None
        unsqueeze_972: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 2);  unsqueeze_971 = None
        unsqueeze_973: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 3);  unsqueeze_972 = None
        mul_1071: "f32[224]" = torch.ops.aten.mul.Tensor(sum_180, 2.4912308673469386e-06)
        mul_1072: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_1073: "f32[224]" = torch.ops.aten.mul.Tensor(mul_1071, mul_1072);  mul_1071 = mul_1072 = None
        unsqueeze_974: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1073, 0);  mul_1073 = None
        unsqueeze_975: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 2);  unsqueeze_974 = None
        unsqueeze_976: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 3);  unsqueeze_975 = None
        mul_1074: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_977: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_1074, 0);  mul_1074 = None
        unsqueeze_978: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
        unsqueeze_979: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
        mul_1075: "f32[32, 224, 112, 112]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_976);  sub_321 = unsqueeze_976 = None
        sub_323: "f32[32, 224, 112, 112]" = torch.ops.aten.sub.Tensor(where_75, mul_1075);  where_75 = mul_1075 = None
        sub_324: "f32[32, 224, 112, 112]" = torch.ops.aten.sub.Tensor(sub_323, unsqueeze_973);  sub_323 = unsqueeze_973 = None
        mul_1076: "f32[32, 224, 112, 112]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_979);  sub_324 = unsqueeze_979 = None
        mul_1077: "f32[224]" = torch.ops.aten.mul.Tensor(sum_180, squeeze_4);  sum_180 = squeeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_98 = torch.ops.aten.convolution_backward.default(mul_1076, relu, primals_8, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1076 = primals_8 = None
        getitem_418: "f32[32, 32, 112, 112]" = convolution_backward_98[0]
        getitem_419: "f32[224, 32, 1, 1]" = convolution_backward_98[1];  convolution_backward_98 = None
        add_366: "f32[32, 32, 112, 112]" = torch.ops.aten.add.Tensor(getitem_403, getitem_418);  getitem_403 = getitem_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_76: "b8[32, 32, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_76: "f32[32, 32, 112, 112]" = torch.ops.aten.where.self(le_76, full_default, add_366);  le_76 = full_default = add_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_181: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
        sub_325: "f32[32, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_982);  convolution = unsqueeze_982 = None
        mul_1078: "f32[32, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_76, sub_325)
        sum_182: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1078, [0, 2, 3]);  mul_1078 = None
        mul_1079: "f32[32]" = torch.ops.aten.mul.Tensor(sum_181, 2.4912308673469386e-06)
        unsqueeze_983: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1079, 0);  mul_1079 = None
        unsqueeze_984: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 2);  unsqueeze_983 = None
        unsqueeze_985: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 3);  unsqueeze_984 = None
        mul_1080: "f32[32]" = torch.ops.aten.mul.Tensor(sum_182, 2.4912308673469386e-06)
        mul_1081: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_1082: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1080, mul_1081);  mul_1080 = mul_1081 = None
        unsqueeze_986: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
        unsqueeze_987: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 2);  unsqueeze_986 = None
        unsqueeze_988: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 3);  unsqueeze_987 = None
        mul_1083: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_989: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1083, 0);  mul_1083 = None
        unsqueeze_990: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 2);  unsqueeze_989 = None
        unsqueeze_991: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 3);  unsqueeze_990 = None
        mul_1084: "f32[32, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_988);  sub_325 = unsqueeze_988 = None
        sub_327: "f32[32, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_76, mul_1084);  where_76 = mul_1084 = None
        sub_328: "f32[32, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_327, unsqueeze_985);  sub_327 = unsqueeze_985 = None
        mul_1085: "f32[32, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_991);  sub_328 = unsqueeze_991 = None
        mul_1086: "f32[32]" = torch.ops.aten.mul.Tensor(sum_182, squeeze_1);  sum_182 = squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_99 = torch.ops.aten.convolution_backward.default(mul_1085, primals_2, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1085 = primals_2 = primals_1 = None
        getitem_422: "f32[32, 3, 3, 3]" = convolution_backward_99[1];  convolution_backward_99 = None
        return (getitem_422, None, None, None, None, mul_1086, sum_181, getitem_419, None, None, None, mul_1077, sum_179, getitem_416, None, None, None, mul_1068, sum_177, getitem_413, sum_176, getitem_410, sum_175, getitem_407, None, None, None, mul_1055, sum_172, getitem_404, None, None, None, mul_1046, sum_170, getitem_401, None, None, None, mul_1037, sum_168, getitem_398, None, None, None, mul_1028, sum_166, getitem_395, sum_165, getitem_392, sum_164, getitem_389, None, None, None, mul_1015, sum_161, getitem_386, None, None, None, mul_1006, sum_159, getitem_383, None, None, None, mul_997, sum_157, getitem_380, sum_156, getitem_377, sum_155, getitem_374, None, None, None, mul_984, sum_152, getitem_371, None, None, None, mul_975, sum_150, getitem_368, None, None, None, mul_966, sum_148, getitem_365, None, None, None, mul_957, sum_146, getitem_362, sum_145, getitem_359, sum_144, getitem_356, None, None, None, mul_944, sum_141, getitem_353, None, None, None, mul_935, sum_139, getitem_350, None, None, None, mul_926, sum_137, getitem_347, sum_136, getitem_344, sum_135, getitem_341, None, None, None, mul_913, sum_132, getitem_338, None, None, None, mul_904, sum_130, getitem_335, None, None, None, mul_895, sum_128, getitem_332, sum_127, getitem_329, sum_126, getitem_326, None, None, None, mul_882, sum_123, getitem_323, None, None, None, mul_873, sum_121, getitem_320, None, None, None, mul_864, sum_119, getitem_317, sum_118, getitem_314, sum_117, getitem_311, None, None, None, mul_851, sum_114, getitem_308, None, None, None, mul_842, sum_112, getitem_305, None, None, None, mul_833, sum_110, getitem_302, sum_109, getitem_299, sum_108, getitem_296, None, None, None, mul_820, sum_105, getitem_293, None, None, None, mul_811, sum_103, getitem_290, None, None, None, mul_802, sum_101, getitem_287, None, None, None, mul_793, sum_99, getitem_284, sum_98, getitem_281, sum_97, getitem_278, None, None, None, mul_780, sum_94, getitem_275, None, None, None, mul_771, sum_92, getitem_272, None, None, None, mul_762, sum_90, getitem_269, sum_89, getitem_266, sum_88, getitem_263, None, None, None, mul_749, sum_85, getitem_260, None, None, None, mul_740, sum_83, getitem_257, None, None, None, mul_731, sum_81, getitem_254, sum_80, getitem_251, sum_79, getitem_248, None, None, None, mul_718, sum_76, getitem_245, None, None, None, mul_709, sum_74, getitem_242, None, None, None, mul_700, sum_72, getitem_239, sum_71, getitem_236, sum_70, getitem_233, None, None, None, mul_687, sum_67, getitem_230, None, None, None, mul_678, sum_65, getitem_227, None, None, None, mul_669, sum_63, getitem_224, sum_62, getitem_221, sum_61, getitem_218, None, None, None, mul_656, sum_58, getitem_215, None, None, None, mul_647, sum_56, getitem_212, None, None, None, mul_638, sum_54, getitem_209, sum_53, getitem_206, sum_52, getitem_203, None, None, None, mul_625, sum_49, getitem_200, None, None, None, mul_616, sum_47, getitem_197, None, None, None, mul_607, sum_45, getitem_194, sum_44, getitem_191, sum_43, getitem_188, None, None, None, mul_594, sum_40, getitem_185, None, None, None, mul_585, sum_38, getitem_182, None, None, None, mul_576, sum_36, getitem_179, sum_35, getitem_176, sum_34, getitem_173, None, None, None, mul_563, sum_31, getitem_170, None, None, None, mul_554, sum_29, getitem_167, None, None, None, mul_545, sum_27, getitem_164, sum_26, getitem_161, sum_25, getitem_158, None, None, None, mul_532, sum_22, getitem_155, None, None, None, mul_523, sum_20, getitem_152, None, None, None, mul_514, sum_18, getitem_149, sum_17, getitem_146, sum_16, getitem_143, None, None, None, mul_501, sum_13, getitem_140, None, None, None, mul_492, sum_11, getitem_137, None, None, None, mul_483, sum_9, getitem_134, sum_8, getitem_131, sum_7, getitem_128, None, None, None, mul_470, sum_4, getitem_125, None, None, None, mul_461, sum_2, mm_1, view_1)
