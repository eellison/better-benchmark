class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "f32[32][1]cuda:0", primals_12: "f32[32][1]cuda:0", primals_18: "f32[64][1]cuda:0", primals_19: "f32[64][1]cuda:0", primals_24: "f32[80][1]cuda:0", primals_30: "f32[192][1]cuda:0", primals_31: "f32[192][1]cuda:0", primals_36: "f32[64][1]cuda:0", primals_37: "f32[64][1]cuda:0", primals_42: "f32[48][1]cuda:0", primals_48: "f32[64][1]cuda:0", primals_49: "f32[64][1]cuda:0", primals_54: "f32[64][1]cuda:0", primals_60: "f32[96][1]cuda:0", primals_66: "f32[96][1]cuda:0", primals_67: "f32[96][1]cuda:0", primals_72: "f32[32][1]cuda:0", primals_73: "f32[32][1]cuda:0", primals_78: "f32[64][1]cuda:0", primals_79: "f32[64][1]cuda:0", primals_84: "f32[48][1]cuda:0", primals_90: "f32[64][1]cuda:0", primals_91: "f32[64][1]cuda:0", primals_96: "f32[64][1]cuda:0", primals_102: "f32[96][1]cuda:0", primals_108: "f32[96][1]cuda:0", primals_109: "f32[96][1]cuda:0", primals_114: "f32[64][1]cuda:0", primals_115: "f32[64][1]cuda:0", primals_120: "f32[64][1]cuda:0", primals_121: "f32[64][1]cuda:0", primals_126: "f32[48][1]cuda:0", primals_132: "f32[64][1]cuda:0", primals_133: "f32[64][1]cuda:0", primals_138: "f32[64][1]cuda:0", primals_144: "f32[96][1]cuda:0", primals_150: "f32[96][1]cuda:0", primals_151: "f32[96][1]cuda:0", primals_156: "f32[64][1]cuda:0", primals_157: "f32[64][1]cuda:0", primals_162: "f32[384][1]cuda:0", primals_163: "f32[384][1]cuda:0", primals_168: "f32[64][1]cuda:0", primals_174: "f32[96][1]cuda:0", primals_180: "f32[96][1]cuda:0", primals_181: "f32[96][1]cuda:0", primals_186: "f32[192][1]cuda:0", primals_187: "f32[192][1]cuda:0", primals_192: "f32[128][1]cuda:0", primals_198: "f32[128][1]cuda:0", primals_204: "f32[192][1]cuda:0", primals_205: "f32[192][1]cuda:0", primals_210: "f32[128][1]cuda:0", primals_216: "f32[128][1]cuda:0", primals_222: "f32[128][1]cuda:0", primals_228: "f32[128][1]cuda:0", primals_234: "f32[192][1]cuda:0", primals_235: "f32[192][1]cuda:0", primals_240: "f32[192][1]cuda:0", primals_241: "f32[192][1]cuda:0", primals_246: "f32[192][1]cuda:0", primals_247: "f32[192][1]cuda:0", primals_252: "f32[160][1]cuda:0", primals_258: "f32[160][1]cuda:0", primals_264: "f32[192][1]cuda:0", primals_265: "f32[192][1]cuda:0", primals_270: "f32[160][1]cuda:0", primals_276: "f32[160][1]cuda:0", primals_282: "f32[160][1]cuda:0", primals_288: "f32[160][1]cuda:0", primals_294: "f32[192][1]cuda:0", primals_295: "f32[192][1]cuda:0", primals_300: "f32[192][1]cuda:0", primals_301: "f32[192][1]cuda:0", primals_306: "f32[192][1]cuda:0", primals_307: "f32[192][1]cuda:0", primals_312: "f32[160][1]cuda:0", primals_318: "f32[160][1]cuda:0", primals_324: "f32[192][1]cuda:0", primals_325: "f32[192][1]cuda:0", primals_330: "f32[160][1]cuda:0", primals_336: "f32[160][1]cuda:0", primals_342: "f32[160][1]cuda:0", primals_348: "f32[160][1]cuda:0", primals_354: "f32[192][1]cuda:0", primals_355: "f32[192][1]cuda:0", primals_360: "f32[192][1]cuda:0", primals_361: "f32[192][1]cuda:0", primals_366: "f32[192][1]cuda:0", primals_367: "f32[192][1]cuda:0", primals_372: "f32[192][1]cuda:0", primals_378: "f32[192][1]cuda:0", primals_384: "f32[192][1]cuda:0", primals_385: "f32[192][1]cuda:0", primals_390: "f32[192][1]cuda:0", primals_396: "f32[192][1]cuda:0", primals_402: "f32[192][1]cuda:0", primals_408: "f32[192][1]cuda:0", primals_414: "f32[192][1]cuda:0", primals_415: "f32[192][1]cuda:0", primals_420: "f32[192][1]cuda:0", primals_421: "f32[192][1]cuda:0", primals_426: "f32[192][1]cuda:0", primals_432: "f32[320][1]cuda:0", primals_433: "f32[320][1]cuda:0", primals_438: "f32[192][1]cuda:0", primals_444: "f32[192][1]cuda:0", primals_450: "f32[192][1]cuda:0", primals_456: "f32[192][1]cuda:0", primals_457: "f32[192][1]cuda:0", primals_462: "f32[320][1]cuda:0", primals_463: "f32[320][1]cuda:0", primals_468: "f32[384][1]cuda:0", primals_474: "f32[384][1]cuda:0", primals_475: "f32[384][1]cuda:0", primals_480: "f32[384][1]cuda:0", primals_481: "f32[384][1]cuda:0", primals_486: "f32[448][1]cuda:0", primals_492: "f32[384][1]cuda:0", primals_498: "f32[384][1]cuda:0", primals_499: "f32[384][1]cuda:0", primals_504: "f32[384][1]cuda:0", primals_505: "f32[384][1]cuda:0", primals_510: "f32[192][1]cuda:0", primals_511: "f32[192][1]cuda:0", primals_516: "f32[320][1]cuda:0", primals_517: "f32[320][1]cuda:0", primals_522: "f32[384][1]cuda:0", primals_528: "f32[384][1]cuda:0", primals_529: "f32[384][1]cuda:0", primals_534: "f32[384][1]cuda:0", primals_535: "f32[384][1]cuda:0", primals_540: "f32[448][1]cuda:0", primals_546: "f32[384][1]cuda:0", primals_552: "f32[384][1]cuda:0", primals_553: "f32[384][1]cuda:0", primals_558: "f32[384][1]cuda:0", primals_559: "f32[384][1]cuda:0", primals_564: "f32[192][1]cuda:0", primals_565: "f32[192][1]cuda:0", convert_element_type: "bf16[32, 3, 3, 3][27, 1, 9, 3]cuda:0", convert_element_type_1: "bf16[128, 3, 299, 299][268203, 1, 897, 3]cuda:0", convolution: "bf16[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0", squeeze_1: "f32[32][1]cuda:0", relu: "bf16[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0", convert_element_type_4: "bf16[32, 32, 3, 3][288, 1, 96, 32]cuda:0", convolution_1: "bf16[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0", squeeze_4: "f32[32][1]cuda:0", relu_1: "bf16[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0", convert_element_type_7: "bf16[64, 32, 3, 3][288, 1, 96, 32]cuda:0", convolution_2: "bf16[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0", getitem_5: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", rsqrt_2: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", getitem_6: "bf16[128, 64, 73, 73][341056, 1, 4672, 64]cuda:0", getitem_7: "i8[128, 64, 73, 73][341056, 1, 4672, 64]cuda:0", convert_element_type_10: "bf16[80, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_3: "bf16[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0", squeeze_10: "f32[80][1]cuda:0", relu_3: "bf16[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0", convert_element_type_13: "bf16[192, 80, 3, 3][720, 1, 240, 80]cuda:0", convolution_4: "bf16[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0", getitem_11: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_4: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", getitem_12: "bf16[128, 192, 35, 35][235200, 1, 6720, 192]cuda:0", getitem_13: "i8[128, 192, 35, 35][235200, 1, 6720, 192]cuda:0", convert_element_type_16: "bf16[64, 192, 1, 1][192, 1, 192, 192]cuda:0", convolution_5: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0", getitem_15: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", rsqrt_5: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", convert_element_type_19: "bf16[48, 192, 1, 1][192, 1, 192, 192]cuda:0", convolution_6: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0", squeeze_19: "f32[48][1]cuda:0", relu_6: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0", convert_element_type_22: "bf16[64, 48, 5, 5][1200, 1, 240, 48]cuda:0", convolution_7: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0", getitem_19: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", rsqrt_7: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", convert_element_type_25: "bf16[64, 192, 1, 1][192, 1, 192, 192]cuda:0", convolution_8: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0", squeeze_25: "f32[64][1]cuda:0", relu_8: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0", convert_element_type_28: "bf16[96, 64, 3, 3][576, 1, 192, 64]cuda:0", convolution_9: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0", squeeze_28: "f32[96][1]cuda:0", relu_9: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0", convert_element_type_31: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0", convolution_10: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0", getitem_25: "f32[1, 96, 1, 1][96, 1, 96, 96]cuda:0", rsqrt_10: "f32[1, 96, 1, 1][96, 1, 96, 96]cuda:0", avg_pool2d: "bf16[128, 192, 35, 35][235200, 1, 6720, 192]cuda:0", convert_element_type_34: "bf16[32, 192, 1, 1][192, 1, 192, 192]cuda:0", convolution_11: "bf16[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0", getitem_27: "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0", rsqrt_11: "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0", cat: "bf16[128, 256, 35, 35][313600, 1, 8960, 256]cuda:0", convert_element_type_37: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_12: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0", getitem_29: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", rsqrt_12: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", convert_element_type_40: "bf16[48, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_13: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0", squeeze_40: "f32[48][1]cuda:0", relu_13: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0", convert_element_type_43: "bf16[64, 48, 5, 5][1200, 1, 240, 48]cuda:0", convolution_14: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0", getitem_33: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", rsqrt_14: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", convert_element_type_46: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_15: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0", squeeze_46: "f32[64][1]cuda:0", relu_15: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0", convert_element_type_49: "bf16[96, 64, 3, 3][576, 1, 192, 64]cuda:0", convolution_16: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0", squeeze_49: "f32[96][1]cuda:0", relu_16: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0", convert_element_type_52: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0", convolution_17: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0", getitem_39: "f32[1, 96, 1, 1][96, 1, 96, 96]cuda:0", rsqrt_17: "f32[1, 96, 1, 1][96, 1, 96, 96]cuda:0", avg_pool2d_1: "bf16[128, 256, 35, 35][313600, 1, 8960, 256]cuda:0", convert_element_type_55: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_18: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0", getitem_41: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", rsqrt_18: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", cat_1: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0", convert_element_type_58: "bf16[64, 288, 1, 1][288, 1, 288, 288]cuda:0", convolution_19: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0", getitem_43: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", rsqrt_19: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", convert_element_type_61: "bf16[48, 288, 1, 1][288, 1, 288, 288]cuda:0", convolution_20: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0", squeeze_61: "f32[48][1]cuda:0", relu_20: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0", convert_element_type_64: "bf16[64, 48, 5, 5][1200, 1, 240, 48]cuda:0", convolution_21: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0", getitem_47: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", rsqrt_21: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", convert_element_type_67: "bf16[64, 288, 1, 1][288, 1, 288, 288]cuda:0", convolution_22: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0", squeeze_67: "f32[64][1]cuda:0", relu_22: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0", convert_element_type_70: "bf16[96, 64, 3, 3][576, 1, 192, 64]cuda:0", convolution_23: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0", squeeze_70: "f32[96][1]cuda:0", relu_23: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0", convert_element_type_73: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0", convolution_24: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0", getitem_53: "f32[1, 96, 1, 1][96, 1, 96, 96]cuda:0", rsqrt_24: "f32[1, 96, 1, 1][96, 1, 96, 96]cuda:0", avg_pool2d_2: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0", convert_element_type_76: "bf16[64, 288, 1, 1][288, 1, 288, 288]cuda:0", convolution_25: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0", getitem_55: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", rsqrt_25: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", cat_2: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0", convert_element_type_79: "bf16[384, 288, 3, 3][2592, 1, 864, 288]cuda:0", convolution_26: "bf16[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0", getitem_57: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_26: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_82: "bf16[64, 288, 1, 1][288, 1, 288, 288]cuda:0", convolution_27: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0", squeeze_82: "f32[64][1]cuda:0", relu_27: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0", convert_element_type_85: "bf16[96, 64, 3, 3][576, 1, 192, 64]cuda:0", convolution_28: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0", squeeze_85: "f32[96][1]cuda:0", relu_28: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0", convert_element_type_88: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0", convolution_29: "bf16[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0", getitem_63: "f32[1, 96, 1, 1][96, 1, 96, 96]cuda:0", rsqrt_29: "f32[1, 96, 1, 1][96, 1, 96, 96]cuda:0", getitem_65: "i8[128, 288, 17, 17][83232, 1, 4896, 288]cuda:0", cat_3: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0", convert_element_type_91: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_30: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", getitem_67: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_30: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_94: "bf16[128, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_31: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0", squeeze_94: "f32[128][1]cuda:0", relu_31: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0", convert_element_type_97: "bf16[128, 128, 1, 7][896, 1, 896, 128]cuda:0", convolution_32: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0", squeeze_97: "f32[128][1]cuda:0", relu_32: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0", convert_element_type_100: "bf16[192, 128, 7, 1][896, 1, 128, 128]cuda:0", convolution_33: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", getitem_73: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_33: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_103: "bf16[128, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_34: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0", squeeze_103: "f32[128][1]cuda:0", relu_34: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0", convert_element_type_106: "bf16[128, 128, 7, 1][896, 1, 128, 128]cuda:0", convolution_35: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0", squeeze_106: "f32[128][1]cuda:0", relu_35: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0", convert_element_type_109: "bf16[128, 128, 1, 7][896, 1, 896, 128]cuda:0", convolution_36: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0", squeeze_109: "f32[128][1]cuda:0", relu_36: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0", convert_element_type_112: "bf16[128, 128, 7, 1][896, 1, 128, 128]cuda:0", convolution_37: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0", squeeze_112: "f32[128][1]cuda:0", relu_37: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0", convert_element_type_115: "bf16[192, 128, 1, 7][896, 1, 896, 128]cuda:0", convolution_38: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", getitem_83: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_38: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", avg_pool2d_3: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0", convert_element_type_118: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_39: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", getitem_85: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_39: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", cat_4: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0", convert_element_type_121: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_40: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", getitem_87: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_40: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_124: "bf16[160, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_41: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", squeeze_124: "f32[160][1]cuda:0", relu_41: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", convert_element_type_127: "bf16[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0", convolution_42: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", squeeze_127: "f32[160][1]cuda:0", relu_42: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", convert_element_type_130: "bf16[192, 160, 7, 1][1120, 1, 160, 160]cuda:0", convolution_43: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", getitem_93: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_43: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_133: "bf16[160, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_44: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", squeeze_133: "f32[160][1]cuda:0", relu_44: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", convert_element_type_136: "bf16[160, 160, 7, 1][1120, 1, 160, 160]cuda:0", convolution_45: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", squeeze_136: "f32[160][1]cuda:0", relu_45: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", convert_element_type_139: "bf16[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0", convolution_46: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", squeeze_139: "f32[160][1]cuda:0", relu_46: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", convert_element_type_142: "bf16[160, 160, 7, 1][1120, 1, 160, 160]cuda:0", convolution_47: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", squeeze_142: "f32[160][1]cuda:0", relu_47: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", convert_element_type_145: "bf16[192, 160, 1, 7][1120, 1, 1120, 160]cuda:0", convolution_48: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", getitem_103: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_48: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", avg_pool2d_4: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0", convert_element_type_148: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_49: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", getitem_105: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_49: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", cat_5: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0", convert_element_type_151: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_50: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", getitem_107: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_50: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_154: "bf16[160, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_51: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", squeeze_154: "f32[160][1]cuda:0", relu_51: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", convert_element_type_157: "bf16[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0", convolution_52: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", squeeze_157: "f32[160][1]cuda:0", relu_52: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", convert_element_type_160: "bf16[192, 160, 7, 1][1120, 1, 160, 160]cuda:0", convolution_53: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", getitem_113: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_53: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_163: "bf16[160, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_54: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", squeeze_163: "f32[160][1]cuda:0", relu_54: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", convert_element_type_166: "bf16[160, 160, 7, 1][1120, 1, 160, 160]cuda:0", convolution_55: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", squeeze_166: "f32[160][1]cuda:0", relu_55: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", convert_element_type_169: "bf16[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0", convolution_56: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", squeeze_169: "f32[160][1]cuda:0", relu_56: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", convert_element_type_172: "bf16[160, 160, 7, 1][1120, 1, 160, 160]cuda:0", convolution_57: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", squeeze_172: "f32[160][1]cuda:0", relu_57: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0", convert_element_type_175: "bf16[192, 160, 1, 7][1120, 1, 1120, 160]cuda:0", convolution_58: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", getitem_123: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_58: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", avg_pool2d_5: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0", convert_element_type_178: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_59: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", getitem_125: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_59: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", cat_6: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0", convert_element_type_181: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_60: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", getitem_127: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_60: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_184: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_61: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", squeeze_184: "f32[192][1]cuda:0", relu_61: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", convert_element_type_187: "bf16[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0", convolution_62: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", squeeze_187: "f32[192][1]cuda:0", relu_62: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", convert_element_type_190: "bf16[192, 192, 7, 1][1344, 1, 192, 192]cuda:0", convolution_63: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", getitem_133: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_63: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_193: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_64: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", squeeze_193: "f32[192][1]cuda:0", relu_64: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", convert_element_type_196: "bf16[192, 192, 7, 1][1344, 1, 192, 192]cuda:0", convolution_65: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", squeeze_196: "f32[192][1]cuda:0", relu_65: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", convert_element_type_199: "bf16[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0", convolution_66: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", squeeze_199: "f32[192][1]cuda:0", relu_66: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", convert_element_type_202: "bf16[192, 192, 7, 1][1344, 1, 192, 192]cuda:0", convolution_67: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", squeeze_202: "f32[192][1]cuda:0", relu_67: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", convert_element_type_205: "bf16[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0", convolution_68: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", getitem_143: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_68: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", avg_pool2d_6: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0", convert_element_type_208: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_69: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", getitem_145: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_69: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", cat_7: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0", convert_element_type_211: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_70: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", squeeze_211: "f32[192][1]cuda:0", relu_70: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", convert_element_type_214: "bf16[320, 192, 3, 3][1728, 1, 576, 192]cuda:0", convolution_71: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0", getitem_149: "f32[1, 320, 1, 1][320, 1, 320, 320]cuda:0", rsqrt_71: "f32[1, 320, 1, 1][320, 1, 320, 320]cuda:0", convert_element_type_217: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0", convolution_72: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", squeeze_217: "f32[192][1]cuda:0", relu_72: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", convert_element_type_220: "bf16[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0", convolution_73: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", squeeze_220: "f32[192][1]cuda:0", relu_73: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", convert_element_type_223: "bf16[192, 192, 7, 1][1344, 1, 192, 192]cuda:0", convolution_74: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", squeeze_223: "f32[192][1]cuda:0", relu_74: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0", convert_element_type_226: "bf16[192, 192, 3, 3][1728, 1, 576, 192]cuda:0", convolution_75: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0", getitem_157: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_75: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", getitem_159: "i8[128, 768, 8, 8][49152, 1, 6144, 768]cuda:0", cat_8: "bf16[128, 1280, 8, 8][81920, 1, 10240, 1280]cuda:0", convert_element_type_229: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", convolution_76: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0", getitem_161: "f32[1, 320, 1, 1][320, 1, 320, 320]cuda:0", rsqrt_76: "f32[1, 320, 1, 1][320, 1, 320, 320]cuda:0", convert_element_type_232: "bf16[384, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", convolution_77: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0", squeeze_232: "f32[384][1]cuda:0", relu_77: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0", convert_element_type_235: "bf16[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0", convolution_78: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0", getitem_165: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_78: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_238: "bf16[384, 384, 3, 1][1152, 1, 384, 384]cuda:0", convolution_79: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0", getitem_167: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_79: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_241: "bf16[448, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", convolution_80: "bf16[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0", squeeze_241: "f32[448][1]cuda:0", relu_80: "bf16[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0", convert_element_type_244: "bf16[384, 448, 3, 3][4032, 1, 1344, 448]cuda:0", convolution_81: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0", squeeze_244: "f32[384][1]cuda:0", relu_81: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0", convert_element_type_247: "bf16[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0", convolution_82: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0", getitem_173: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_82: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_250: "bf16[384, 384, 3, 1][1152, 1, 384, 384]cuda:0", convolution_83: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0", getitem_175: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_83: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", avg_pool2d_7: "bf16[128, 1280, 8, 8][81920, 1, 10240, 1280]cuda:0", convert_element_type_253: "bf16[192, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", convolution_84: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0", getitem_177: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_84: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", cat_11: "bf16[128, 2048, 8, 8][131072, 1, 16384, 2048]cuda:0", convert_element_type_256: "bf16[320, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0", convolution_85: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0", getitem_179: "f32[1, 320, 1, 1][320, 1, 320, 320]cuda:0", rsqrt_85: "f32[1, 320, 1, 1][320, 1, 320, 320]cuda:0", convert_element_type_259: "bf16[384, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0", convolution_86: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0", squeeze_259: "f32[384][1]cuda:0", relu_86: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0", convert_element_type_262: "bf16[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0", convolution_87: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0", getitem_183: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_87: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_265: "bf16[384, 384, 3, 1][1152, 1, 384, 384]cuda:0", convolution_88: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0", getitem_185: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_88: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_268: "bf16[448, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0", convolution_89: "bf16[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0", squeeze_268: "f32[448][1]cuda:0", relu_89: "bf16[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0", convert_element_type_271: "bf16[384, 448, 3, 3][4032, 1, 1344, 448]cuda:0", convolution_90: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0", squeeze_271: "f32[384][1]cuda:0", relu_90: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0", convert_element_type_274: "bf16[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0", convolution_91: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0", getitem_191: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_91: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_277: "bf16[384, 384, 3, 1][1152, 1, 384, 384]cuda:0", convolution_92: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0", getitem_193: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_92: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", avg_pool2d_8: "bf16[128, 2048, 8, 8][131072, 1, 16384, 2048]cuda:0", convert_element_type_280: "bf16[192, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0", convolution_93: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0", getitem_195: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", rsqrt_93: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", view: "bf16[128, 2048][2048, 1]cuda:0", permute_1: "bf16[1000, 2048][2048, 1]cuda:0", unsqueeze_414: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_426: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0", unsqueeze_462: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_522: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_534: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0", unsqueeze_570: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_606: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_618: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_630: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_654: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_690: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_702: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_714: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_726: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_750: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_762: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_810: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_822: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_834: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_846: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_870: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_882: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_930: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_942: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_954: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_966: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_990: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_1002: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_1050: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1062: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1074: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1086: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1110: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1122: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1158: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_1170: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_1218: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_1230: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_1254: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0", unsqueeze_1302: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_1314: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_1338: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0", unsqueeze_1386: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_1398: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_1422: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0", unsqueeze_1458: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_1482: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_1494: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", tangents_1: "bf16[128, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:430 in forward_head, code: x = self.fc(x)
        mm: "bf16[128, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "bf16[1000, 128][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_1: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type_292: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_293: "f32[1000, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_294: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_292, torch.float32);  convert_element_type_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_2: "bf16[128, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [128, 2048, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze_282: "bf16[128, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2, 3);  view_2 = None
        squeeze_283: "bf16[128, 2048][2048, 1]cuda:0" = torch.ops.aten.squeeze.dim(squeeze_282, 2);  squeeze_282 = None
        full: "bf16[262144][1]cuda:0" = torch.ops.aten.full.default([262144], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter: "bf16[262144][1]cuda:0" = torch.ops.aten.as_strided_scatter.default(full, squeeze_283, [128, 2048], [2048, 1], 0);  full = squeeze_283 = None
        as_strided_5: "bf16[128, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(as_strided_scatter, [128, 2048, 1, 1], [2048, 1, 1, 1], 0);  as_strided_scatter = None
        expand_1: "bf16[128, 2048, 8, 8][2048, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(as_strided_5, [128, 2048, 8, 8]);  as_strided_5 = None
        div: "bf16[128, 2048, 8, 8][131072, 64, 8, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_1, 64);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:244 in forward, code: return torch.cat(outputs, 1)
        slice_1: "bf16[128, 320, 8, 8][131072, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(div, 1, 0, 320)
        slice_2: "bf16[128, 768, 8, 8][131072, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(div, 1, 320, 1088)
        slice_3: "bf16[128, 768, 8, 8][131072, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(div, 1, 1088, 1856)
        slice_4: "bf16[128, 192, 8, 8][131072, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(div, 1, 1856, 2048);  div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_93: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_93, getitem_195)
        mul_651: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = None
        unsqueeze_372: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_564, -1)
        unsqueeze_373: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_657: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_651, unsqueeze_373);  mul_651 = unsqueeze_373 = None
        unsqueeze_374: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_565, -1);  primals_565 = None
        unsqueeze_375: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_469: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_657, unsqueeze_375);  mul_657 = unsqueeze_375 = None
        convert_element_type_282: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_469, torch.bfloat16);  add_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_93: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_282);  convert_element_type_282 = None
        le: "b8[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_93, 0);  relu_93 = None
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.where.self(le, full_default, slice_4);  le = slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_295: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze_279: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_195, [0, 2, 3]);  getitem_195 = None
        unsqueeze_376: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_279, 0);  squeeze_279 = None
        unsqueeze_377: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
        unsqueeze_378: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
        sum_2: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_295, [0, 2, 3])
        convert_element_type_281: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_93, torch.float32);  convolution_93 = None
        sub_94: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_281, unsqueeze_378);  convert_element_type_281 = unsqueeze_378 = None
        mul_658: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_295, sub_94)
        sum_3: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_658, [0, 2, 3]);  mul_658 = None
        mul_659: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_2, 0.0001220703125)
        unsqueeze_379: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_659, 0);  mul_659 = None
        unsqueeze_380: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
        unsqueeze_381: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
        mul_660: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 0.0001220703125)
        squeeze_280: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_93, [0, 2, 3]);  rsqrt_93 = None
        mul_661: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_280, squeeze_280)
        mul_662: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_660, mul_661);  mul_660 = mul_661 = None
        unsqueeze_382: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_662, 0);  mul_662 = None
        unsqueeze_383: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
        unsqueeze_384: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
        mul_663: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_280, primals_564);  primals_564 = None
        unsqueeze_385: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
        unsqueeze_386: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
        unsqueeze_387: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
        mul_664: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_384);  sub_94 = unsqueeze_384 = None
        sub_96: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_295, mul_664);  convert_element_type_295 = mul_664 = None
        sub_97: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_381);  sub_96 = unsqueeze_381 = None
        mul_665: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_387);  sub_97 = unsqueeze_387 = None
        mul_666: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, squeeze_280);  sum_3 = squeeze_280 = None
        convert_element_type_297: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_665, torch.bfloat16);  mul_665 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(convert_element_type_297, avg_pool2d_8, convert_element_type_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_297 = avg_pool2d_8 = convert_element_type_280 = None
        getitem_196: "bf16[128, 2048, 8, 8][131072, 1, 16384, 2048]cuda:0" = convolution_backward[0]
        getitem_197: "bf16[192, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_298: "f32[192, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_197, torch.float32);  getitem_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:236 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward: "bf16[128, 2048, 8, 8][131072, 1, 16384, 2048]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(getitem_196, cat_11, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:234 in _forward, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
        slice_5: "bf16[128, 384, 8, 8][131072, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_3, 1, 0, 384)
        slice_6: "bf16[128, 384, 8, 8][131072, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_3, 1, 384, 768);  slice_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_92: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_92, getitem_193)
        mul_644: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = None
        unsqueeze_368: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_558, -1)
        unsqueeze_369: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        mul_650: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_644, unsqueeze_369);  mul_644 = unsqueeze_369 = None
        unsqueeze_370: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_559, -1);  primals_559 = None
        unsqueeze_371: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        add_464: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_650, unsqueeze_371);  mul_650 = unsqueeze_371 = None
        convert_element_type_279: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_464, torch.bfloat16);  add_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_92: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_279);  convert_element_type_279 = None
        le_1: "b8[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_92, 0);  relu_92 = None
        where_1: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.where.self(le_1, full_default, slice_6);  le_1 = slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_299: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        squeeze_276: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_193, [0, 2, 3]);  getitem_193 = None
        unsqueeze_388: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_276, 0);  squeeze_276 = None
        unsqueeze_389: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
        unsqueeze_390: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
        sum_4: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_299, [0, 2, 3])
        convert_element_type_278: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_92, torch.float32);  convolution_92 = None
        sub_98: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_278, unsqueeze_390);  convert_element_type_278 = unsqueeze_390 = None
        mul_667: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_299, sub_98)
        sum_5: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_667, [0, 2, 3]);  mul_667 = None
        mul_668: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 0.0001220703125)
        unsqueeze_391: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_668, 0);  mul_668 = None
        unsqueeze_392: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
        unsqueeze_393: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
        mul_669: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, 0.0001220703125)
        squeeze_277: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_92, [0, 2, 3]);  rsqrt_92 = None
        mul_670: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_277, squeeze_277)
        mul_671: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
        unsqueeze_394: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
        unsqueeze_395: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
        unsqueeze_396: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
        mul_672: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_277, primals_558);  primals_558 = None
        unsqueeze_397: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
        unsqueeze_398: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
        unsqueeze_399: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
        mul_673: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_396);  sub_98 = unsqueeze_396 = None
        sub_100: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_299, mul_673);  convert_element_type_299 = mul_673 = None
        sub_101: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_100, unsqueeze_393);  sub_100 = unsqueeze_393 = None
        mul_674: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_399);  sub_101 = unsqueeze_399 = None
        mul_675: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, squeeze_277);  sum_5 = squeeze_277 = None
        convert_element_type_301: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_674, torch.bfloat16);  mul_674 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_301, relu_90, convert_element_type_277, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_301 = convert_element_type_277 = None
        getitem_199: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = convolution_backward_1[0]
        getitem_200: "bf16[384, 384, 3, 1][1152, 1, 384, 384]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_302: "f32[384, 384, 3, 1][1152, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_200, torch.float32);  getitem_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_91: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_91, getitem_191)
        mul_637: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = None
        unsqueeze_364: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_552, -1)
        unsqueeze_365: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_643: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_637, unsqueeze_365);  mul_637 = unsqueeze_365 = None
        unsqueeze_366: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_553, -1);  primals_553 = None
        unsqueeze_367: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_459: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_643, unsqueeze_367);  mul_643 = unsqueeze_367 = None
        convert_element_type_276: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_459, torch.bfloat16);  add_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_91: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_276);  convert_element_type_276 = None
        le_2: "b8[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_91, 0);  relu_91 = None
        where_2: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.where.self(le_2, full_default, slice_5);  le_2 = slice_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_303: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        squeeze_273: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_191, [0, 2, 3]);  getitem_191 = None
        unsqueeze_400: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_273, 0);  squeeze_273 = None
        unsqueeze_401: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
        unsqueeze_402: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
        sum_6: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_303, [0, 2, 3])
        convert_element_type_275: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_91, torch.float32);  convolution_91 = None
        sub_102: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_275, unsqueeze_402);  convert_element_type_275 = unsqueeze_402 = None
        mul_676: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_303, sub_102)
        sum_7: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_676, [0, 2, 3]);  mul_676 = None
        mul_677: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, 0.0001220703125)
        unsqueeze_403: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_677, 0);  mul_677 = None
        unsqueeze_404: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
        unsqueeze_405: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
        mul_678: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, 0.0001220703125)
        squeeze_274: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_91, [0, 2, 3]);  rsqrt_91 = None
        mul_679: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_274, squeeze_274)
        mul_680: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_678, mul_679);  mul_678 = mul_679 = None
        unsqueeze_406: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
        unsqueeze_407: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
        unsqueeze_408: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
        mul_681: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_274, primals_552);  primals_552 = None
        unsqueeze_409: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
        unsqueeze_410: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
        unsqueeze_411: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
        mul_682: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_408);  sub_102 = unsqueeze_408 = None
        sub_104: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_303, mul_682);  convert_element_type_303 = mul_682 = None
        sub_105: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_405);  sub_104 = unsqueeze_405 = None
        mul_683: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_411);  sub_105 = unsqueeze_411 = None
        mul_684: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, squeeze_274);  sum_7 = squeeze_274 = None
        convert_element_type_305: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_683, torch.bfloat16);  mul_683 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(convert_element_type_305, relu_90, convert_element_type_274, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_305 = convert_element_type_274 = None
        getitem_202: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = convolution_backward_2[0]
        getitem_203: "bf16[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        add_470: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_199, getitem_202);  getitem_199 = getitem_202 = None
        convert_element_type_306: "f32[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_203, torch.float32);  getitem_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_3: "b8[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_90, 0);  relu_90 = None
        where_3: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.where.self(le_3, full_default, add_470);  le_3 = add_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_307: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        sum_8: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_307, [0, 2, 3])
        convert_element_type_272: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_90, torch.float32);  convolution_90 = None
        sub_106: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_272, unsqueeze_414);  convert_element_type_272 = unsqueeze_414 = None
        mul_685: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_307, sub_106)
        sum_9: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_685, [0, 2, 3]);  mul_685 = None
        mul_686: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_8, 0.0001220703125)
        unsqueeze_415: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
        unsqueeze_416: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
        unsqueeze_417: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
        mul_687: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, 0.0001220703125)
        mul_688: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_271, squeeze_271)
        mul_689: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_687, mul_688);  mul_687 = mul_688 = None
        unsqueeze_418: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
        unsqueeze_419: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
        unsqueeze_420: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
        mul_690: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_271, primals_546);  primals_546 = None
        unsqueeze_421: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
        unsqueeze_422: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
        unsqueeze_423: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
        mul_691: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_420);  sub_106 = unsqueeze_420 = None
        sub_108: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_307, mul_691);  convert_element_type_307 = mul_691 = None
        sub_109: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_108, unsqueeze_417);  sub_108 = unsqueeze_417 = None
        mul_692: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_423);  sub_109 = unsqueeze_423 = None
        mul_693: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, squeeze_271);  sum_9 = squeeze_271 = None
        convert_element_type_309: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_692, torch.bfloat16);  mul_692 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(convert_element_type_309, relu_89, convert_element_type_271, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_309 = convert_element_type_271 = None
        getitem_205: "bf16[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = convolution_backward_3[0]
        getitem_206: "bf16[384, 448, 3, 3][4032, 1, 1344, 448]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_310: "f32[384, 448, 3, 3][4032, 1, 1344, 448]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_206, torch.float32);  getitem_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_4: "b8[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.le.Scalar(relu_89, 0);  relu_89 = None
        where_4: "bf16[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.where.self(le_4, full_default, getitem_205);  le_4 = getitem_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_311: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.float32);  where_4 = None
        sum_10: "f32[448][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_311, [0, 2, 3])
        convert_element_type_269: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_89, torch.float32);  convolution_89 = None
        sub_110: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_269, unsqueeze_426);  convert_element_type_269 = unsqueeze_426 = None
        mul_694: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_311, sub_110)
        sum_11: "f32[448][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_694, [0, 2, 3]);  mul_694 = None
        mul_695: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, 0.0001220703125)
        unsqueeze_427: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
        unsqueeze_428: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
        unsqueeze_429: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
        mul_696: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, 0.0001220703125)
        mul_697: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_268, squeeze_268)
        mul_698: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_696, mul_697);  mul_696 = mul_697 = None
        unsqueeze_430: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
        unsqueeze_431: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
        unsqueeze_432: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
        mul_699: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_268, primals_540);  primals_540 = None
        unsqueeze_433: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
        unsqueeze_434: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
        unsqueeze_435: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
        mul_700: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_432);  sub_110 = unsqueeze_432 = None
        sub_112: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_311, mul_700);  convert_element_type_311 = mul_700 = None
        sub_113: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.sub.Tensor(sub_112, unsqueeze_429);  sub_112 = unsqueeze_429 = None
        mul_701: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_435);  sub_113 = unsqueeze_435 = None
        mul_702: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, squeeze_268);  sum_11 = squeeze_268 = None
        convert_element_type_313: "bf16[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.prims.convert_element_type.default(mul_701, torch.bfloat16);  mul_701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(convert_element_type_313, cat_11, convert_element_type_268, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_313 = convert_element_type_268 = None
        getitem_208: "bf16[128, 2048, 8, 8][131072, 1, 16384, 2048]cuda:0" = convolution_backward_4[0]
        getitem_209: "bf16[448, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        add_471: "bf16[128, 2048, 8, 8][131072, 1, 16384, 2048]cuda:0" = torch.ops.aten.add.Tensor(avg_pool2d_backward, getitem_208);  avg_pool2d_backward = getitem_208 = None
        convert_element_type_314: "f32[448, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_209, torch.float32);  getitem_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:226 in _forward, code: branch3x3 = torch.cat(branch3x3, 1)
        slice_7: "bf16[128, 384, 8, 8][131072, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_2, 1, 0, 384)
        slice_8: "bf16[128, 384, 8, 8][131072, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_2, 1, 384, 768);  slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_88: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_88, getitem_185)
        mul_616: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = None
        unsqueeze_352: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_534, -1)
        unsqueeze_353: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        mul_622: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_616, unsqueeze_353);  mul_616 = unsqueeze_353 = None
        unsqueeze_354: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_535, -1);  primals_535 = None
        unsqueeze_355: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        add_444: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_622, unsqueeze_355);  mul_622 = unsqueeze_355 = None
        convert_element_type_267: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_444, torch.bfloat16);  add_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_88: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_267);  convert_element_type_267 = None
        le_5: "b8[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_88, 0);  relu_88 = None
        where_5: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.where.self(le_5, full_default, slice_8);  le_5 = slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_315: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        squeeze_264: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_185, [0, 2, 3]);  getitem_185 = None
        unsqueeze_436: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_264, 0);  squeeze_264 = None
        unsqueeze_437: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
        unsqueeze_438: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
        sum_12: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_315, [0, 2, 3])
        convert_element_type_266: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_88, torch.float32);  convolution_88 = None
        sub_114: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_266, unsqueeze_438);  convert_element_type_266 = unsqueeze_438 = None
        mul_703: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_315, sub_114)
        sum_13: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_703, [0, 2, 3]);  mul_703 = None
        mul_704: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, 0.0001220703125)
        unsqueeze_439: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_704, 0);  mul_704 = None
        unsqueeze_440: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
        unsqueeze_441: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
        mul_705: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, 0.0001220703125)
        squeeze_265: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_88, [0, 2, 3]);  rsqrt_88 = None
        mul_706: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_265, squeeze_265)
        mul_707: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_705, mul_706);  mul_705 = mul_706 = None
        unsqueeze_442: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_707, 0);  mul_707 = None
        unsqueeze_443: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
        unsqueeze_444: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
        mul_708: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_265, primals_534);  primals_534 = None
        unsqueeze_445: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
        unsqueeze_446: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
        unsqueeze_447: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
        mul_709: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_444);  sub_114 = unsqueeze_444 = None
        sub_116: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_315, mul_709);  convert_element_type_315 = mul_709 = None
        sub_117: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_116, unsqueeze_441);  sub_116 = unsqueeze_441 = None
        mul_710: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_447);  sub_117 = unsqueeze_447 = None
        mul_711: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, squeeze_265);  sum_13 = squeeze_265 = None
        convert_element_type_317: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_710, torch.bfloat16);  mul_710 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(convert_element_type_317, relu_86, convert_element_type_265, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_317 = convert_element_type_265 = None
        getitem_211: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = convolution_backward_5[0]
        getitem_212: "bf16[384, 384, 3, 1][1152, 1, 384, 384]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_318: "f32[384, 384, 3, 1][1152, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_212, torch.float32);  getitem_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_87: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_87, getitem_183)
        mul_609: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = None
        unsqueeze_348: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_528, -1)
        unsqueeze_349: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_615: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_609, unsqueeze_349);  mul_609 = unsqueeze_349 = None
        unsqueeze_350: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_529, -1);  primals_529 = None
        unsqueeze_351: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_439: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_615, unsqueeze_351);  mul_615 = unsqueeze_351 = None
        convert_element_type_264: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_439, torch.bfloat16);  add_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_87: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_264);  convert_element_type_264 = None
        le_6: "b8[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_87, 0);  relu_87 = None
        where_6: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.where.self(le_6, full_default, slice_7);  le_6 = slice_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_319: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_6, torch.float32);  where_6 = None
        squeeze_261: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_183, [0, 2, 3]);  getitem_183 = None
        unsqueeze_448: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_261, 0);  squeeze_261 = None
        unsqueeze_449: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
        unsqueeze_450: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
        sum_14: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_319, [0, 2, 3])
        convert_element_type_263: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_87, torch.float32);  convolution_87 = None
        sub_118: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_263, unsqueeze_450);  convert_element_type_263 = unsqueeze_450 = None
        mul_712: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_319, sub_118)
        sum_15: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_712, [0, 2, 3]);  mul_712 = None
        mul_713: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_14, 0.0001220703125)
        unsqueeze_451: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_713, 0);  mul_713 = None
        unsqueeze_452: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
        unsqueeze_453: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
        mul_714: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, 0.0001220703125)
        squeeze_262: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_87, [0, 2, 3]);  rsqrt_87 = None
        mul_715: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_262, squeeze_262)
        mul_716: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_714, mul_715);  mul_714 = mul_715 = None
        unsqueeze_454: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_716, 0);  mul_716 = None
        unsqueeze_455: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
        unsqueeze_456: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
        mul_717: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_262, primals_528);  primals_528 = None
        unsqueeze_457: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
        unsqueeze_458: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
        unsqueeze_459: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
        mul_718: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_456);  sub_118 = unsqueeze_456 = None
        sub_120: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_319, mul_718);  convert_element_type_319 = mul_718 = None
        sub_121: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_120, unsqueeze_453);  sub_120 = unsqueeze_453 = None
        mul_719: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_459);  sub_121 = unsqueeze_459 = None
        mul_720: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, squeeze_262);  sum_15 = squeeze_262 = None
        convert_element_type_321: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_719, torch.bfloat16);  mul_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(convert_element_type_321, relu_86, convert_element_type_262, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_321 = convert_element_type_262 = None
        getitem_214: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = convolution_backward_6[0]
        getitem_215: "bf16[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        add_472: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_211, getitem_214);  getitem_211 = getitem_214 = None
        convert_element_type_322: "f32[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_215, torch.float32);  getitem_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_7: "b8[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_86, 0);  relu_86 = None
        where_7: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.where.self(le_7, full_default, add_472);  le_7 = add_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_323: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.float32);  where_7 = None
        sum_16: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_323, [0, 2, 3])
        convert_element_type_260: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_86, torch.float32);  convolution_86 = None
        sub_122: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_260, unsqueeze_462);  convert_element_type_260 = unsqueeze_462 = None
        mul_721: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_323, sub_122)
        sum_17: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_721, [0, 2, 3]);  mul_721 = None
        mul_722: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, 0.0001220703125)
        unsqueeze_463: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
        unsqueeze_464: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
        unsqueeze_465: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
        mul_723: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, 0.0001220703125)
        mul_724: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
        mul_725: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_723, mul_724);  mul_723 = mul_724 = None
        unsqueeze_466: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_725, 0);  mul_725 = None
        unsqueeze_467: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
        unsqueeze_468: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
        mul_726: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_259, primals_522);  primals_522 = None
        unsqueeze_469: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
        unsqueeze_470: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
        unsqueeze_471: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
        mul_727: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_468);  sub_122 = unsqueeze_468 = None
        sub_124: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_323, mul_727);  convert_element_type_323 = mul_727 = None
        sub_125: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_124, unsqueeze_465);  sub_124 = unsqueeze_465 = None
        mul_728: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_471);  sub_125 = unsqueeze_471 = None
        mul_729: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, squeeze_259);  sum_17 = squeeze_259 = None
        convert_element_type_325: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_728, torch.bfloat16);  mul_728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(convert_element_type_325, cat_11, convert_element_type_259, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_325 = convert_element_type_259 = None
        getitem_217: "bf16[128, 2048, 8, 8][131072, 1, 16384, 2048]cuda:0" = convolution_backward_7[0]
        getitem_218: "bf16[384, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        add_473: "bf16[128, 2048, 8, 8][131072, 1, 16384, 2048]cuda:0" = torch.ops.aten.add.Tensor(add_471, getitem_217);  add_471 = getitem_217 = None
        convert_element_type_326: "f32[384, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_218, torch.float32);  getitem_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_85: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.sub.Tensor(convolution_85, getitem_179)
        mul_595: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = None
        unsqueeze_340: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_516, -1)
        unsqueeze_341: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_601: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_595, unsqueeze_341);  mul_595 = unsqueeze_341 = None
        unsqueeze_342: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_517, -1);  primals_517 = None
        unsqueeze_343: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_429: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.add.Tensor(mul_601, unsqueeze_343);  mul_601 = unsqueeze_343 = None
        convert_element_type_258: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(add_429, torch.bfloat16);  add_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_85: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.relu.default(convert_element_type_258);  convert_element_type_258 = None
        le_8: "b8[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.le.Scalar(relu_85, 0);  relu_85 = None
        where_8: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.where.self(le_8, full_default, slice_1);  le_8 = slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_327: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(where_8, torch.float32);  where_8 = None
        squeeze_255: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_179, [0, 2, 3]);  getitem_179 = None
        unsqueeze_472: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_255, 0);  squeeze_255 = None
        unsqueeze_473: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
        unsqueeze_474: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
        sum_18: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_327, [0, 2, 3])
        convert_element_type_257: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_85, torch.float32);  convolution_85 = None
        sub_126: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_257, unsqueeze_474);  convert_element_type_257 = unsqueeze_474 = None
        mul_730: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_327, sub_126)
        sum_19: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_730, [0, 2, 3]);  mul_730 = None
        mul_731: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_18, 0.0001220703125)
        unsqueeze_475: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_731, 0);  mul_731 = None
        unsqueeze_476: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
        unsqueeze_477: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
        mul_732: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, 0.0001220703125)
        squeeze_256: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_85, [0, 2, 3]);  rsqrt_85 = None
        mul_733: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
        mul_734: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_732, mul_733);  mul_732 = mul_733 = None
        unsqueeze_478: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
        unsqueeze_479: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
        unsqueeze_480: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
        mul_735: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_256, primals_516);  primals_516 = None
        unsqueeze_481: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_735, 0);  mul_735 = None
        unsqueeze_482: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
        unsqueeze_483: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
        mul_736: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_480);  sub_126 = unsqueeze_480 = None
        sub_128: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_327, mul_736);  convert_element_type_327 = mul_736 = None
        sub_129: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.sub.Tensor(sub_128, unsqueeze_477);  sub_128 = unsqueeze_477 = None
        mul_737: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_483);  sub_129 = unsqueeze_483 = None
        mul_738: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, squeeze_256);  sum_19 = squeeze_256 = None
        convert_element_type_329: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(mul_737, torch.bfloat16);  mul_737 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(convert_element_type_329, cat_11, convert_element_type_256, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_329 = cat_11 = convert_element_type_256 = None
        getitem_220: "bf16[128, 2048, 8, 8][131072, 1, 16384, 2048]cuda:0" = convolution_backward_8[0]
        getitem_221: "bf16[320, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        add_474: "bf16[128, 2048, 8, 8][131072, 1, 16384, 2048]cuda:0" = torch.ops.aten.add.Tensor(add_473, getitem_220);  add_473 = getitem_220 = None
        convert_element_type_330: "f32[320, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_221, torch.float32);  getitem_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:244 in forward, code: return torch.cat(outputs, 1)
        slice_9: "bf16[128, 320, 8, 8][131072, 1, 16384, 2048]cuda:0" = torch.ops.aten.slice.Tensor(add_474, 1, 0, 320)
        slice_10: "bf16[128, 768, 8, 8][131072, 1, 16384, 2048]cuda:0" = torch.ops.aten.slice.Tensor(add_474, 1, 320, 1088)
        slice_11: "bf16[128, 768, 8, 8][131072, 1, 16384, 2048]cuda:0" = torch.ops.aten.slice.Tensor(add_474, 1, 1088, 1856)
        slice_12: "bf16[128, 192, 8, 8][131072, 1, 16384, 2048]cuda:0" = torch.ops.aten.slice.Tensor(add_474, 1, 1856, 2048);  add_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_84: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_84, getitem_177)
        mul_588: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = None
        unsqueeze_336: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_510, -1)
        unsqueeze_337: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        mul_594: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_588, unsqueeze_337);  mul_588 = unsqueeze_337 = None
        unsqueeze_338: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_511, -1);  primals_511 = None
        unsqueeze_339: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        add_424: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_594, unsqueeze_339);  mul_594 = unsqueeze_339 = None
        convert_element_type_255: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_424, torch.bfloat16);  add_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_84: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_255);  convert_element_type_255 = None
        le_9: "b8[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_84, 0);  relu_84 = None
        where_9: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.where.self(le_9, full_default, slice_12);  le_9 = slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_331: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.float32);  where_9 = None
        squeeze_252: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_177, [0, 2, 3]);  getitem_177 = None
        unsqueeze_484: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_252, 0);  squeeze_252 = None
        unsqueeze_485: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
        unsqueeze_486: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
        sum_20: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_331, [0, 2, 3])
        convert_element_type_254: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_84, torch.float32);  convolution_84 = None
        sub_130: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_254, unsqueeze_486);  convert_element_type_254 = unsqueeze_486 = None
        mul_739: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_331, sub_130)
        sum_21: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_739, [0, 2, 3]);  mul_739 = None
        mul_740: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_20, 0.0001220703125)
        unsqueeze_487: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
        unsqueeze_488: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
        unsqueeze_489: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
        mul_741: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, 0.0001220703125)
        squeeze_253: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_84, [0, 2, 3]);  rsqrt_84 = None
        mul_742: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
        mul_743: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_741, mul_742);  mul_741 = mul_742 = None
        unsqueeze_490: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
        unsqueeze_491: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
        unsqueeze_492: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
        mul_744: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_253, primals_510);  primals_510 = None
        unsqueeze_493: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
        unsqueeze_494: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
        unsqueeze_495: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
        mul_745: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_492);  sub_130 = unsqueeze_492 = None
        sub_132: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_331, mul_745);  convert_element_type_331 = mul_745 = None
        sub_133: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_132, unsqueeze_489);  sub_132 = unsqueeze_489 = None
        mul_746: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_495);  sub_133 = unsqueeze_495 = None
        mul_747: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, squeeze_253);  sum_21 = squeeze_253 = None
        convert_element_type_333: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_746, torch.bfloat16);  mul_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(convert_element_type_333, avg_pool2d_7, convert_element_type_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_333 = avg_pool2d_7 = convert_element_type_253 = None
        getitem_223: "bf16[128, 1280, 8, 8][81920, 1, 10240, 1280]cuda:0" = convolution_backward_9[0]
        getitem_224: "bf16[192, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_334: "f32[192, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_224, torch.float32);  getitem_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:236 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_1: "bf16[128, 1280, 8, 8][81920, 1, 10240, 1280]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(getitem_223, cat_8, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:234 in _forward, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
        slice_13: "bf16[128, 384, 8, 8][131072, 1, 16384, 2048]cuda:0" = torch.ops.aten.slice.Tensor(slice_11, 1, 0, 384)
        slice_14: "bf16[128, 384, 8, 8][131072, 1, 16384, 2048]cuda:0" = torch.ops.aten.slice.Tensor(slice_11, 1, 384, 768);  slice_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_83: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_83, getitem_175)
        mul_581: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = None
        unsqueeze_332: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_504, -1)
        unsqueeze_333: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_587: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_581, unsqueeze_333);  mul_581 = unsqueeze_333 = None
        unsqueeze_334: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_505, -1);  primals_505 = None
        unsqueeze_335: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_419: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_587, unsqueeze_335);  mul_587 = unsqueeze_335 = None
        convert_element_type_252: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_419, torch.bfloat16);  add_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_83: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_252);  convert_element_type_252 = None
        le_10: "b8[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_83, 0);  relu_83 = None
        where_10: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.where.self(le_10, full_default, slice_14);  le_10 = slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_335: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_10, torch.float32);  where_10 = None
        squeeze_249: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_175, [0, 2, 3]);  getitem_175 = None
        unsqueeze_496: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_249, 0);  squeeze_249 = None
        unsqueeze_497: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
        unsqueeze_498: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
        sum_22: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_335, [0, 2, 3])
        convert_element_type_251: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_83, torch.float32);  convolution_83 = None
        sub_134: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_251, unsqueeze_498);  convert_element_type_251 = unsqueeze_498 = None
        mul_748: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_335, sub_134)
        sum_23: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_748, [0, 2, 3]);  mul_748 = None
        mul_749: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_22, 0.0001220703125)
        unsqueeze_499: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
        unsqueeze_500: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
        unsqueeze_501: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
        mul_750: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, 0.0001220703125)
        squeeze_250: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_83, [0, 2, 3]);  rsqrt_83 = None
        mul_751: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
        mul_752: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_750, mul_751);  mul_750 = mul_751 = None
        unsqueeze_502: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
        unsqueeze_503: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
        unsqueeze_504: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
        mul_753: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_250, primals_504);  primals_504 = None
        unsqueeze_505: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
        unsqueeze_506: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
        unsqueeze_507: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
        mul_754: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_504);  sub_134 = unsqueeze_504 = None
        sub_136: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_335, mul_754);  convert_element_type_335 = mul_754 = None
        sub_137: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_136, unsqueeze_501);  sub_136 = unsqueeze_501 = None
        mul_755: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_507);  sub_137 = unsqueeze_507 = None
        mul_756: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, squeeze_250);  sum_23 = squeeze_250 = None
        convert_element_type_337: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_755, torch.bfloat16);  mul_755 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(convert_element_type_337, relu_81, convert_element_type_250, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_337 = convert_element_type_250 = None
        getitem_226: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = convolution_backward_10[0]
        getitem_227: "bf16[384, 384, 3, 1][1152, 1, 384, 384]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_338: "f32[384, 384, 3, 1][1152, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_227, torch.float32);  getitem_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_82: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_82, getitem_173)
        mul_574: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = None
        unsqueeze_328: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_498, -1)
        unsqueeze_329: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        mul_580: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_574, unsqueeze_329);  mul_574 = unsqueeze_329 = None
        unsqueeze_330: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_499, -1);  primals_499 = None
        unsqueeze_331: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        add_414: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_580, unsqueeze_331);  mul_580 = unsqueeze_331 = None
        convert_element_type_249: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_414, torch.bfloat16);  add_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_82: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_249);  convert_element_type_249 = None
        le_11: "b8[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_82, 0);  relu_82 = None
        where_11: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.where.self(le_11, full_default, slice_13);  le_11 = slice_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_339: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.float32);  where_11 = None
        squeeze_246: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_173, [0, 2, 3]);  getitem_173 = None
        unsqueeze_508: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_246, 0);  squeeze_246 = None
        unsqueeze_509: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
        unsqueeze_510: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
        sum_24: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_339, [0, 2, 3])
        convert_element_type_248: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_82, torch.float32);  convolution_82 = None
        sub_138: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_248, unsqueeze_510);  convert_element_type_248 = unsqueeze_510 = None
        mul_757: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_339, sub_138)
        sum_25: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_757, [0, 2, 3]);  mul_757 = None
        mul_758: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_24, 0.0001220703125)
        unsqueeze_511: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
        unsqueeze_512: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
        unsqueeze_513: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
        mul_759: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, 0.0001220703125)
        squeeze_247: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_82, [0, 2, 3]);  rsqrt_82 = None
        mul_760: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
        mul_761: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_759, mul_760);  mul_759 = mul_760 = None
        unsqueeze_514: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
        unsqueeze_515: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
        unsqueeze_516: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
        mul_762: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_247, primals_498);  primals_498 = None
        unsqueeze_517: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
        unsqueeze_518: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
        unsqueeze_519: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
        mul_763: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_516);  sub_138 = unsqueeze_516 = None
        sub_140: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_339, mul_763);  convert_element_type_339 = mul_763 = None
        sub_141: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_140, unsqueeze_513);  sub_140 = unsqueeze_513 = None
        mul_764: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_519);  sub_141 = unsqueeze_519 = None
        mul_765: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, squeeze_247);  sum_25 = squeeze_247 = None
        convert_element_type_341: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_764, torch.bfloat16);  mul_764 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(convert_element_type_341, relu_81, convert_element_type_247, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_341 = convert_element_type_247 = None
        getitem_229: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = convolution_backward_11[0]
        getitem_230: "bf16[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        add_475: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_226, getitem_229);  getitem_226 = getitem_229 = None
        convert_element_type_342: "f32[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_230, torch.float32);  getitem_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_12: "b8[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_81, 0);  relu_81 = None
        where_12: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.where.self(le_12, full_default, add_475);  le_12 = add_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_343: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_12, torch.float32);  where_12 = None
        sum_26: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_343, [0, 2, 3])
        convert_element_type_245: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_81, torch.float32);  convolution_81 = None
        sub_142: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_245, unsqueeze_522);  convert_element_type_245 = unsqueeze_522 = None
        mul_766: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_343, sub_142)
        sum_27: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_766, [0, 2, 3]);  mul_766 = None
        mul_767: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_26, 0.0001220703125)
        unsqueeze_523: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_767, 0);  mul_767 = None
        unsqueeze_524: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
        unsqueeze_525: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
        mul_768: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, 0.0001220703125)
        mul_769: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
        mul_770: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_768, mul_769);  mul_768 = mul_769 = None
        unsqueeze_526: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
        unsqueeze_527: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
        unsqueeze_528: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
        mul_771: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_244, primals_492);  primals_492 = None
        unsqueeze_529: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
        unsqueeze_530: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
        unsqueeze_531: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
        mul_772: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_528);  sub_142 = unsqueeze_528 = None
        sub_144: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_343, mul_772);  convert_element_type_343 = mul_772 = None
        sub_145: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_144, unsqueeze_525);  sub_144 = unsqueeze_525 = None
        mul_773: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_531);  sub_145 = unsqueeze_531 = None
        mul_774: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, squeeze_244);  sum_27 = squeeze_244 = None
        convert_element_type_345: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_773, torch.bfloat16);  mul_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(convert_element_type_345, relu_80, convert_element_type_244, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_345 = convert_element_type_244 = None
        getitem_232: "bf16[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = convolution_backward_12[0]
        getitem_233: "bf16[384, 448, 3, 3][4032, 1, 1344, 448]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        convert_element_type_346: "f32[384, 448, 3, 3][4032, 1, 1344, 448]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_233, torch.float32);  getitem_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_13: "b8[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.le.Scalar(relu_80, 0);  relu_80 = None
        where_13: "bf16[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.where.self(le_13, full_default, getitem_232);  le_13 = getitem_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_347: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.prims.convert_element_type.default(where_13, torch.float32);  where_13 = None
        sum_28: "f32[448][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_347, [0, 2, 3])
        convert_element_type_242: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_80, torch.float32);  convolution_80 = None
        sub_146: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_242, unsqueeze_534);  convert_element_type_242 = unsqueeze_534 = None
        mul_775: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_347, sub_146)
        sum_29: "f32[448][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_775, [0, 2, 3]);  mul_775 = None
        mul_776: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, 0.0001220703125)
        unsqueeze_535: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
        unsqueeze_536: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
        unsqueeze_537: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
        mul_777: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, 0.0001220703125)
        mul_778: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
        mul_779: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_777, mul_778);  mul_777 = mul_778 = None
        unsqueeze_538: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_779, 0);  mul_779 = None
        unsqueeze_539: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
        unsqueeze_540: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
        mul_780: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_241, primals_486);  primals_486 = None
        unsqueeze_541: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_780, 0);  mul_780 = None
        unsqueeze_542: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
        unsqueeze_543: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
        mul_781: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_540);  sub_146 = unsqueeze_540 = None
        sub_148: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_347, mul_781);  convert_element_type_347 = mul_781 = None
        sub_149: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.sub.Tensor(sub_148, unsqueeze_537);  sub_148 = unsqueeze_537 = None
        mul_782: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_543);  sub_149 = unsqueeze_543 = None
        mul_783: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, squeeze_241);  sum_29 = squeeze_241 = None
        convert_element_type_349: "bf16[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.prims.convert_element_type.default(mul_782, torch.bfloat16);  mul_782 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(convert_element_type_349, cat_8, convert_element_type_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_349 = convert_element_type_241 = None
        getitem_235: "bf16[128, 1280, 8, 8][81920, 1, 10240, 1280]cuda:0" = convolution_backward_13[0]
        getitem_236: "bf16[448, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        add_476: "bf16[128, 1280, 8, 8][81920, 1, 10240, 1280]cuda:0" = torch.ops.aten.add.Tensor(avg_pool2d_backward_1, getitem_235);  avg_pool2d_backward_1 = getitem_235 = None
        convert_element_type_350: "f32[448, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_236, torch.float32);  getitem_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:226 in _forward, code: branch3x3 = torch.cat(branch3x3, 1)
        slice_15: "bf16[128, 384, 8, 8][131072, 1, 16384, 2048]cuda:0" = torch.ops.aten.slice.Tensor(slice_10, 1, 0, 384)
        slice_16: "bf16[128, 384, 8, 8][131072, 1, 16384, 2048]cuda:0" = torch.ops.aten.slice.Tensor(slice_10, 1, 384, 768);  slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_79: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_79, getitem_167)
        mul_553: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = None
        unsqueeze_316: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_480, -1)
        unsqueeze_317: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_559: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_553, unsqueeze_317);  mul_553 = unsqueeze_317 = None
        unsqueeze_318: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_481, -1);  primals_481 = None
        unsqueeze_319: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_399: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_559, unsqueeze_319);  mul_559 = unsqueeze_319 = None
        convert_element_type_240: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_399, torch.bfloat16);  add_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_79: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_240);  convert_element_type_240 = None
        le_14: "b8[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_79, 0);  relu_79 = None
        where_14: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.where.self(le_14, full_default, slice_16);  le_14 = slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_351: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_14, torch.float32);  where_14 = None
        squeeze_237: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
        unsqueeze_544: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_237, 0);  squeeze_237 = None
        unsqueeze_545: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
        unsqueeze_546: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
        sum_30: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_351, [0, 2, 3])
        convert_element_type_239: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_79, torch.float32);  convolution_79 = None
        sub_150: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_239, unsqueeze_546);  convert_element_type_239 = unsqueeze_546 = None
        mul_784: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_351, sub_150)
        sum_31: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_784, [0, 2, 3]);  mul_784 = None
        mul_785: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_30, 0.0001220703125)
        unsqueeze_547: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_785, 0);  mul_785 = None
        unsqueeze_548: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
        unsqueeze_549: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
        mul_786: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, 0.0001220703125)
        squeeze_238: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
        mul_787: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
        mul_788: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_786, mul_787);  mul_786 = mul_787 = None
        unsqueeze_550: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_788, 0);  mul_788 = None
        unsqueeze_551: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
        unsqueeze_552: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
        mul_789: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_238, primals_480);  primals_480 = None
        unsqueeze_553: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_789, 0);  mul_789 = None
        unsqueeze_554: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
        unsqueeze_555: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
        mul_790: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_552);  sub_150 = unsqueeze_552 = None
        sub_152: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_351, mul_790);  convert_element_type_351 = mul_790 = None
        sub_153: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_152, unsqueeze_549);  sub_152 = unsqueeze_549 = None
        mul_791: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_555);  sub_153 = unsqueeze_555 = None
        mul_792: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, squeeze_238);  sum_31 = squeeze_238 = None
        convert_element_type_353: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_791, torch.bfloat16);  mul_791 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(convert_element_type_353, relu_77, convert_element_type_238, [0], [1, 1], [1, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_353 = convert_element_type_238 = None
        getitem_238: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = convolution_backward_14[0]
        getitem_239: "bf16[384, 384, 3, 1][1152, 1, 384, 384]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        convert_element_type_354: "f32[384, 384, 3, 1][1152, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_239, torch.float32);  getitem_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_78: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_78, getitem_165)
        mul_546: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = None
        unsqueeze_312: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_474, -1)
        unsqueeze_313: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        mul_552: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_546, unsqueeze_313);  mul_546 = unsqueeze_313 = None
        unsqueeze_314: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_475, -1);  primals_475 = None
        unsqueeze_315: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        add_394: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_552, unsqueeze_315);  mul_552 = unsqueeze_315 = None
        convert_element_type_237: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_394, torch.bfloat16);  add_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_78: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_237);  convert_element_type_237 = None
        le_15: "b8[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_78, 0);  relu_78 = None
        where_15: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.where.self(le_15, full_default, slice_15);  le_15 = slice_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_355: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_15, torch.float32);  where_15 = None
        squeeze_234: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3]);  getitem_165 = None
        unsqueeze_556: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_234, 0);  squeeze_234 = None
        unsqueeze_557: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
        unsqueeze_558: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
        sum_32: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_355, [0, 2, 3])
        convert_element_type_236: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_78, torch.float32);  convolution_78 = None
        sub_154: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_236, unsqueeze_558);  convert_element_type_236 = unsqueeze_558 = None
        mul_793: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_355, sub_154)
        sum_33: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_793, [0, 2, 3]);  mul_793 = None
        mul_794: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_32, 0.0001220703125)
        unsqueeze_559: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_794, 0);  mul_794 = None
        unsqueeze_560: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
        unsqueeze_561: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
        mul_795: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, 0.0001220703125)
        squeeze_235: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
        mul_796: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
        mul_797: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_795, mul_796);  mul_795 = mul_796 = None
        unsqueeze_562: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
        unsqueeze_563: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
        unsqueeze_564: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
        mul_798: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_235, primals_474);  primals_474 = None
        unsqueeze_565: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
        unsqueeze_566: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
        unsqueeze_567: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
        mul_799: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_564);  sub_154 = unsqueeze_564 = None
        sub_156: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_355, mul_799);  convert_element_type_355 = mul_799 = None
        sub_157: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_156, unsqueeze_561);  sub_156 = unsqueeze_561 = None
        mul_800: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_567);  sub_157 = unsqueeze_567 = None
        mul_801: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, squeeze_235);  sum_33 = squeeze_235 = None
        convert_element_type_357: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_800, torch.bfloat16);  mul_800 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(convert_element_type_357, relu_77, convert_element_type_235, [0], [1, 1], [0, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_357 = convert_element_type_235 = None
        getitem_241: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = convolution_backward_15[0]
        getitem_242: "bf16[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        add_477: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_238, getitem_241);  getitem_238 = getitem_241 = None
        convert_element_type_358: "f32[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_242, torch.float32);  getitem_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_16: "b8[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_77, 0);  relu_77 = None
        where_16: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.where.self(le_16, full_default, add_477);  le_16 = add_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_359: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_16, torch.float32);  where_16 = None
        sum_34: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_359, [0, 2, 3])
        convert_element_type_233: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_77, torch.float32);  convolution_77 = None
        sub_158: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_233, unsqueeze_570);  convert_element_type_233 = unsqueeze_570 = None
        mul_802: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_359, sub_158)
        sum_35: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_802, [0, 2, 3]);  mul_802 = None
        mul_803: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, 0.0001220703125)
        unsqueeze_571: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
        unsqueeze_572: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
        unsqueeze_573: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
        mul_804: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, 0.0001220703125)
        mul_805: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
        mul_806: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_804, mul_805);  mul_804 = mul_805 = None
        unsqueeze_574: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
        unsqueeze_575: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
        unsqueeze_576: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
        mul_807: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_232, primals_468);  primals_468 = None
        unsqueeze_577: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
        unsqueeze_578: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
        unsqueeze_579: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
        mul_808: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_576);  sub_158 = unsqueeze_576 = None
        sub_160: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_359, mul_808);  convert_element_type_359 = mul_808 = None
        sub_161: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_160, unsqueeze_573);  sub_160 = unsqueeze_573 = None
        mul_809: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_579);  sub_161 = unsqueeze_579 = None
        mul_810: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, squeeze_232);  sum_35 = squeeze_232 = None
        convert_element_type_361: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_809, torch.bfloat16);  mul_809 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(convert_element_type_361, cat_8, convert_element_type_232, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_361 = convert_element_type_232 = None
        getitem_244: "bf16[128, 1280, 8, 8][81920, 1, 10240, 1280]cuda:0" = convolution_backward_16[0]
        getitem_245: "bf16[384, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        add_478: "bf16[128, 1280, 8, 8][81920, 1, 10240, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_476, getitem_244);  add_476 = getitem_244 = None
        convert_element_type_362: "f32[384, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_245, torch.float32);  getitem_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_76: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.sub.Tensor(convolution_76, getitem_161)
        mul_532: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = None
        unsqueeze_304: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_462, -1)
        unsqueeze_305: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        mul_538: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_532, unsqueeze_305);  mul_532 = unsqueeze_305 = None
        unsqueeze_306: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_463, -1);  primals_463 = None
        unsqueeze_307: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        add_384: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.add.Tensor(mul_538, unsqueeze_307);  mul_538 = unsqueeze_307 = None
        convert_element_type_231: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(add_384, torch.bfloat16);  add_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_76: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.relu.default(convert_element_type_231);  convert_element_type_231 = None
        le_17: "b8[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.le.Scalar(relu_76, 0);  relu_76 = None
        where_17: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.where.self(le_17, full_default, slice_9);  le_17 = slice_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_363: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(where_17, torch.float32);  where_17 = None
        squeeze_228: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_161, [0, 2, 3]);  getitem_161 = None
        unsqueeze_580: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_228, 0);  squeeze_228 = None
        unsqueeze_581: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
        unsqueeze_582: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
        sum_36: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_363, [0, 2, 3])
        convert_element_type_230: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_76, torch.float32);  convolution_76 = None
        sub_162: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_230, unsqueeze_582);  convert_element_type_230 = unsqueeze_582 = None
        mul_811: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_363, sub_162)
        sum_37: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_811, [0, 2, 3]);  mul_811 = None
        mul_812: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_36, 0.0001220703125)
        unsqueeze_583: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
        unsqueeze_584: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
        unsqueeze_585: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
        mul_813: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, 0.0001220703125)
        squeeze_229: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
        mul_814: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
        mul_815: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_813, mul_814);  mul_813 = mul_814 = None
        unsqueeze_586: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
        unsqueeze_587: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
        unsqueeze_588: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
        mul_816: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_229, primals_462);  primals_462 = None
        unsqueeze_589: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
        unsqueeze_590: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
        unsqueeze_591: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
        mul_817: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_588);  sub_162 = unsqueeze_588 = None
        sub_164: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_363, mul_817);  convert_element_type_363 = mul_817 = None
        sub_165: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.sub.Tensor(sub_164, unsqueeze_585);  sub_164 = unsqueeze_585 = None
        mul_818: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_591);  sub_165 = unsqueeze_591 = None
        mul_819: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, squeeze_229);  sum_37 = squeeze_229 = None
        convert_element_type_365: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(mul_818, torch.bfloat16);  mul_818 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(convert_element_type_365, cat_8, convert_element_type_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_365 = cat_8 = convert_element_type_229 = None
        getitem_247: "bf16[128, 1280, 8, 8][81920, 1, 10240, 1280]cuda:0" = convolution_backward_17[0]
        getitem_248: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        add_479: "bf16[128, 1280, 8, 8][81920, 1, 10240, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_478, getitem_247);  add_478 = getitem_247 = None
        convert_element_type_366: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_248, torch.float32);  getitem_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:190 in forward, code: return torch.cat(outputs, 1)
        slice_17: "bf16[128, 320, 8, 8][81920, 1, 10240, 1280]cuda:0" = torch.ops.aten.slice.Tensor(add_479, 1, 0, 320)
        slice_18: "bf16[128, 192, 8, 8][81920, 1, 10240, 1280]cuda:0" = torch.ops.aten.slice.Tensor(add_479, 1, 320, 512)
        slice_19: "bf16[128, 768, 8, 8][81920, 1, 10240, 1280]cuda:0" = torch.ops.aten.slice.Tensor(add_479, 1, 512, 1280);  add_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:184 in _forward, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        full_default_18: "f32[98304, 289][289, 1]cuda:0" = torch.ops.aten.full.default([98304, 289], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        clone_1: "bf16[128, 768, 8, 8][49152, 64, 8, 1]cuda:0" = torch.ops.aten.clone.default(slice_19, memory_format = torch.contiguous_format);  slice_19 = None
        view_3: "bf16[98304, 64][64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [98304, 64]);  clone_1 = None
        _low_memory_max_pool_offsets_to_indices_3: "i64[128, 768, 8, 8][49152, 1, 6144, 768]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_159, [3, 3], [17, 17], [2, 2], [0, 0], [1, 1]);  getitem_159 = None
        clone_2: "i64[128, 768, 8, 8][49152, 64, 8, 1]cuda:0" = torch.ops.aten.clone.default(_low_memory_max_pool_offsets_to_indices_3, memory_format = torch.contiguous_format);  _low_memory_max_pool_offsets_to_indices_3 = None
        view_4: "i64[98304, 64][64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [98304, 64]);  clone_2 = None
        convert_element_type_367: "f32[98304, 64][64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        scatter_add: "f32[98304, 289][289, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_18, 1, view_4, convert_element_type_367);  full_default_18 = view_4 = convert_element_type_367 = None
        view_5: "f32[128, 768, 17, 17][221952, 289, 17, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add, [128, 768, 17, 17]);  scatter_add = None
        convert_element_type_368: "bf16[128, 768, 17, 17][221952, 289, 17, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        clone_3: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.clone.default(convert_element_type_368, memory_format = torch.channels_last);  convert_element_type_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_75: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_75, getitem_157)
        mul_525: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = None
        unsqueeze_300: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_456, -1)
        unsqueeze_301: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_531: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_525, unsqueeze_301);  mul_525 = unsqueeze_301 = None
        unsqueeze_302: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_457, -1);  primals_457 = None
        unsqueeze_303: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_379: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_531, unsqueeze_303);  mul_531 = unsqueeze_303 = None
        convert_element_type_228: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_379, torch.bfloat16);  add_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_75: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_228);  convert_element_type_228 = None
        le_18: "b8[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_75, 0);  relu_75 = None
        where_18: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.where.self(le_18, full_default, slice_18);  le_18 = slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_369: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_18, torch.float32);  where_18 = None
        squeeze_225: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
        unsqueeze_592: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_225, 0);  squeeze_225 = None
        unsqueeze_593: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
        unsqueeze_594: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
        sum_38: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_369, [0, 2, 3])
        convert_element_type_227: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_75, torch.float32);  convolution_75 = None
        sub_166: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_227, unsqueeze_594);  convert_element_type_227 = unsqueeze_594 = None
        mul_820: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_369, sub_166)
        sum_39: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_820, [0, 2, 3]);  mul_820 = None
        mul_821: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_38, 0.0001220703125)
        unsqueeze_595: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
        unsqueeze_596: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
        unsqueeze_597: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
        mul_822: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, 0.0001220703125)
        squeeze_226: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
        mul_823: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
        mul_824: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_822, mul_823);  mul_822 = mul_823 = None
        unsqueeze_598: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_824, 0);  mul_824 = None
        unsqueeze_599: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
        unsqueeze_600: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
        mul_825: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_226, primals_456);  primals_456 = None
        unsqueeze_601: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_825, 0);  mul_825 = None
        unsqueeze_602: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
        unsqueeze_603: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
        mul_826: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_600);  sub_166 = unsqueeze_600 = None
        sub_168: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_369, mul_826);  convert_element_type_369 = mul_826 = None
        sub_169: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_168, unsqueeze_597);  sub_168 = unsqueeze_597 = None
        mul_827: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_603);  sub_169 = unsqueeze_603 = None
        mul_828: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, squeeze_226);  sum_39 = squeeze_226 = None
        convert_element_type_371: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_827, torch.bfloat16);  mul_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(convert_element_type_371, relu_74, convert_element_type_226, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_371 = convert_element_type_226 = None
        getitem_250: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = convolution_backward_18[0]
        getitem_251: "bf16[192, 192, 3, 3][1728, 1, 576, 192]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        convert_element_type_372: "f32[192, 192, 3, 3][1728, 1, 576, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_251, torch.float32);  getitem_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_19: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_74, 0);  relu_74 = None
        where_19: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_19, full_default, getitem_250);  le_19 = getitem_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_373: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_19, torch.float32);  where_19 = None
        sum_40: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_373, [0, 2, 3])
        convert_element_type_224: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_74, torch.float32);  convolution_74 = None
        sub_170: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_224, unsqueeze_606);  convert_element_type_224 = unsqueeze_606 = None
        mul_829: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_373, sub_170)
        sum_41: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_829, [0, 2, 3]);  mul_829 = None
        mul_830: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, 2.703287197231834e-05)
        unsqueeze_607: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_830, 0);  mul_830 = None
        unsqueeze_608: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
        unsqueeze_609: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
        mul_831: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, 2.703287197231834e-05)
        mul_832: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
        mul_833: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_831, mul_832);  mul_831 = mul_832 = None
        unsqueeze_610: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_833, 0);  mul_833 = None
        unsqueeze_611: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
        unsqueeze_612: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
        mul_834: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_223, primals_450);  primals_450 = None
        unsqueeze_613: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_834, 0);  mul_834 = None
        unsqueeze_614: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
        unsqueeze_615: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
        mul_835: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_612);  sub_170 = unsqueeze_612 = None
        sub_172: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_373, mul_835);  convert_element_type_373 = mul_835 = None
        sub_173: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_172, unsqueeze_609);  sub_172 = unsqueeze_609 = None
        mul_836: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_615);  sub_173 = unsqueeze_615 = None
        mul_837: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, squeeze_223);  sum_41 = squeeze_223 = None
        convert_element_type_375: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_836, torch.bfloat16);  mul_836 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(convert_element_type_375, relu_73, convert_element_type_223, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_375 = convert_element_type_223 = None
        getitem_253: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = convolution_backward_19[0]
        getitem_254: "bf16[192, 192, 7, 1][1344, 1, 192, 192]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_376: "f32[192, 192, 7, 1][1344, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_254, torch.float32);  getitem_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_20: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_73, 0);  relu_73 = None
        where_20: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_20, full_default, getitem_253);  le_20 = getitem_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_377: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_20, torch.float32);  where_20 = None
        sum_42: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_377, [0, 2, 3])
        convert_element_type_221: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_73, torch.float32);  convolution_73 = None
        sub_174: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_221, unsqueeze_618);  convert_element_type_221 = unsqueeze_618 = None
        mul_838: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_377, sub_174)
        sum_43: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_838, [0, 2, 3]);  mul_838 = None
        mul_839: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, 2.703287197231834e-05)
        unsqueeze_619: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_839, 0);  mul_839 = None
        unsqueeze_620: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
        unsqueeze_621: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
        mul_840: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, 2.703287197231834e-05)
        mul_841: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
        mul_842: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_840, mul_841);  mul_840 = mul_841 = None
        unsqueeze_622: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_842, 0);  mul_842 = None
        unsqueeze_623: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
        unsqueeze_624: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
        mul_843: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_220, primals_444);  primals_444 = None
        unsqueeze_625: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
        unsqueeze_626: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
        unsqueeze_627: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
        mul_844: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_624);  sub_174 = unsqueeze_624 = None
        sub_176: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_377, mul_844);  convert_element_type_377 = mul_844 = None
        sub_177: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_176, unsqueeze_621);  sub_176 = unsqueeze_621 = None
        mul_845: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_627);  sub_177 = unsqueeze_627 = None
        mul_846: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, squeeze_220);  sum_43 = squeeze_220 = None
        convert_element_type_379: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_845, torch.bfloat16);  mul_845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(convert_element_type_379, relu_72, convert_element_type_220, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_379 = convert_element_type_220 = None
        getitem_256: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = convolution_backward_20[0]
        getitem_257: "bf16[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        convert_element_type_380: "f32[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_257, torch.float32);  getitem_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_21: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_72, 0);  relu_72 = None
        where_21: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_21, full_default, getitem_256);  le_21 = getitem_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_381: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_21, torch.float32);  where_21 = None
        sum_44: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_381, [0, 2, 3])
        convert_element_type_218: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_72, torch.float32);  convolution_72 = None
        sub_178: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_218, unsqueeze_630);  convert_element_type_218 = unsqueeze_630 = None
        mul_847: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_381, sub_178)
        sum_45: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_847, [0, 2, 3]);  mul_847 = None
        mul_848: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_44, 2.703287197231834e-05)
        unsqueeze_631: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_848, 0);  mul_848 = None
        unsqueeze_632: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
        unsqueeze_633: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
        mul_849: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, 2.703287197231834e-05)
        mul_850: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
        mul_851: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_849, mul_850);  mul_849 = mul_850 = None
        unsqueeze_634: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_851, 0);  mul_851 = None
        unsqueeze_635: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
        unsqueeze_636: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
        mul_852: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_217, primals_438);  primals_438 = None
        unsqueeze_637: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
        unsqueeze_638: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
        unsqueeze_639: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
        mul_853: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_636);  sub_178 = unsqueeze_636 = None
        sub_180: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_381, mul_853);  convert_element_type_381 = mul_853 = None
        sub_181: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_180, unsqueeze_633);  sub_180 = unsqueeze_633 = None
        mul_854: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_639);  sub_181 = unsqueeze_639 = None
        mul_855: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, squeeze_217);  sum_45 = squeeze_217 = None
        convert_element_type_383: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_854, torch.bfloat16);  mul_854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(convert_element_type_383, cat_7, convert_element_type_217, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_383 = convert_element_type_217 = None
        getitem_259: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_21[0]
        getitem_260: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        add_480: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.add.Tensor(clone_3, getitem_259);  clone_3 = getitem_259 = None
        convert_element_type_384: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_260, torch.float32);  getitem_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_71: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.sub.Tensor(convolution_71, getitem_149)
        mul_497: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = None
        unsqueeze_284: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_432, -1)
        unsqueeze_285: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_503: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_497, unsqueeze_285);  mul_497 = unsqueeze_285 = None
        unsqueeze_286: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_433, -1);  primals_433 = None
        unsqueeze_287: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_359: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.add.Tensor(mul_503, unsqueeze_287);  mul_503 = unsqueeze_287 = None
        convert_element_type_216: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(add_359, torch.bfloat16);  add_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_71: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.relu.default(convert_element_type_216);  convert_element_type_216 = None
        le_22: "b8[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.le.Scalar(relu_71, 0);  relu_71 = None
        where_22: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.where.self(le_22, full_default, slice_17);  le_22 = slice_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_385: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(where_22, torch.float32);  where_22 = None
        squeeze_213: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
        unsqueeze_640: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_213, 0);  squeeze_213 = None
        unsqueeze_641: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
        unsqueeze_642: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
        sum_46: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_385, [0, 2, 3])
        convert_element_type_215: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_71, torch.float32);  convolution_71 = None
        sub_182: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_215, unsqueeze_642);  convert_element_type_215 = unsqueeze_642 = None
        mul_856: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_385, sub_182)
        sum_47: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_856, [0, 2, 3]);  mul_856 = None
        mul_857: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, 0.0001220703125)
        unsqueeze_643: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_857, 0);  mul_857 = None
        unsqueeze_644: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
        unsqueeze_645: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
        mul_858: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, 0.0001220703125)
        squeeze_214: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
        mul_859: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
        mul_860: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_858, mul_859);  mul_858 = mul_859 = None
        unsqueeze_646: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_860, 0);  mul_860 = None
        unsqueeze_647: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
        unsqueeze_648: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
        mul_861: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_214, primals_432);  primals_432 = None
        unsqueeze_649: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_861, 0);  mul_861 = None
        unsqueeze_650: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
        unsqueeze_651: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
        mul_862: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_648);  sub_182 = unsqueeze_648 = None
        sub_184: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_385, mul_862);  convert_element_type_385 = mul_862 = None
        sub_185: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.sub.Tensor(sub_184, unsqueeze_645);  sub_184 = unsqueeze_645 = None
        mul_863: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_651);  sub_185 = unsqueeze_651 = None
        mul_864: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, squeeze_214);  sum_47 = squeeze_214 = None
        convert_element_type_387: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(mul_863, torch.bfloat16);  mul_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(convert_element_type_387, relu_70, convert_element_type_214, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_387 = convert_element_type_214 = None
        getitem_262: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = convolution_backward_22[0]
        getitem_263: "bf16[320, 192, 3, 3][1728, 1, 576, 192]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_388: "f32[320, 192, 3, 3][1728, 1, 576, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_263, torch.float32);  getitem_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_23: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_70, 0);  relu_70 = None
        where_23: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_23, full_default, getitem_262);  le_23 = getitem_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_389: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_23, torch.float32);  where_23 = None
        sum_48: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_389, [0, 2, 3])
        convert_element_type_212: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_70, torch.float32);  convolution_70 = None
        sub_186: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_212, unsqueeze_654);  convert_element_type_212 = unsqueeze_654 = None
        mul_865: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_389, sub_186)
        sum_49: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_865, [0, 2, 3]);  mul_865 = None
        mul_866: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, 2.703287197231834e-05)
        unsqueeze_655: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_866, 0);  mul_866 = None
        unsqueeze_656: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
        unsqueeze_657: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
        mul_867: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, 2.703287197231834e-05)
        mul_868: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
        mul_869: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_867, mul_868);  mul_867 = mul_868 = None
        unsqueeze_658: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_869, 0);  mul_869 = None
        unsqueeze_659: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
        unsqueeze_660: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
        mul_870: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_211, primals_426);  primals_426 = None
        unsqueeze_661: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_870, 0);  mul_870 = None
        unsqueeze_662: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
        unsqueeze_663: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
        mul_871: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_660);  sub_186 = unsqueeze_660 = None
        sub_188: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_389, mul_871);  convert_element_type_389 = mul_871 = None
        sub_189: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_188, unsqueeze_657);  sub_188 = unsqueeze_657 = None
        mul_872: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_663);  sub_189 = unsqueeze_663 = None
        mul_873: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, squeeze_211);  sum_49 = squeeze_211 = None
        convert_element_type_391: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_872, torch.bfloat16);  mul_872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(convert_element_type_391, cat_7, convert_element_type_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_391 = cat_7 = convert_element_type_211 = None
        getitem_265: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_23[0]
        getitem_266: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        add_481: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.add.Tensor(add_480, getitem_265);  add_480 = getitem_265 = None
        convert_element_type_392: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_266, torch.float32);  getitem_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:152 in forward, code: return torch.cat(outputs, 1)
        slice_20: "bf16[128, 192, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_481, 1, 0, 192)
        slice_21: "bf16[128, 192, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_481, 1, 192, 384)
        slice_22: "bf16[128, 192, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_481, 1, 384, 576)
        slice_23: "bf16[128, 192, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_481, 1, 576, 768);  add_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_69: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_69, getitem_145)
        mul_483: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = None
        unsqueeze_276: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_420, -1)
        unsqueeze_277: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_489: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_483, unsqueeze_277);  mul_483 = unsqueeze_277 = None
        unsqueeze_278: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_421, -1);  primals_421 = None
        unsqueeze_279: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_349: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_489, unsqueeze_279);  mul_489 = unsqueeze_279 = None
        convert_element_type_210: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_349, torch.bfloat16);  add_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_69: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_210);  convert_element_type_210 = None
        le_24: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_69, 0);  relu_69 = None
        where_24: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_24, full_default, slice_23);  le_24 = slice_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_393: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_24, torch.float32);  where_24 = None
        squeeze_207: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
        unsqueeze_664: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_207, 0);  squeeze_207 = None
        unsqueeze_665: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
        unsqueeze_666: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
        sum_50: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_393, [0, 2, 3])
        convert_element_type_209: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_69, torch.float32);  convolution_69 = None
        sub_190: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_209, unsqueeze_666);  convert_element_type_209 = unsqueeze_666 = None
        mul_874: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_393, sub_190)
        sum_51: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_874, [0, 2, 3]);  mul_874 = None
        mul_875: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, 2.703287197231834e-05)
        unsqueeze_667: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
        unsqueeze_668: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
        unsqueeze_669: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
        mul_876: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, 2.703287197231834e-05)
        squeeze_208: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
        mul_877: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
        mul_878: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_876, mul_877);  mul_876 = mul_877 = None
        unsqueeze_670: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_878, 0);  mul_878 = None
        unsqueeze_671: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
        unsqueeze_672: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
        mul_879: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_208, primals_420);  primals_420 = None
        unsqueeze_673: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_879, 0);  mul_879 = None
        unsqueeze_674: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
        unsqueeze_675: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
        mul_880: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_672);  sub_190 = unsqueeze_672 = None
        sub_192: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_393, mul_880);  convert_element_type_393 = mul_880 = None
        sub_193: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_192, unsqueeze_669);  sub_192 = unsqueeze_669 = None
        mul_881: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_675);  sub_193 = unsqueeze_675 = None
        mul_882: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, squeeze_208);  sum_51 = squeeze_208 = None
        convert_element_type_395: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_881, torch.bfloat16);  mul_881 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(convert_element_type_395, avg_pool2d_6, convert_element_type_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_395 = avg_pool2d_6 = convert_element_type_208 = None
        getitem_268: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_24[0]
        getitem_269: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        convert_element_type_396: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_269, torch.float32);  getitem_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:144 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_2: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(getitem_268, cat_6, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_68: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_68, getitem_143)
        mul_476: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = None
        unsqueeze_272: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_414, -1)
        unsqueeze_273: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        mul_482: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_476, unsqueeze_273);  mul_476 = unsqueeze_273 = None
        unsqueeze_274: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_415, -1);  primals_415 = None
        unsqueeze_275: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        add_344: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_482, unsqueeze_275);  mul_482 = unsqueeze_275 = None
        convert_element_type_207: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_344, torch.bfloat16);  add_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_68: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_207);  convert_element_type_207 = None
        le_25: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_68, 0);  relu_68 = None
        where_25: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_25, full_default, slice_22);  le_25 = slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_397: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_25, torch.float32);  where_25 = None
        squeeze_204: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
        unsqueeze_676: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_204, 0);  squeeze_204 = None
        unsqueeze_677: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
        unsqueeze_678: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
        sum_52: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_397, [0, 2, 3])
        convert_element_type_206: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_68, torch.float32);  convolution_68 = None
        sub_194: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_206, unsqueeze_678);  convert_element_type_206 = unsqueeze_678 = None
        mul_883: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_397, sub_194)
        sum_53: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_883, [0, 2, 3]);  mul_883 = None
        mul_884: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, 2.703287197231834e-05)
        unsqueeze_679: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
        unsqueeze_680: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
        unsqueeze_681: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
        mul_885: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, 2.703287197231834e-05)
        squeeze_205: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
        mul_886: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
        mul_887: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_885, mul_886);  mul_885 = mul_886 = None
        unsqueeze_682: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_887, 0);  mul_887 = None
        unsqueeze_683: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
        unsqueeze_684: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
        mul_888: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_205, primals_414);  primals_414 = None
        unsqueeze_685: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_888, 0);  mul_888 = None
        unsqueeze_686: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
        unsqueeze_687: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
        mul_889: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_684);  sub_194 = unsqueeze_684 = None
        sub_196: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_397, mul_889);  convert_element_type_397 = mul_889 = None
        sub_197: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_196, unsqueeze_681);  sub_196 = unsqueeze_681 = None
        mul_890: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_687);  sub_197 = unsqueeze_687 = None
        mul_891: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, squeeze_205);  sum_53 = squeeze_205 = None
        convert_element_type_399: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_890, torch.bfloat16);  mul_890 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(convert_element_type_399, relu_67, convert_element_type_205, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_399 = convert_element_type_205 = None
        getitem_271: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = convolution_backward_25[0]
        getitem_272: "bf16[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_400: "f32[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_272, torch.float32);  getitem_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_26: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_67, 0);  relu_67 = None
        where_26: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_26, full_default, getitem_271);  le_26 = getitem_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_401: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_26, torch.float32);  where_26 = None
        sum_54: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_401, [0, 2, 3])
        convert_element_type_203: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_67, torch.float32);  convolution_67 = None
        sub_198: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_203, unsqueeze_690);  convert_element_type_203 = unsqueeze_690 = None
        mul_892: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_401, sub_198)
        sum_55: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_892, [0, 2, 3]);  mul_892 = None
        mul_893: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_54, 2.703287197231834e-05)
        unsqueeze_691: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_893, 0);  mul_893 = None
        unsqueeze_692: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
        unsqueeze_693: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
        mul_894: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, 2.703287197231834e-05)
        mul_895: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
        mul_896: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_894, mul_895);  mul_894 = mul_895 = None
        unsqueeze_694: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_896, 0);  mul_896 = None
        unsqueeze_695: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
        unsqueeze_696: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
        mul_897: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_202, primals_408);  primals_408 = None
        unsqueeze_697: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_897, 0);  mul_897 = None
        unsqueeze_698: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
        unsqueeze_699: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
        mul_898: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_696);  sub_198 = unsqueeze_696 = None
        sub_200: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_401, mul_898);  convert_element_type_401 = mul_898 = None
        sub_201: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_200, unsqueeze_693);  sub_200 = unsqueeze_693 = None
        mul_899: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_699);  sub_201 = unsqueeze_699 = None
        mul_900: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, squeeze_202);  sum_55 = squeeze_202 = None
        convert_element_type_403: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_899, torch.bfloat16);  mul_899 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(convert_element_type_403, relu_66, convert_element_type_202, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_403 = convert_element_type_202 = None
        getitem_274: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = convolution_backward_26[0]
        getitem_275: "bf16[192, 192, 7, 1][1344, 1, 192, 192]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None
        convert_element_type_404: "f32[192, 192, 7, 1][1344, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_275, torch.float32);  getitem_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_27: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_66, 0);  relu_66 = None
        where_27: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_27, full_default, getitem_274);  le_27 = getitem_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_405: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_27, torch.float32);  where_27 = None
        sum_56: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_405, [0, 2, 3])
        convert_element_type_200: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_66, torch.float32);  convolution_66 = None
        sub_202: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_200, unsqueeze_702);  convert_element_type_200 = unsqueeze_702 = None
        mul_901: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_405, sub_202)
        sum_57: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_901, [0, 2, 3]);  mul_901 = None
        mul_902: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_56, 2.703287197231834e-05)
        unsqueeze_703: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_902, 0);  mul_902 = None
        unsqueeze_704: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
        unsqueeze_705: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
        mul_903: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, 2.703287197231834e-05)
        mul_904: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
        mul_905: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_903, mul_904);  mul_903 = mul_904 = None
        unsqueeze_706: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_905, 0);  mul_905 = None
        unsqueeze_707: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
        unsqueeze_708: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
        mul_906: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_199, primals_402);  primals_402 = None
        unsqueeze_709: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_906, 0);  mul_906 = None
        unsqueeze_710: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
        unsqueeze_711: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
        mul_907: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_708);  sub_202 = unsqueeze_708 = None
        sub_204: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_405, mul_907);  convert_element_type_405 = mul_907 = None
        sub_205: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_204, unsqueeze_705);  sub_204 = unsqueeze_705 = None
        mul_908: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_711);  sub_205 = unsqueeze_711 = None
        mul_909: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, squeeze_199);  sum_57 = squeeze_199 = None
        convert_element_type_407: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_908, torch.bfloat16);  mul_908 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(convert_element_type_407, relu_65, convert_element_type_199, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_407 = convert_element_type_199 = None
        getitem_277: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = convolution_backward_27[0]
        getitem_278: "bf16[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        convert_element_type_408: "f32[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_278, torch.float32);  getitem_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_28: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_65, 0);  relu_65 = None
        where_28: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_28, full_default, getitem_277);  le_28 = getitem_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_409: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_28, torch.float32);  where_28 = None
        sum_58: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_409, [0, 2, 3])
        convert_element_type_197: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_65, torch.float32);  convolution_65 = None
        sub_206: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_197, unsqueeze_714);  convert_element_type_197 = unsqueeze_714 = None
        mul_910: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_409, sub_206)
        sum_59: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_910, [0, 2, 3]);  mul_910 = None
        mul_911: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, 2.703287197231834e-05)
        unsqueeze_715: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_911, 0);  mul_911 = None
        unsqueeze_716: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
        unsqueeze_717: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
        mul_912: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, 2.703287197231834e-05)
        mul_913: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
        mul_914: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_912, mul_913);  mul_912 = mul_913 = None
        unsqueeze_718: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_914, 0);  mul_914 = None
        unsqueeze_719: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
        unsqueeze_720: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
        mul_915: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_196, primals_396);  primals_396 = None
        unsqueeze_721: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_915, 0);  mul_915 = None
        unsqueeze_722: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
        unsqueeze_723: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
        mul_916: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_720);  sub_206 = unsqueeze_720 = None
        sub_208: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_409, mul_916);  convert_element_type_409 = mul_916 = None
        sub_209: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_208, unsqueeze_717);  sub_208 = unsqueeze_717 = None
        mul_917: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_723);  sub_209 = unsqueeze_723 = None
        mul_918: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, squeeze_196);  sum_59 = squeeze_196 = None
        convert_element_type_411: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_917, torch.bfloat16);  mul_917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(convert_element_type_411, relu_64, convert_element_type_196, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_411 = convert_element_type_196 = None
        getitem_280: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = convolution_backward_28[0]
        getitem_281: "bf16[192, 192, 7, 1][1344, 1, 192, 192]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None
        convert_element_type_412: "f32[192, 192, 7, 1][1344, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_281, torch.float32);  getitem_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_29: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_64, 0);  relu_64 = None
        where_29: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_29, full_default, getitem_280);  le_29 = getitem_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_413: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_29, torch.float32);  where_29 = None
        sum_60: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_413, [0, 2, 3])
        convert_element_type_194: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_64, torch.float32);  convolution_64 = None
        sub_210: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_194, unsqueeze_726);  convert_element_type_194 = unsqueeze_726 = None
        mul_919: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_413, sub_210)
        sum_61: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_919, [0, 2, 3]);  mul_919 = None
        mul_920: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_60, 2.703287197231834e-05)
        unsqueeze_727: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_920, 0);  mul_920 = None
        unsqueeze_728: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
        unsqueeze_729: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
        mul_921: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, 2.703287197231834e-05)
        mul_922: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
        mul_923: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_921, mul_922);  mul_921 = mul_922 = None
        unsqueeze_730: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_923, 0);  mul_923 = None
        unsqueeze_731: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
        unsqueeze_732: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
        mul_924: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_193, primals_390);  primals_390 = None
        unsqueeze_733: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_924, 0);  mul_924 = None
        unsqueeze_734: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
        unsqueeze_735: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
        mul_925: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_732);  sub_210 = unsqueeze_732 = None
        sub_212: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_413, mul_925);  convert_element_type_413 = mul_925 = None
        sub_213: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_212, unsqueeze_729);  sub_212 = unsqueeze_729 = None
        mul_926: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_735);  sub_213 = unsqueeze_735 = None
        mul_927: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, squeeze_193);  sum_61 = squeeze_193 = None
        convert_element_type_415: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_926, torch.bfloat16);  mul_926 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(convert_element_type_415, cat_6, convert_element_type_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_415 = convert_element_type_193 = None
        getitem_283: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_29[0]
        getitem_284: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None
        add_482: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.add.Tensor(avg_pool2d_backward_2, getitem_283);  avg_pool2d_backward_2 = getitem_283 = None
        convert_element_type_416: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_284, torch.float32);  getitem_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_63: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_63, getitem_133)
        mul_441: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
        unsqueeze_252: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_384, -1)
        unsqueeze_253: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_447: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_441, unsqueeze_253);  mul_441 = unsqueeze_253 = None
        unsqueeze_254: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_385, -1);  primals_385 = None
        unsqueeze_255: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_319: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_447, unsqueeze_255);  mul_447 = unsqueeze_255 = None
        convert_element_type_192: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_319, torch.bfloat16);  add_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_63: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_192);  convert_element_type_192 = None
        le_30: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_63, 0);  relu_63 = None
        where_30: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_30, full_default, slice_21);  le_30 = slice_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_417: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_30, torch.float32);  where_30 = None
        squeeze_189: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
        unsqueeze_736: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
        unsqueeze_737: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
        unsqueeze_738: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
        sum_62: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_417, [0, 2, 3])
        convert_element_type_191: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_63, torch.float32);  convolution_63 = None
        sub_214: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_191, unsqueeze_738);  convert_element_type_191 = unsqueeze_738 = None
        mul_928: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_417, sub_214)
        sum_63: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_928, [0, 2, 3]);  mul_928 = None
        mul_929: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_62, 2.703287197231834e-05)
        unsqueeze_739: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_929, 0);  mul_929 = None
        unsqueeze_740: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
        unsqueeze_741: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
        mul_930: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, 2.703287197231834e-05)
        squeeze_190: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
        mul_931: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
        mul_932: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_930, mul_931);  mul_930 = mul_931 = None
        unsqueeze_742: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
        unsqueeze_743: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
        unsqueeze_744: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
        mul_933: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_190, primals_384);  primals_384 = None
        unsqueeze_745: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_933, 0);  mul_933 = None
        unsqueeze_746: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
        unsqueeze_747: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
        mul_934: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_744);  sub_214 = unsqueeze_744 = None
        sub_216: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_417, mul_934);  convert_element_type_417 = mul_934 = None
        sub_217: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_741);  sub_216 = unsqueeze_741 = None
        mul_935: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_747);  sub_217 = unsqueeze_747 = None
        mul_936: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, squeeze_190);  sum_63 = squeeze_190 = None
        convert_element_type_419: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_935, torch.bfloat16);  mul_935 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(convert_element_type_419, relu_62, convert_element_type_190, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_419 = convert_element_type_190 = None
        getitem_286: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = convolution_backward_30[0]
        getitem_287: "bf16[192, 192, 7, 1][1344, 1, 192, 192]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        convert_element_type_420: "f32[192, 192, 7, 1][1344, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_287, torch.float32);  getitem_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_31: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_62, 0);  relu_62 = None
        where_31: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_31, full_default, getitem_286);  le_31 = getitem_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_421: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_31, torch.float32);  where_31 = None
        sum_64: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_421, [0, 2, 3])
        convert_element_type_188: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_62, torch.float32);  convolution_62 = None
        sub_218: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_188, unsqueeze_750);  convert_element_type_188 = unsqueeze_750 = None
        mul_937: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_421, sub_218)
        sum_65: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_937, [0, 2, 3]);  mul_937 = None
        mul_938: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, 2.703287197231834e-05)
        unsqueeze_751: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_938, 0);  mul_938 = None
        unsqueeze_752: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
        unsqueeze_753: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
        mul_939: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, 2.703287197231834e-05)
        mul_940: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
        mul_941: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_939, mul_940);  mul_939 = mul_940 = None
        unsqueeze_754: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_941, 0);  mul_941 = None
        unsqueeze_755: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
        unsqueeze_756: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
        mul_942: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_187, primals_378);  primals_378 = None
        unsqueeze_757: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_942, 0);  mul_942 = None
        unsqueeze_758: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
        unsqueeze_759: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
        mul_943: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_756);  sub_218 = unsqueeze_756 = None
        sub_220: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_421, mul_943);  convert_element_type_421 = mul_943 = None
        sub_221: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_220, unsqueeze_753);  sub_220 = unsqueeze_753 = None
        mul_944: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_759);  sub_221 = unsqueeze_759 = None
        mul_945: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, squeeze_187);  sum_65 = squeeze_187 = None
        convert_element_type_423: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_944, torch.bfloat16);  mul_944 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(convert_element_type_423, relu_61, convert_element_type_187, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_423 = convert_element_type_187 = None
        getitem_289: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = convolution_backward_31[0]
        getitem_290: "bf16[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None
        convert_element_type_424: "f32[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_290, torch.float32);  getitem_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_32: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_61, 0);  relu_61 = None
        where_32: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_32, full_default, getitem_289);  le_32 = getitem_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_425: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_32, torch.float32);  where_32 = None
        sum_66: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_425, [0, 2, 3])
        convert_element_type_185: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_61, torch.float32);  convolution_61 = None
        sub_222: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_185, unsqueeze_762);  convert_element_type_185 = unsqueeze_762 = None
        mul_946: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_425, sub_222)
        sum_67: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_946, [0, 2, 3]);  mul_946 = None
        mul_947: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_66, 2.703287197231834e-05)
        unsqueeze_763: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_947, 0);  mul_947 = None
        unsqueeze_764: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
        unsqueeze_765: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
        mul_948: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, 2.703287197231834e-05)
        mul_949: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
        mul_950: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_948, mul_949);  mul_948 = mul_949 = None
        unsqueeze_766: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_950, 0);  mul_950 = None
        unsqueeze_767: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
        unsqueeze_768: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
        mul_951: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_184, primals_372);  primals_372 = None
        unsqueeze_769: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_951, 0);  mul_951 = None
        unsqueeze_770: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
        unsqueeze_771: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
        mul_952: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_768);  sub_222 = unsqueeze_768 = None
        sub_224: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_425, mul_952);  convert_element_type_425 = mul_952 = None
        sub_225: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_224, unsqueeze_765);  sub_224 = unsqueeze_765 = None
        mul_953: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_771);  sub_225 = unsqueeze_771 = None
        mul_954: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, squeeze_184);  sum_67 = squeeze_184 = None
        convert_element_type_427: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_953, torch.bfloat16);  mul_953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(convert_element_type_427, cat_6, convert_element_type_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_427 = convert_element_type_184 = None
        getitem_292: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_32[0]
        getitem_293: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None
        add_483: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.add.Tensor(add_482, getitem_292);  add_482 = getitem_292 = None
        convert_element_type_428: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_293, torch.float32);  getitem_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_60: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_60, getitem_127)
        mul_420: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
        unsqueeze_240: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_366, -1)
        unsqueeze_241: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        mul_426: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_241);  mul_420 = unsqueeze_241 = None
        unsqueeze_242: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_367, -1);  primals_367 = None
        unsqueeze_243: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        add_304: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_426, unsqueeze_243);  mul_426 = unsqueeze_243 = None
        convert_element_type_183: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_304, torch.bfloat16);  add_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_60: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_183);  convert_element_type_183 = None
        le_33: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None
        where_33: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_33, full_default, slice_20);  le_33 = slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_429: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_33, torch.float32);  where_33 = None
        squeeze_180: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
        unsqueeze_772: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
        unsqueeze_773: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
        unsqueeze_774: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
        sum_68: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_429, [0, 2, 3])
        convert_element_type_182: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_60, torch.float32);  convolution_60 = None
        sub_226: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_182, unsqueeze_774);  convert_element_type_182 = unsqueeze_774 = None
        mul_955: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_429, sub_226)
        sum_69: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_955, [0, 2, 3]);  mul_955 = None
        mul_956: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_68, 2.703287197231834e-05)
        unsqueeze_775: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_956, 0);  mul_956 = None
        unsqueeze_776: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
        unsqueeze_777: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
        mul_957: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, 2.703287197231834e-05)
        squeeze_181: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
        mul_958: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
        mul_959: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_957, mul_958);  mul_957 = mul_958 = None
        unsqueeze_778: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_959, 0);  mul_959 = None
        unsqueeze_779: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
        unsqueeze_780: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
        mul_960: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_181, primals_366);  primals_366 = None
        unsqueeze_781: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_960, 0);  mul_960 = None
        unsqueeze_782: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
        unsqueeze_783: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
        mul_961: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_780);  sub_226 = unsqueeze_780 = None
        sub_228: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_429, mul_961);  convert_element_type_429 = mul_961 = None
        sub_229: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_228, unsqueeze_777);  sub_228 = unsqueeze_777 = None
        mul_962: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_783);  sub_229 = unsqueeze_783 = None
        mul_963: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, squeeze_181);  sum_69 = squeeze_181 = None
        convert_element_type_431: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_962, torch.bfloat16);  mul_962 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(convert_element_type_431, cat_6, convert_element_type_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_431 = cat_6 = convert_element_type_181 = None
        getitem_295: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_33[0]
        getitem_296: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None
        add_484: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.add.Tensor(add_483, getitem_295);  add_483 = getitem_295 = None
        convert_element_type_432: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_296, torch.float32);  getitem_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:152 in forward, code: return torch.cat(outputs, 1)
        slice_24: "bf16[128, 192, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_484, 1, 0, 192)
        slice_25: "bf16[128, 192, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_484, 1, 192, 384)
        slice_26: "bf16[128, 192, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_484, 1, 384, 576)
        slice_27: "bf16[128, 192, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_484, 1, 576, 768);  add_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_59: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_59, getitem_125)
        mul_413: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
        unsqueeze_236: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_360, -1)
        unsqueeze_237: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_419: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_237);  mul_413 = unsqueeze_237 = None
        unsqueeze_238: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_361, -1);  primals_361 = None
        unsqueeze_239: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_299: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_239);  mul_419 = unsqueeze_239 = None
        convert_element_type_180: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_299, torch.bfloat16);  add_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_59: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_180);  convert_element_type_180 = None
        le_34: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_59, 0);  relu_59 = None
        where_34: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_34, full_default, slice_27);  le_34 = slice_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_433: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_34, torch.float32);  where_34 = None
        squeeze_177: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
        unsqueeze_784: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
        unsqueeze_785: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
        unsqueeze_786: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
        sum_70: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_433, [0, 2, 3])
        convert_element_type_179: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_59, torch.float32);  convolution_59 = None
        sub_230: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_179, unsqueeze_786);  convert_element_type_179 = unsqueeze_786 = None
        mul_964: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_433, sub_230)
        sum_71: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_964, [0, 2, 3]);  mul_964 = None
        mul_965: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_70, 2.703287197231834e-05)
        unsqueeze_787: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_965, 0);  mul_965 = None
        unsqueeze_788: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
        unsqueeze_789: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
        mul_966: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, 2.703287197231834e-05)
        squeeze_178: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
        mul_967: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
        mul_968: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_966, mul_967);  mul_966 = mul_967 = None
        unsqueeze_790: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_968, 0);  mul_968 = None
        unsqueeze_791: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
        unsqueeze_792: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
        mul_969: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_178, primals_360);  primals_360 = None
        unsqueeze_793: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_969, 0);  mul_969 = None
        unsqueeze_794: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
        unsqueeze_795: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
        mul_970: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_792);  sub_230 = unsqueeze_792 = None
        sub_232: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_433, mul_970);  convert_element_type_433 = mul_970 = None
        sub_233: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_232, unsqueeze_789);  sub_232 = unsqueeze_789 = None
        mul_971: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_795);  sub_233 = unsqueeze_795 = None
        mul_972: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, squeeze_178);  sum_71 = squeeze_178 = None
        convert_element_type_435: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_971, torch.bfloat16);  mul_971 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(convert_element_type_435, avg_pool2d_5, convert_element_type_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_435 = avg_pool2d_5 = convert_element_type_178 = None
        getitem_298: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_34[0]
        getitem_299: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        convert_element_type_436: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_299, torch.float32);  getitem_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:144 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_3: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(getitem_298, cat_5, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_58: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_58, getitem_123)
        mul_406: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
        unsqueeze_232: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_354, -1)
        unsqueeze_233: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        mul_412: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_233);  mul_406 = unsqueeze_233 = None
        unsqueeze_234: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_355, -1);  primals_355 = None
        unsqueeze_235: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        add_294: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_412, unsqueeze_235);  mul_412 = unsqueeze_235 = None
        convert_element_type_177: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_294, torch.bfloat16);  add_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_58: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_177);  convert_element_type_177 = None
        le_35: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_58, 0);  relu_58 = None
        where_35: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_35, full_default, slice_26);  le_35 = slice_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_437: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_35, torch.float32);  where_35 = None
        squeeze_174: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
        unsqueeze_796: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
        unsqueeze_797: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
        unsqueeze_798: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None
        sum_72: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_437, [0, 2, 3])
        convert_element_type_176: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_58, torch.float32);  convolution_58 = None
        sub_234: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_176, unsqueeze_798);  convert_element_type_176 = unsqueeze_798 = None
        mul_973: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_437, sub_234)
        sum_73: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_973, [0, 2, 3]);  mul_973 = None
        mul_974: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_72, 2.703287197231834e-05)
        unsqueeze_799: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_974, 0);  mul_974 = None
        unsqueeze_800: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
        unsqueeze_801: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
        mul_975: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, 2.703287197231834e-05)
        squeeze_175: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
        mul_976: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
        mul_977: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_975, mul_976);  mul_975 = mul_976 = None
        unsqueeze_802: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_977, 0);  mul_977 = None
        unsqueeze_803: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
        unsqueeze_804: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
        mul_978: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_175, primals_354);  primals_354 = None
        unsqueeze_805: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_978, 0);  mul_978 = None
        unsqueeze_806: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
        unsqueeze_807: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
        mul_979: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_804);  sub_234 = unsqueeze_804 = None
        sub_236: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_437, mul_979);  convert_element_type_437 = mul_979 = None
        sub_237: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_236, unsqueeze_801);  sub_236 = unsqueeze_801 = None
        mul_980: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_807);  sub_237 = unsqueeze_807 = None
        mul_981: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, squeeze_175);  sum_73 = squeeze_175 = None
        convert_element_type_439: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_980, torch.bfloat16);  mul_980 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(convert_element_type_439, relu_57, convert_element_type_175, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_439 = convert_element_type_175 = None
        getitem_301: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = convolution_backward_35[0]
        getitem_302: "bf16[192, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None
        convert_element_type_440: "f32[192, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_302, torch.float32);  getitem_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_36: "b8[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.le.Scalar(relu_57, 0);  relu_57 = None
        where_36: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.where.self(le_36, full_default, getitem_301);  le_36 = getitem_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_441: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(where_36, torch.float32);  where_36 = None
        sum_74: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_441, [0, 2, 3])
        convert_element_type_173: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_57, torch.float32);  convolution_57 = None
        sub_238: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_173, unsqueeze_810);  convert_element_type_173 = unsqueeze_810 = None
        mul_982: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_441, sub_238)
        sum_75: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_982, [0, 2, 3]);  mul_982 = None
        mul_983: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_74, 2.703287197231834e-05)
        unsqueeze_811: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
        unsqueeze_812: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
        unsqueeze_813: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
        mul_984: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, 2.703287197231834e-05)
        mul_985: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
        mul_986: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_984, mul_985);  mul_984 = mul_985 = None
        unsqueeze_814: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_986, 0);  mul_986 = None
        unsqueeze_815: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
        unsqueeze_816: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
        mul_987: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_172, primals_348);  primals_348 = None
        unsqueeze_817: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_987, 0);  mul_987 = None
        unsqueeze_818: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
        unsqueeze_819: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
        mul_988: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_816);  sub_238 = unsqueeze_816 = None
        sub_240: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_441, mul_988);  convert_element_type_441 = mul_988 = None
        sub_241: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_240, unsqueeze_813);  sub_240 = unsqueeze_813 = None
        mul_989: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_819);  sub_241 = unsqueeze_819 = None
        mul_990: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, squeeze_172);  sum_75 = squeeze_172 = None
        convert_element_type_443: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_989, torch.bfloat16);  mul_989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(convert_element_type_443, relu_56, convert_element_type_172, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_443 = convert_element_type_172 = None
        getitem_304: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = convolution_backward_36[0]
        getitem_305: "bf16[160, 160, 7, 1][1120, 1, 160, 160]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None
        convert_element_type_444: "f32[160, 160, 7, 1][1120, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_305, torch.float32);  getitem_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_37: "b8[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.le.Scalar(relu_56, 0);  relu_56 = None
        where_37: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.where.self(le_37, full_default, getitem_304);  le_37 = getitem_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_445: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(where_37, torch.float32);  where_37 = None
        sum_76: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_445, [0, 2, 3])
        convert_element_type_170: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_56, torch.float32);  convolution_56 = None
        sub_242: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_170, unsqueeze_822);  convert_element_type_170 = unsqueeze_822 = None
        mul_991: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_445, sub_242)
        sum_77: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_991, [0, 2, 3]);  mul_991 = None
        mul_992: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, 2.703287197231834e-05)
        unsqueeze_823: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_992, 0);  mul_992 = None
        unsqueeze_824: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
        unsqueeze_825: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
        mul_993: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, 2.703287197231834e-05)
        mul_994: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
        mul_995: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_993, mul_994);  mul_993 = mul_994 = None
        unsqueeze_826: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_995, 0);  mul_995 = None
        unsqueeze_827: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
        unsqueeze_828: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
        mul_996: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_169, primals_342);  primals_342 = None
        unsqueeze_829: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_996, 0);  mul_996 = None
        unsqueeze_830: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
        unsqueeze_831: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
        mul_997: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_828);  sub_242 = unsqueeze_828 = None
        sub_244: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_445, mul_997);  convert_element_type_445 = mul_997 = None
        sub_245: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_244, unsqueeze_825);  sub_244 = unsqueeze_825 = None
        mul_998: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_831);  sub_245 = unsqueeze_831 = None
        mul_999: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, squeeze_169);  sum_77 = squeeze_169 = None
        convert_element_type_447: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_998, torch.bfloat16);  mul_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(convert_element_type_447, relu_55, convert_element_type_169, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_447 = convert_element_type_169 = None
        getitem_307: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = convolution_backward_37[0]
        getitem_308: "bf16[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None
        convert_element_type_448: "f32[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_308, torch.float32);  getitem_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_38: "b8[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.le.Scalar(relu_55, 0);  relu_55 = None
        where_38: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.where.self(le_38, full_default, getitem_307);  le_38 = getitem_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_449: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(where_38, torch.float32);  where_38 = None
        sum_78: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_449, [0, 2, 3])
        convert_element_type_167: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32);  convolution_55 = None
        sub_246: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_167, unsqueeze_834);  convert_element_type_167 = unsqueeze_834 = None
        mul_1000: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_449, sub_246)
        sum_79: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1000, [0, 2, 3]);  mul_1000 = None
        mul_1001: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_78, 2.703287197231834e-05)
        unsqueeze_835: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1001, 0);  mul_1001 = None
        unsqueeze_836: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 2);  unsqueeze_835 = None
        unsqueeze_837: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 3);  unsqueeze_836 = None
        mul_1002: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, 2.703287197231834e-05)
        mul_1003: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
        mul_1004: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1002, mul_1003);  mul_1002 = mul_1003 = None
        unsqueeze_838: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1004, 0);  mul_1004 = None
        unsqueeze_839: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 2);  unsqueeze_838 = None
        unsqueeze_840: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 3);  unsqueeze_839 = None
        mul_1005: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_166, primals_336);  primals_336 = None
        unsqueeze_841: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1005, 0);  mul_1005 = None
        unsqueeze_842: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
        unsqueeze_843: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
        mul_1006: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_840);  sub_246 = unsqueeze_840 = None
        sub_248: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_449, mul_1006);  convert_element_type_449 = mul_1006 = None
        sub_249: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_248, unsqueeze_837);  sub_248 = unsqueeze_837 = None
        mul_1007: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_843);  sub_249 = unsqueeze_843 = None
        mul_1008: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, squeeze_166);  sum_79 = squeeze_166 = None
        convert_element_type_451: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1007, torch.bfloat16);  mul_1007 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(convert_element_type_451, relu_54, convert_element_type_166, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_451 = convert_element_type_166 = None
        getitem_310: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = convolution_backward_38[0]
        getitem_311: "bf16[160, 160, 7, 1][1120, 1, 160, 160]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None
        convert_element_type_452: "f32[160, 160, 7, 1][1120, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_311, torch.float32);  getitem_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_39: "b8[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.le.Scalar(relu_54, 0);  relu_54 = None
        where_39: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.where.self(le_39, full_default, getitem_310);  le_39 = getitem_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_453: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(where_39, torch.float32);  where_39 = None
        sum_80: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_453, [0, 2, 3])
        convert_element_type_164: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_54, torch.float32);  convolution_54 = None
        sub_250: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_164, unsqueeze_846);  convert_element_type_164 = unsqueeze_846 = None
        mul_1009: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_453, sub_250)
        sum_81: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1009, [0, 2, 3]);  mul_1009 = None
        mul_1010: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_80, 2.703287197231834e-05)
        unsqueeze_847: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1010, 0);  mul_1010 = None
        unsqueeze_848: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 2);  unsqueeze_847 = None
        unsqueeze_849: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 3);  unsqueeze_848 = None
        mul_1011: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, 2.703287197231834e-05)
        mul_1012: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
        mul_1013: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1011, mul_1012);  mul_1011 = mul_1012 = None
        unsqueeze_850: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1013, 0);  mul_1013 = None
        unsqueeze_851: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 2);  unsqueeze_850 = None
        unsqueeze_852: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 3);  unsqueeze_851 = None
        mul_1014: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_163, primals_330);  primals_330 = None
        unsqueeze_853: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1014, 0);  mul_1014 = None
        unsqueeze_854: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
        unsqueeze_855: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
        mul_1015: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_852);  sub_250 = unsqueeze_852 = None
        sub_252: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_453, mul_1015);  convert_element_type_453 = mul_1015 = None
        sub_253: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_252, unsqueeze_849);  sub_252 = unsqueeze_849 = None
        mul_1016: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_855);  sub_253 = unsqueeze_855 = None
        mul_1017: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, squeeze_163);  sum_81 = squeeze_163 = None
        convert_element_type_455: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1016, torch.bfloat16);  mul_1016 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(convert_element_type_455, cat_5, convert_element_type_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_455 = convert_element_type_163 = None
        getitem_313: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_39[0]
        getitem_314: "bf16[160, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None
        add_485: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.add.Tensor(avg_pool2d_backward_3, getitem_313);  avg_pool2d_backward_3 = getitem_313 = None
        convert_element_type_456: "f32[160, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_314, torch.float32);  getitem_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_53: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_53, getitem_113)
        mul_371: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
        unsqueeze_212: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_324, -1)
        unsqueeze_213: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_377: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_213);  mul_371 = unsqueeze_213 = None
        unsqueeze_214: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_325, -1);  primals_325 = None
        unsqueeze_215: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_269: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_215);  mul_377 = unsqueeze_215 = None
        convert_element_type_162: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_269, torch.bfloat16);  add_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_53: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_162);  convert_element_type_162 = None
        le_40: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_53, 0);  relu_53 = None
        where_40: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_40, full_default, slice_25);  le_40 = slice_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_457: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_40, torch.float32);  where_40 = None
        squeeze_159: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
        unsqueeze_856: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
        unsqueeze_857: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
        unsqueeze_858: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None
        sum_82: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_457, [0, 2, 3])
        convert_element_type_161: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_53, torch.float32);  convolution_53 = None
        sub_254: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_161, unsqueeze_858);  convert_element_type_161 = unsqueeze_858 = None
        mul_1018: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_457, sub_254)
        sum_83: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1018, [0, 2, 3]);  mul_1018 = None
        mul_1019: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, 2.703287197231834e-05)
        unsqueeze_859: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
        unsqueeze_860: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 2);  unsqueeze_859 = None
        unsqueeze_861: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 3);  unsqueeze_860 = None
        mul_1020: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, 2.703287197231834e-05)
        squeeze_160: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
        mul_1021: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
        mul_1022: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1020, mul_1021);  mul_1020 = mul_1021 = None
        unsqueeze_862: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1022, 0);  mul_1022 = None
        unsqueeze_863: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 2);  unsqueeze_862 = None
        unsqueeze_864: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 3);  unsqueeze_863 = None
        mul_1023: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_160, primals_324);  primals_324 = None
        unsqueeze_865: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1023, 0);  mul_1023 = None
        unsqueeze_866: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
        unsqueeze_867: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
        mul_1024: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_864);  sub_254 = unsqueeze_864 = None
        sub_256: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_457, mul_1024);  convert_element_type_457 = mul_1024 = None
        sub_257: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_256, unsqueeze_861);  sub_256 = unsqueeze_861 = None
        mul_1025: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_867);  sub_257 = unsqueeze_867 = None
        mul_1026: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, squeeze_160);  sum_83 = squeeze_160 = None
        convert_element_type_459: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1025, torch.bfloat16);  mul_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(convert_element_type_459, relu_52, convert_element_type_160, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_459 = convert_element_type_160 = None
        getitem_316: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = convolution_backward_40[0]
        getitem_317: "bf16[192, 160, 7, 1][1120, 1, 160, 160]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None
        convert_element_type_460: "f32[192, 160, 7, 1][1120, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_317, torch.float32);  getitem_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_41: "b8[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.le.Scalar(relu_52, 0);  relu_52 = None
        where_41: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.where.self(le_41, full_default, getitem_316);  le_41 = getitem_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_461: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(where_41, torch.float32);  where_41 = None
        sum_84: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_461, [0, 2, 3])
        convert_element_type_158: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_52, torch.float32);  convolution_52 = None
        sub_258: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_158, unsqueeze_870);  convert_element_type_158 = unsqueeze_870 = None
        mul_1027: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_461, sub_258)
        sum_85: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1027, [0, 2, 3]);  mul_1027 = None
        mul_1028: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_84, 2.703287197231834e-05)
        unsqueeze_871: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1028, 0);  mul_1028 = None
        unsqueeze_872: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 2);  unsqueeze_871 = None
        unsqueeze_873: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 3);  unsqueeze_872 = None
        mul_1029: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, 2.703287197231834e-05)
        mul_1030: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
        mul_1031: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1029, mul_1030);  mul_1029 = mul_1030 = None
        unsqueeze_874: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1031, 0);  mul_1031 = None
        unsqueeze_875: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 2);  unsqueeze_874 = None
        unsqueeze_876: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 3);  unsqueeze_875 = None
        mul_1032: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_157, primals_318);  primals_318 = None
        unsqueeze_877: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1032, 0);  mul_1032 = None
        unsqueeze_878: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
        unsqueeze_879: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
        mul_1033: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_876);  sub_258 = unsqueeze_876 = None
        sub_260: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_461, mul_1033);  convert_element_type_461 = mul_1033 = None
        sub_261: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_260, unsqueeze_873);  sub_260 = unsqueeze_873 = None
        mul_1034: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_879);  sub_261 = unsqueeze_879 = None
        mul_1035: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, squeeze_157);  sum_85 = squeeze_157 = None
        convert_element_type_463: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1034, torch.bfloat16);  mul_1034 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(convert_element_type_463, relu_51, convert_element_type_157, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_463 = convert_element_type_157 = None
        getitem_319: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = convolution_backward_41[0]
        getitem_320: "bf16[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None
        convert_element_type_464: "f32[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_320, torch.float32);  getitem_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_42: "b8[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.le.Scalar(relu_51, 0);  relu_51 = None
        where_42: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.where.self(le_42, full_default, getitem_319);  le_42 = getitem_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_465: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(where_42, torch.float32);  where_42 = None
        sum_86: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_465, [0, 2, 3])
        convert_element_type_155: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32);  convolution_51 = None
        sub_262: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_155, unsqueeze_882);  convert_element_type_155 = unsqueeze_882 = None
        mul_1036: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_465, sub_262)
        sum_87: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1036, [0, 2, 3]);  mul_1036 = None
        mul_1037: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_86, 2.703287197231834e-05)
        unsqueeze_883: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1037, 0);  mul_1037 = None
        unsqueeze_884: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 2);  unsqueeze_883 = None
        unsqueeze_885: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 3);  unsqueeze_884 = None
        mul_1038: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, 2.703287197231834e-05)
        mul_1039: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_1040: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1038, mul_1039);  mul_1038 = mul_1039 = None
        unsqueeze_886: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1040, 0);  mul_1040 = None
        unsqueeze_887: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 2);  unsqueeze_886 = None
        unsqueeze_888: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 3);  unsqueeze_887 = None
        mul_1041: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, primals_312);  primals_312 = None
        unsqueeze_889: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1041, 0);  mul_1041 = None
        unsqueeze_890: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
        unsqueeze_891: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
        mul_1042: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_888);  sub_262 = unsqueeze_888 = None
        sub_264: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_465, mul_1042);  convert_element_type_465 = mul_1042 = None
        sub_265: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_264, unsqueeze_885);  sub_264 = unsqueeze_885 = None
        mul_1043: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_891);  sub_265 = unsqueeze_891 = None
        mul_1044: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, squeeze_154);  sum_87 = squeeze_154 = None
        convert_element_type_467: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1043, torch.bfloat16);  mul_1043 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(convert_element_type_467, cat_5, convert_element_type_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_467 = convert_element_type_154 = None
        getitem_322: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_42[0]
        getitem_323: "bf16[160, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None
        add_486: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.add.Tensor(add_485, getitem_322);  add_485 = getitem_322 = None
        convert_element_type_468: "f32[160, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_323, torch.float32);  getitem_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_50: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_50, getitem_107)
        mul_350: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        unsqueeze_200: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_306, -1)
        unsqueeze_201: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        mul_356: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
        unsqueeze_202: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_307, -1);  primals_307 = None
        unsqueeze_203: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        add_254: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
        convert_element_type_153: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_254, torch.bfloat16);  add_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_50: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_153);  convert_element_type_153 = None
        le_43: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
        where_43: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_43, full_default, slice_24);  le_43 = slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_469: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_43, torch.float32);  where_43 = None
        squeeze_150: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        unsqueeze_892: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
        unsqueeze_893: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 2);  unsqueeze_892 = None
        unsqueeze_894: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 3);  unsqueeze_893 = None
        sum_88: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_469, [0, 2, 3])
        convert_element_type_152: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32);  convolution_50 = None
        sub_266: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_152, unsqueeze_894);  convert_element_type_152 = unsqueeze_894 = None
        mul_1045: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_469, sub_266)
        sum_89: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1045, [0, 2, 3]);  mul_1045 = None
        mul_1046: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_88, 2.703287197231834e-05)
        unsqueeze_895: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1046, 0);  mul_1046 = None
        unsqueeze_896: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 2);  unsqueeze_895 = None
        unsqueeze_897: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 3);  unsqueeze_896 = None
        mul_1047: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, 2.703287197231834e-05)
        squeeze_151: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_1048: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_1049: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1047, mul_1048);  mul_1047 = mul_1048 = None
        unsqueeze_898: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1049, 0);  mul_1049 = None
        unsqueeze_899: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 2);  unsqueeze_898 = None
        unsqueeze_900: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 3);  unsqueeze_899 = None
        mul_1050: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, primals_306);  primals_306 = None
        unsqueeze_901: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1050, 0);  mul_1050 = None
        unsqueeze_902: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
        unsqueeze_903: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
        mul_1051: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_900);  sub_266 = unsqueeze_900 = None
        sub_268: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_469, mul_1051);  convert_element_type_469 = mul_1051 = None
        sub_269: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_268, unsqueeze_897);  sub_268 = unsqueeze_897 = None
        mul_1052: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_903);  sub_269 = unsqueeze_903 = None
        mul_1053: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, squeeze_151);  sum_89 = squeeze_151 = None
        convert_element_type_471: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1052, torch.bfloat16);  mul_1052 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(convert_element_type_471, cat_5, convert_element_type_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_471 = cat_5 = convert_element_type_151 = None
        getitem_325: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_43[0]
        getitem_326: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None
        add_487: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.add.Tensor(add_486, getitem_325);  add_486 = getitem_325 = None
        convert_element_type_472: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_326, torch.float32);  getitem_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:152 in forward, code: return torch.cat(outputs, 1)
        slice_28: "bf16[128, 192, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_487, 1, 0, 192)
        slice_29: "bf16[128, 192, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_487, 1, 192, 384)
        slice_30: "bf16[128, 192, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_487, 1, 384, 576)
        slice_31: "bf16[128, 192, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_487, 1, 576, 768);  add_487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_49: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_49, getitem_105)
        mul_343: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        unsqueeze_196: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_300, -1)
        unsqueeze_197: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_349: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
        unsqueeze_198: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_301, -1);  primals_301 = None
        unsqueeze_199: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_249: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
        convert_element_type_150: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_249, torch.bfloat16);  add_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_49: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_150);  convert_element_type_150 = None
        le_44: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
        where_44: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_44, full_default, slice_31);  le_44 = slice_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_473: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_44, torch.float32);  where_44 = None
        squeeze_147: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        unsqueeze_904: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
        unsqueeze_905: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 2);  unsqueeze_904 = None
        unsqueeze_906: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 3);  unsqueeze_905 = None
        sum_90: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_473, [0, 2, 3])
        convert_element_type_149: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32);  convolution_49 = None
        sub_270: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_149, unsqueeze_906);  convert_element_type_149 = unsqueeze_906 = None
        mul_1054: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_473, sub_270)
        sum_91: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1054, [0, 2, 3]);  mul_1054 = None
        mul_1055: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_90, 2.703287197231834e-05)
        unsqueeze_907: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1055, 0);  mul_1055 = None
        unsqueeze_908: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 2);  unsqueeze_907 = None
        unsqueeze_909: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 3);  unsqueeze_908 = None
        mul_1056: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, 2.703287197231834e-05)
        squeeze_148: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_1057: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_1058: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1056, mul_1057);  mul_1056 = mul_1057 = None
        unsqueeze_910: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1058, 0);  mul_1058 = None
        unsqueeze_911: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 2);  unsqueeze_910 = None
        unsqueeze_912: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 3);  unsqueeze_911 = None
        mul_1059: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, primals_300);  primals_300 = None
        unsqueeze_913: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1059, 0);  mul_1059 = None
        unsqueeze_914: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
        unsqueeze_915: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
        mul_1060: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_912);  sub_270 = unsqueeze_912 = None
        sub_272: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_473, mul_1060);  convert_element_type_473 = mul_1060 = None
        sub_273: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_272, unsqueeze_909);  sub_272 = unsqueeze_909 = None
        mul_1061: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_915);  sub_273 = unsqueeze_915 = None
        mul_1062: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, squeeze_148);  sum_91 = squeeze_148 = None
        convert_element_type_475: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1061, torch.bfloat16);  mul_1061 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(convert_element_type_475, avg_pool2d_4, convert_element_type_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_475 = avg_pool2d_4 = convert_element_type_148 = None
        getitem_328: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_44[0]
        getitem_329: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_44[1];  convolution_backward_44 = None
        convert_element_type_476: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_329, torch.float32);  getitem_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:144 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_4: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(getitem_328, cat_4, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_48: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_48, getitem_103)
        mul_336: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        unsqueeze_192: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_294, -1)
        unsqueeze_193: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        mul_342: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
        unsqueeze_194: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_295, -1);  primals_295 = None
        unsqueeze_195: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        add_244: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
        convert_element_type_147: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_244, torch.bfloat16);  add_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_48: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_147);  convert_element_type_147 = None
        le_45: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
        where_45: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_45, full_default, slice_30);  le_45 = slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_477: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_45, torch.float32);  where_45 = None
        squeeze_144: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        unsqueeze_916: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
        unsqueeze_917: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 2);  unsqueeze_916 = None
        unsqueeze_918: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 3);  unsqueeze_917 = None
        sum_92: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_477, [0, 2, 3])
        convert_element_type_146: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_48, torch.float32);  convolution_48 = None
        sub_274: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_146, unsqueeze_918);  convert_element_type_146 = unsqueeze_918 = None
        mul_1063: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_477, sub_274)
        sum_93: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1063, [0, 2, 3]);  mul_1063 = None
        mul_1064: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_92, 2.703287197231834e-05)
        unsqueeze_919: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1064, 0);  mul_1064 = None
        unsqueeze_920: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 2);  unsqueeze_919 = None
        unsqueeze_921: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 3);  unsqueeze_920 = None
        mul_1065: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, 2.703287197231834e-05)
        squeeze_145: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_1066: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_1067: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1065, mul_1066);  mul_1065 = mul_1066 = None
        unsqueeze_922: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1067, 0);  mul_1067 = None
        unsqueeze_923: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 2);  unsqueeze_922 = None
        unsqueeze_924: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 3);  unsqueeze_923 = None
        mul_1068: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, primals_294);  primals_294 = None
        unsqueeze_925: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1068, 0);  mul_1068 = None
        unsqueeze_926: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
        unsqueeze_927: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
        mul_1069: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_924);  sub_274 = unsqueeze_924 = None
        sub_276: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_477, mul_1069);  convert_element_type_477 = mul_1069 = None
        sub_277: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_276, unsqueeze_921);  sub_276 = unsqueeze_921 = None
        mul_1070: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_927);  sub_277 = unsqueeze_927 = None
        mul_1071: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, squeeze_145);  sum_93 = squeeze_145 = None
        convert_element_type_479: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1070, torch.bfloat16);  mul_1070 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(convert_element_type_479, relu_47, convert_element_type_145, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_479 = convert_element_type_145 = None
        getitem_331: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = convolution_backward_45[0]
        getitem_332: "bf16[192, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = convolution_backward_45[1];  convolution_backward_45 = None
        convert_element_type_480: "f32[192, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_332, torch.float32);  getitem_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_46: "b8[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
        where_46: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.where.self(le_46, full_default, getitem_331);  le_46 = getitem_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_481: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(where_46, torch.float32);  where_46 = None
        sum_94: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_481, [0, 2, 3])
        convert_element_type_143: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32);  convolution_47 = None
        sub_278: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_143, unsqueeze_930);  convert_element_type_143 = unsqueeze_930 = None
        mul_1072: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_481, sub_278)
        sum_95: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1072, [0, 2, 3]);  mul_1072 = None
        mul_1073: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_94, 2.703287197231834e-05)
        unsqueeze_931: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1073, 0);  mul_1073 = None
        unsqueeze_932: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 2);  unsqueeze_931 = None
        unsqueeze_933: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 3);  unsqueeze_932 = None
        mul_1074: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, 2.703287197231834e-05)
        mul_1075: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_1076: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1074, mul_1075);  mul_1074 = mul_1075 = None
        unsqueeze_934: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1076, 0);  mul_1076 = None
        unsqueeze_935: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 2);  unsqueeze_934 = None
        unsqueeze_936: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 3);  unsqueeze_935 = None
        mul_1077: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, primals_288);  primals_288 = None
        unsqueeze_937: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1077, 0);  mul_1077 = None
        unsqueeze_938: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 2);  unsqueeze_937 = None
        unsqueeze_939: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 3);  unsqueeze_938 = None
        mul_1078: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_936);  sub_278 = unsqueeze_936 = None
        sub_280: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_481, mul_1078);  convert_element_type_481 = mul_1078 = None
        sub_281: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_280, unsqueeze_933);  sub_280 = unsqueeze_933 = None
        mul_1079: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_939);  sub_281 = unsqueeze_939 = None
        mul_1080: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, squeeze_142);  sum_95 = squeeze_142 = None
        convert_element_type_483: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1079, torch.bfloat16);  mul_1079 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(convert_element_type_483, relu_46, convert_element_type_142, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_483 = convert_element_type_142 = None
        getitem_334: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = convolution_backward_46[0]
        getitem_335: "bf16[160, 160, 7, 1][1120, 1, 160, 160]cuda:0" = convolution_backward_46[1];  convolution_backward_46 = None
        convert_element_type_484: "f32[160, 160, 7, 1][1120, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_335, torch.float32);  getitem_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_47: "b8[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
        where_47: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.where.self(le_47, full_default, getitem_334);  le_47 = getitem_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_485: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(where_47, torch.float32);  where_47 = None
        sum_96: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_485, [0, 2, 3])
        convert_element_type_140: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_46, torch.float32);  convolution_46 = None
        sub_282: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_140, unsqueeze_942);  convert_element_type_140 = unsqueeze_942 = None
        mul_1081: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_485, sub_282)
        sum_97: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1081, [0, 2, 3]);  mul_1081 = None
        mul_1082: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_96, 2.703287197231834e-05)
        unsqueeze_943: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
        unsqueeze_944: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 2);  unsqueeze_943 = None
        unsqueeze_945: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 3);  unsqueeze_944 = None
        mul_1083: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, 2.703287197231834e-05)
        mul_1084: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_1085: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1083, mul_1084);  mul_1083 = mul_1084 = None
        unsqueeze_946: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1085, 0);  mul_1085 = None
        unsqueeze_947: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 2);  unsqueeze_946 = None
        unsqueeze_948: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 3);  unsqueeze_947 = None
        mul_1086: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, primals_282);  primals_282 = None
        unsqueeze_949: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1086, 0);  mul_1086 = None
        unsqueeze_950: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 2);  unsqueeze_949 = None
        unsqueeze_951: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 3);  unsqueeze_950 = None
        mul_1087: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_948);  sub_282 = unsqueeze_948 = None
        sub_284: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_485, mul_1087);  convert_element_type_485 = mul_1087 = None
        sub_285: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_284, unsqueeze_945);  sub_284 = unsqueeze_945 = None
        mul_1088: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_951);  sub_285 = unsqueeze_951 = None
        mul_1089: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, squeeze_139);  sum_97 = squeeze_139 = None
        convert_element_type_487: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1088, torch.bfloat16);  mul_1088 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(convert_element_type_487, relu_45, convert_element_type_139, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_487 = convert_element_type_139 = None
        getitem_337: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = convolution_backward_47[0]
        getitem_338: "bf16[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = convolution_backward_47[1];  convolution_backward_47 = None
        convert_element_type_488: "f32[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_338, torch.float32);  getitem_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_48: "b8[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
        where_48: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.where.self(le_48, full_default, getitem_337);  le_48 = getitem_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_489: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(where_48, torch.float32);  where_48 = None
        sum_98: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_489, [0, 2, 3])
        convert_element_type_137: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32);  convolution_45 = None
        sub_286: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_137, unsqueeze_954);  convert_element_type_137 = unsqueeze_954 = None
        mul_1090: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_489, sub_286)
        sum_99: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1090, [0, 2, 3]);  mul_1090 = None
        mul_1091: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_98, 2.703287197231834e-05)
        unsqueeze_955: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1091, 0);  mul_1091 = None
        unsqueeze_956: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 2);  unsqueeze_955 = None
        unsqueeze_957: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 3);  unsqueeze_956 = None
        mul_1092: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, 2.703287197231834e-05)
        mul_1093: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_1094: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1092, mul_1093);  mul_1092 = mul_1093 = None
        unsqueeze_958: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1094, 0);  mul_1094 = None
        unsqueeze_959: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 2);  unsqueeze_958 = None
        unsqueeze_960: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 3);  unsqueeze_959 = None
        mul_1095: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, primals_276);  primals_276 = None
        unsqueeze_961: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1095, 0);  mul_1095 = None
        unsqueeze_962: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 2);  unsqueeze_961 = None
        unsqueeze_963: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 3);  unsqueeze_962 = None
        mul_1096: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_960);  sub_286 = unsqueeze_960 = None
        sub_288: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_489, mul_1096);  convert_element_type_489 = mul_1096 = None
        sub_289: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_288, unsqueeze_957);  sub_288 = unsqueeze_957 = None
        mul_1097: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_963);  sub_289 = unsqueeze_963 = None
        mul_1098: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, squeeze_136);  sum_99 = squeeze_136 = None
        convert_element_type_491: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1097, torch.bfloat16);  mul_1097 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(convert_element_type_491, relu_44, convert_element_type_136, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_491 = convert_element_type_136 = None
        getitem_340: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = convolution_backward_48[0]
        getitem_341: "bf16[160, 160, 7, 1][1120, 1, 160, 160]cuda:0" = convolution_backward_48[1];  convolution_backward_48 = None
        convert_element_type_492: "f32[160, 160, 7, 1][1120, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_341, torch.float32);  getitem_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_49: "b8[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
        where_49: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.where.self(le_49, full_default, getitem_340);  le_49 = getitem_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_493: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(where_49, torch.float32);  where_49 = None
        sum_100: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_493, [0, 2, 3])
        convert_element_type_134: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32);  convolution_44 = None
        sub_290: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_134, unsqueeze_966);  convert_element_type_134 = unsqueeze_966 = None
        mul_1099: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_493, sub_290)
        sum_101: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1099, [0, 2, 3]);  mul_1099 = None
        mul_1100: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_100, 2.703287197231834e-05)
        unsqueeze_967: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1100, 0);  mul_1100 = None
        unsqueeze_968: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 2);  unsqueeze_967 = None
        unsqueeze_969: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 3);  unsqueeze_968 = None
        mul_1101: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, 2.703287197231834e-05)
        mul_1102: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_1103: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1101, mul_1102);  mul_1101 = mul_1102 = None
        unsqueeze_970: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1103, 0);  mul_1103 = None
        unsqueeze_971: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 2);  unsqueeze_970 = None
        unsqueeze_972: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 3);  unsqueeze_971 = None
        mul_1104: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, primals_270);  primals_270 = None
        unsqueeze_973: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1104, 0);  mul_1104 = None
        unsqueeze_974: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 2);  unsqueeze_973 = None
        unsqueeze_975: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 3);  unsqueeze_974 = None
        mul_1105: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_972);  sub_290 = unsqueeze_972 = None
        sub_292: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_493, mul_1105);  convert_element_type_493 = mul_1105 = None
        sub_293: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_292, unsqueeze_969);  sub_292 = unsqueeze_969 = None
        mul_1106: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_975);  sub_293 = unsqueeze_975 = None
        mul_1107: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, squeeze_133);  sum_101 = squeeze_133 = None
        convert_element_type_495: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1106, torch.bfloat16);  mul_1106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(convert_element_type_495, cat_4, convert_element_type_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_495 = convert_element_type_133 = None
        getitem_343: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_49[0]
        getitem_344: "bf16[160, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_49[1];  convolution_backward_49 = None
        add_488: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.add.Tensor(avg_pool2d_backward_4, getitem_343);  avg_pool2d_backward_4 = getitem_343 = None
        convert_element_type_496: "f32[160, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_344, torch.float32);  getitem_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_43: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_43, getitem_93)
        mul_301: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        unsqueeze_172: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_264, -1)
        unsqueeze_173: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_307: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
        unsqueeze_174: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_265, -1);  primals_265 = None
        unsqueeze_175: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_219: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
        convert_element_type_132: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_219, torch.bfloat16);  add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_43: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_132);  convert_element_type_132 = None
        le_50: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
        where_50: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_50, full_default, slice_29);  le_50 = slice_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_497: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_50, torch.float32);  where_50 = None
        squeeze_129: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        unsqueeze_976: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
        unsqueeze_977: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 2);  unsqueeze_976 = None
        unsqueeze_978: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 3);  unsqueeze_977 = None
        sum_102: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_497, [0, 2, 3])
        convert_element_type_131: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32);  convolution_43 = None
        sub_294: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_131, unsqueeze_978);  convert_element_type_131 = unsqueeze_978 = None
        mul_1108: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_497, sub_294)
        sum_103: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1108, [0, 2, 3]);  mul_1108 = None
        mul_1109: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_102, 2.703287197231834e-05)
        unsqueeze_979: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1109, 0);  mul_1109 = None
        unsqueeze_980: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 2);  unsqueeze_979 = None
        unsqueeze_981: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 3);  unsqueeze_980 = None
        mul_1110: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, 2.703287197231834e-05)
        squeeze_130: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_1111: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_1112: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1110, mul_1111);  mul_1110 = mul_1111 = None
        unsqueeze_982: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1112, 0);  mul_1112 = None
        unsqueeze_983: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 2);  unsqueeze_982 = None
        unsqueeze_984: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 3);  unsqueeze_983 = None
        mul_1113: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, primals_264);  primals_264 = None
        unsqueeze_985: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1113, 0);  mul_1113 = None
        unsqueeze_986: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 2);  unsqueeze_985 = None
        unsqueeze_987: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 3);  unsqueeze_986 = None
        mul_1114: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_984);  sub_294 = unsqueeze_984 = None
        sub_296: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_497, mul_1114);  convert_element_type_497 = mul_1114 = None
        sub_297: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_296, unsqueeze_981);  sub_296 = unsqueeze_981 = None
        mul_1115: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_987);  sub_297 = unsqueeze_987 = None
        mul_1116: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, squeeze_130);  sum_103 = squeeze_130 = None
        convert_element_type_499: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1115, torch.bfloat16);  mul_1115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(convert_element_type_499, relu_42, convert_element_type_130, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_499 = convert_element_type_130 = None
        getitem_346: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = convolution_backward_50[0]
        getitem_347: "bf16[192, 160, 7, 1][1120, 1, 160, 160]cuda:0" = convolution_backward_50[1];  convolution_backward_50 = None
        convert_element_type_500: "f32[192, 160, 7, 1][1120, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_347, torch.float32);  getitem_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_51: "b8[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
        where_51: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.where.self(le_51, full_default, getitem_346);  le_51 = getitem_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_501: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(where_51, torch.float32);  where_51 = None
        sum_104: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_501, [0, 2, 3])
        convert_element_type_128: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32);  convolution_42 = None
        sub_298: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_128, unsqueeze_990);  convert_element_type_128 = unsqueeze_990 = None
        mul_1117: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_501, sub_298)
        sum_105: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1117, [0, 2, 3]);  mul_1117 = None
        mul_1118: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_104, 2.703287197231834e-05)
        unsqueeze_991: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1118, 0);  mul_1118 = None
        unsqueeze_992: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 2);  unsqueeze_991 = None
        unsqueeze_993: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 3);  unsqueeze_992 = None
        mul_1119: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, 2.703287197231834e-05)
        mul_1120: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_1121: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1119, mul_1120);  mul_1119 = mul_1120 = None
        unsqueeze_994: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1121, 0);  mul_1121 = None
        unsqueeze_995: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 2);  unsqueeze_994 = None
        unsqueeze_996: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 3);  unsqueeze_995 = None
        mul_1122: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, primals_258);  primals_258 = None
        unsqueeze_997: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1122, 0);  mul_1122 = None
        unsqueeze_998: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 2);  unsqueeze_997 = None
        unsqueeze_999: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 3);  unsqueeze_998 = None
        mul_1123: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_996);  sub_298 = unsqueeze_996 = None
        sub_300: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_501, mul_1123);  convert_element_type_501 = mul_1123 = None
        sub_301: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_300, unsqueeze_993);  sub_300 = unsqueeze_993 = None
        mul_1124: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_999);  sub_301 = unsqueeze_999 = None
        mul_1125: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, squeeze_127);  sum_105 = squeeze_127 = None
        convert_element_type_503: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1124, torch.bfloat16);  mul_1124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(convert_element_type_503, relu_41, convert_element_type_127, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_503 = convert_element_type_127 = None
        getitem_349: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = convolution_backward_51[0]
        getitem_350: "bf16[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = convolution_backward_51[1];  convolution_backward_51 = None
        convert_element_type_504: "f32[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_350, torch.float32);  getitem_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_52: "b8[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
        where_52: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.where.self(le_52, full_default, getitem_349);  le_52 = getitem_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_505: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(where_52, torch.float32);  where_52 = None
        sum_106: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_505, [0, 2, 3])
        convert_element_type_125: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32);  convolution_41 = None
        sub_302: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_125, unsqueeze_1002);  convert_element_type_125 = unsqueeze_1002 = None
        mul_1126: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_505, sub_302)
        sum_107: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1126, [0, 2, 3]);  mul_1126 = None
        mul_1127: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_106, 2.703287197231834e-05)
        unsqueeze_1003: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1127, 0);  mul_1127 = None
        unsqueeze_1004: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 2);  unsqueeze_1003 = None
        unsqueeze_1005: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 3);  unsqueeze_1004 = None
        mul_1128: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, 2.703287197231834e-05)
        mul_1129: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_1130: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1128, mul_1129);  mul_1128 = mul_1129 = None
        unsqueeze_1006: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1130, 0);  mul_1130 = None
        unsqueeze_1007: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 2);  unsqueeze_1006 = None
        unsqueeze_1008: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 3);  unsqueeze_1007 = None
        mul_1131: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, primals_252);  primals_252 = None
        unsqueeze_1009: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1131, 0);  mul_1131 = None
        unsqueeze_1010: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 2);  unsqueeze_1009 = None
        unsqueeze_1011: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 3);  unsqueeze_1010 = None
        mul_1132: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_1008);  sub_302 = unsqueeze_1008 = None
        sub_304: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_505, mul_1132);  convert_element_type_505 = mul_1132 = None
        sub_305: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_304, unsqueeze_1005);  sub_304 = unsqueeze_1005 = None
        mul_1133: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_1011);  sub_305 = unsqueeze_1011 = None
        mul_1134: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, squeeze_124);  sum_107 = squeeze_124 = None
        convert_element_type_507: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1133, torch.bfloat16);  mul_1133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(convert_element_type_507, cat_4, convert_element_type_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_507 = convert_element_type_124 = None
        getitem_352: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_52[0]
        getitem_353: "bf16[160, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_52[1];  convolution_backward_52 = None
        add_489: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.add.Tensor(add_488, getitem_352);  add_488 = getitem_352 = None
        convert_element_type_508: "f32[160, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_353, torch.float32);  getitem_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_40: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_40, getitem_87)
        mul_280: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        unsqueeze_160: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_246, -1)
        unsqueeze_161: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_286: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
        unsqueeze_162: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_247, -1);  primals_247 = None
        unsqueeze_163: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_204: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
        convert_element_type_123: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_204, torch.bfloat16);  add_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_40: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_123);  convert_element_type_123 = None
        le_53: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
        where_53: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_53, full_default, slice_28);  le_53 = slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_509: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_53, torch.float32);  where_53 = None
        squeeze_120: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        unsqueeze_1012: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
        unsqueeze_1013: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 2);  unsqueeze_1012 = None
        unsqueeze_1014: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 3);  unsqueeze_1013 = None
        sum_108: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_509, [0, 2, 3])
        convert_element_type_122: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32);  convolution_40 = None
        sub_306: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_122, unsqueeze_1014);  convert_element_type_122 = unsqueeze_1014 = None
        mul_1135: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_509, sub_306)
        sum_109: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1135, [0, 2, 3]);  mul_1135 = None
        mul_1136: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_108, 2.703287197231834e-05)
        unsqueeze_1015: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1136, 0);  mul_1136 = None
        unsqueeze_1016: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 2);  unsqueeze_1015 = None
        unsqueeze_1017: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 3);  unsqueeze_1016 = None
        mul_1137: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, 2.703287197231834e-05)
        squeeze_121: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_1138: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_1139: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1137, mul_1138);  mul_1137 = mul_1138 = None
        unsqueeze_1018: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1139, 0);  mul_1139 = None
        unsqueeze_1019: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 2);  unsqueeze_1018 = None
        unsqueeze_1020: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 3);  unsqueeze_1019 = None
        mul_1140: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, primals_246);  primals_246 = None
        unsqueeze_1021: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1140, 0);  mul_1140 = None
        unsqueeze_1022: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 2);  unsqueeze_1021 = None
        unsqueeze_1023: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 3);  unsqueeze_1022 = None
        mul_1141: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_1020);  sub_306 = unsqueeze_1020 = None
        sub_308: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_509, mul_1141);  convert_element_type_509 = mul_1141 = None
        sub_309: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_308, unsqueeze_1017);  sub_308 = unsqueeze_1017 = None
        mul_1142: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_1023);  sub_309 = unsqueeze_1023 = None
        mul_1143: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, squeeze_121);  sum_109 = squeeze_121 = None
        convert_element_type_511: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1142, torch.bfloat16);  mul_1142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(convert_element_type_511, cat_4, convert_element_type_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_511 = cat_4 = convert_element_type_121 = None
        getitem_355: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_53[0]
        getitem_356: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_53[1];  convolution_backward_53 = None
        add_490: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.add.Tensor(add_489, getitem_355);  add_489 = getitem_355 = None
        convert_element_type_512: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_356, torch.float32);  getitem_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:152 in forward, code: return torch.cat(outputs, 1)
        slice_32: "bf16[128, 192, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_490, 1, 0, 192)
        slice_33: "bf16[128, 192, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_490, 1, 192, 384)
        slice_34: "bf16[128, 192, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_490, 1, 384, 576)
        slice_35: "bf16[128, 192, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_490, 1, 576, 768);  add_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_39: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_39, getitem_85)
        mul_273: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        unsqueeze_156: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_240, -1)
        unsqueeze_157: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_279: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
        unsqueeze_158: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_241, -1);  primals_241 = None
        unsqueeze_159: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_199: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
        convert_element_type_120: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_199, torch.bfloat16);  add_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_39: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_120);  convert_element_type_120 = None
        le_54: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
        where_54: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_54, full_default, slice_35);  le_54 = slice_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_513: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_54, torch.float32);  where_54 = None
        squeeze_117: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        unsqueeze_1024: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
        unsqueeze_1025: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 2);  unsqueeze_1024 = None
        unsqueeze_1026: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 3);  unsqueeze_1025 = None
        sum_110: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_513, [0, 2, 3])
        convert_element_type_119: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32);  convolution_39 = None
        sub_310: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_119, unsqueeze_1026);  convert_element_type_119 = unsqueeze_1026 = None
        mul_1144: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_513, sub_310)
        sum_111: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1144, [0, 2, 3]);  mul_1144 = None
        mul_1145: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_110, 2.703287197231834e-05)
        unsqueeze_1027: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1145, 0);  mul_1145 = None
        unsqueeze_1028: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 2);  unsqueeze_1027 = None
        unsqueeze_1029: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 3);  unsqueeze_1028 = None
        mul_1146: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, 2.703287197231834e-05)
        squeeze_118: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_1147: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_1148: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1146, mul_1147);  mul_1146 = mul_1147 = None
        unsqueeze_1030: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1148, 0);  mul_1148 = None
        unsqueeze_1031: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, 2);  unsqueeze_1030 = None
        unsqueeze_1032: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 3);  unsqueeze_1031 = None
        mul_1149: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, primals_240);  primals_240 = None
        unsqueeze_1033: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1149, 0);  mul_1149 = None
        unsqueeze_1034: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1033, 2);  unsqueeze_1033 = None
        unsqueeze_1035: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 3);  unsqueeze_1034 = None
        mul_1150: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_1032);  sub_310 = unsqueeze_1032 = None
        sub_312: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_513, mul_1150);  convert_element_type_513 = mul_1150 = None
        sub_313: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_312, unsqueeze_1029);  sub_312 = unsqueeze_1029 = None
        mul_1151: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_1035);  sub_313 = unsqueeze_1035 = None
        mul_1152: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, squeeze_118);  sum_111 = squeeze_118 = None
        convert_element_type_515: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1151, torch.bfloat16);  mul_1151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(convert_element_type_515, avg_pool2d_3, convert_element_type_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_515 = avg_pool2d_3 = convert_element_type_118 = None
        getitem_358: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_54[0]
        getitem_359: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_54[1];  convolution_backward_54 = None
        convert_element_type_516: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_359, torch.float32);  getitem_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:144 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_5: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(getitem_358, cat_3, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_38: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_38, getitem_83)
        mul_266: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        unsqueeze_152: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_234, -1)
        unsqueeze_153: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        mul_272: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
        unsqueeze_154: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_235, -1);  primals_235 = None
        unsqueeze_155: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        add_194: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
        convert_element_type_117: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_194, torch.bfloat16);  add_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_38: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_117);  convert_element_type_117 = None
        le_55: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
        where_55: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_55, full_default, slice_34);  le_55 = slice_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_517: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_55, torch.float32);  where_55 = None
        squeeze_114: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        unsqueeze_1036: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
        unsqueeze_1037: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 2);  unsqueeze_1036 = None
        unsqueeze_1038: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 3);  unsqueeze_1037 = None
        sum_112: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_517, [0, 2, 3])
        convert_element_type_116: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_38, torch.float32);  convolution_38 = None
        sub_314: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_116, unsqueeze_1038);  convert_element_type_116 = unsqueeze_1038 = None
        mul_1153: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_517, sub_314)
        sum_113: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1153, [0, 2, 3]);  mul_1153 = None
        mul_1154: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_112, 2.703287197231834e-05)
        unsqueeze_1039: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1154, 0);  mul_1154 = None
        unsqueeze_1040: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 2);  unsqueeze_1039 = None
        unsqueeze_1041: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, 3);  unsqueeze_1040 = None
        mul_1155: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, 2.703287197231834e-05)
        squeeze_115: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_1156: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_1157: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1155, mul_1156);  mul_1155 = mul_1156 = None
        unsqueeze_1042: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1157, 0);  mul_1157 = None
        unsqueeze_1043: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, 2);  unsqueeze_1042 = None
        unsqueeze_1044: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1043, 3);  unsqueeze_1043 = None
        mul_1158: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, primals_234);  primals_234 = None
        unsqueeze_1045: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1158, 0);  mul_1158 = None
        unsqueeze_1046: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 2);  unsqueeze_1045 = None
        unsqueeze_1047: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 3);  unsqueeze_1046 = None
        mul_1159: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_1044);  sub_314 = unsqueeze_1044 = None
        sub_316: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_517, mul_1159);  convert_element_type_517 = mul_1159 = None
        sub_317: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_316, unsqueeze_1041);  sub_316 = unsqueeze_1041 = None
        mul_1160: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_1047);  sub_317 = unsqueeze_1047 = None
        mul_1161: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, squeeze_115);  sum_113 = squeeze_115 = None
        convert_element_type_519: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1160, torch.bfloat16);  mul_1160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(convert_element_type_519, relu_37, convert_element_type_115, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_519 = convert_element_type_115 = None
        getitem_361: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = convolution_backward_55[0]
        getitem_362: "bf16[192, 128, 1, 7][896, 1, 896, 128]cuda:0" = convolution_backward_55[1];  convolution_backward_55 = None
        convert_element_type_520: "f32[192, 128, 1, 7][896, 1, 896, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_362, torch.float32);  getitem_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_56: "b8[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
        where_56: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.where.self(le_56, full_default, getitem_361);  le_56 = getitem_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_521: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_56, torch.float32);  where_56 = None
        sum_114: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_521, [0, 2, 3])
        convert_element_type_113: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32);  convolution_37 = None
        sub_318: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_113, unsqueeze_1050);  convert_element_type_113 = unsqueeze_1050 = None
        mul_1162: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_521, sub_318)
        sum_115: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1162, [0, 2, 3]);  mul_1162 = None
        mul_1163: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_114, 2.703287197231834e-05)
        unsqueeze_1051: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1163, 0);  mul_1163 = None
        unsqueeze_1052: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 2);  unsqueeze_1051 = None
        unsqueeze_1053: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, 3);  unsqueeze_1052 = None
        mul_1164: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, 2.703287197231834e-05)
        mul_1165: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_1166: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1164, mul_1165);  mul_1164 = mul_1165 = None
        unsqueeze_1054: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1166, 0);  mul_1166 = None
        unsqueeze_1055: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, 2);  unsqueeze_1054 = None
        unsqueeze_1056: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1055, 3);  unsqueeze_1055 = None
        mul_1167: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, primals_228);  primals_228 = None
        unsqueeze_1057: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1167, 0);  mul_1167 = None
        unsqueeze_1058: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 2);  unsqueeze_1057 = None
        unsqueeze_1059: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 3);  unsqueeze_1058 = None
        mul_1168: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_1056);  sub_318 = unsqueeze_1056 = None
        sub_320: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_521, mul_1168);  convert_element_type_521 = mul_1168 = None
        sub_321: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_320, unsqueeze_1053);  sub_320 = unsqueeze_1053 = None
        mul_1169: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_1059);  sub_321 = unsqueeze_1059 = None
        mul_1170: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, squeeze_112);  sum_115 = squeeze_112 = None
        convert_element_type_523: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1169, torch.bfloat16);  mul_1169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(convert_element_type_523, relu_36, convert_element_type_112, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_523 = convert_element_type_112 = None
        getitem_364: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = convolution_backward_56[0]
        getitem_365: "bf16[128, 128, 7, 1][896, 1, 128, 128]cuda:0" = convolution_backward_56[1];  convolution_backward_56 = None
        convert_element_type_524: "f32[128, 128, 7, 1][896, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_365, torch.float32);  getitem_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_57: "b8[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
        where_57: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.where.self(le_57, full_default, getitem_364);  le_57 = getitem_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_525: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_57, torch.float32);  where_57 = None
        sum_116: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_525, [0, 2, 3])
        convert_element_type_110: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32);  convolution_36 = None
        sub_322: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_110, unsqueeze_1062);  convert_element_type_110 = unsqueeze_1062 = None
        mul_1171: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_525, sub_322)
        sum_117: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1171, [0, 2, 3]);  mul_1171 = None
        mul_1172: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_116, 2.703287197231834e-05)
        unsqueeze_1063: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1172, 0);  mul_1172 = None
        unsqueeze_1064: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 2);  unsqueeze_1063 = None
        unsqueeze_1065: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, 3);  unsqueeze_1064 = None
        mul_1173: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, 2.703287197231834e-05)
        mul_1174: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_1175: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1173, mul_1174);  mul_1173 = mul_1174 = None
        unsqueeze_1066: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1175, 0);  mul_1175 = None
        unsqueeze_1067: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, 2);  unsqueeze_1066 = None
        unsqueeze_1068: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1067, 3);  unsqueeze_1067 = None
        mul_1176: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, primals_222);  primals_222 = None
        unsqueeze_1069: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1176, 0);  mul_1176 = None
        unsqueeze_1070: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1069, 2);  unsqueeze_1069 = None
        unsqueeze_1071: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, 3);  unsqueeze_1070 = None
        mul_1177: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_1068);  sub_322 = unsqueeze_1068 = None
        sub_324: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_525, mul_1177);  convert_element_type_525 = mul_1177 = None
        sub_325: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_324, unsqueeze_1065);  sub_324 = unsqueeze_1065 = None
        mul_1178: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_1071);  sub_325 = unsqueeze_1071 = None
        mul_1179: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, squeeze_109);  sum_117 = squeeze_109 = None
        convert_element_type_527: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1178, torch.bfloat16);  mul_1178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_57 = torch.ops.aten.convolution_backward.default(convert_element_type_527, relu_35, convert_element_type_109, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_527 = convert_element_type_109 = None
        getitem_367: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = convolution_backward_57[0]
        getitem_368: "bf16[128, 128, 1, 7][896, 1, 896, 128]cuda:0" = convolution_backward_57[1];  convolution_backward_57 = None
        convert_element_type_528: "f32[128, 128, 1, 7][896, 1, 896, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_368, torch.float32);  getitem_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_58: "b8[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        where_58: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.where.self(le_58, full_default, getitem_367);  le_58 = getitem_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_529: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_58, torch.float32);  where_58 = None
        sum_118: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_529, [0, 2, 3])
        convert_element_type_107: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_35, torch.float32);  convolution_35 = None
        sub_326: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_107, unsqueeze_1074);  convert_element_type_107 = unsqueeze_1074 = None
        mul_1180: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_529, sub_326)
        sum_119: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1180, [0, 2, 3]);  mul_1180 = None
        mul_1181: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_118, 2.703287197231834e-05)
        unsqueeze_1075: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1181, 0);  mul_1181 = None
        unsqueeze_1076: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1075, 2);  unsqueeze_1075 = None
        unsqueeze_1077: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, 3);  unsqueeze_1076 = None
        mul_1182: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, 2.703287197231834e-05)
        mul_1183: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_1184: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1182, mul_1183);  mul_1182 = mul_1183 = None
        unsqueeze_1078: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1184, 0);  mul_1184 = None
        unsqueeze_1079: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, 2);  unsqueeze_1078 = None
        unsqueeze_1080: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1079, 3);  unsqueeze_1079 = None
        mul_1185: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, primals_216);  primals_216 = None
        unsqueeze_1081: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1185, 0);  mul_1185 = None
        unsqueeze_1082: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1081, 2);  unsqueeze_1081 = None
        unsqueeze_1083: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, 3);  unsqueeze_1082 = None
        mul_1186: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_1080);  sub_326 = unsqueeze_1080 = None
        sub_328: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_529, mul_1186);  convert_element_type_529 = mul_1186 = None
        sub_329: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_328, unsqueeze_1077);  sub_328 = unsqueeze_1077 = None
        mul_1187: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_1083);  sub_329 = unsqueeze_1083 = None
        mul_1188: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, squeeze_106);  sum_119 = squeeze_106 = None
        convert_element_type_531: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1187, torch.bfloat16);  mul_1187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_58 = torch.ops.aten.convolution_backward.default(convert_element_type_531, relu_34, convert_element_type_106, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_531 = convert_element_type_106 = None
        getitem_370: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = convolution_backward_58[0]
        getitem_371: "bf16[128, 128, 7, 1][896, 1, 128, 128]cuda:0" = convolution_backward_58[1];  convolution_backward_58 = None
        convert_element_type_532: "f32[128, 128, 7, 1][896, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_371, torch.float32);  getitem_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_59: "b8[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        where_59: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.where.self(le_59, full_default, getitem_370);  le_59 = getitem_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_533: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_59, torch.float32);  where_59 = None
        sum_120: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_533, [0, 2, 3])
        convert_element_type_104: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32);  convolution_34 = None
        sub_330: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_104, unsqueeze_1086);  convert_element_type_104 = unsqueeze_1086 = None
        mul_1189: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_533, sub_330)
        sum_121: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1189, [0, 2, 3]);  mul_1189 = None
        mul_1190: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_120, 2.703287197231834e-05)
        unsqueeze_1087: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1190, 0);  mul_1190 = None
        unsqueeze_1088: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1087, 2);  unsqueeze_1087 = None
        unsqueeze_1089: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, 3);  unsqueeze_1088 = None
        mul_1191: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, 2.703287197231834e-05)
        mul_1192: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_1193: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1191, mul_1192);  mul_1191 = mul_1192 = None
        unsqueeze_1090: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1193, 0);  mul_1193 = None
        unsqueeze_1091: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, 2);  unsqueeze_1090 = None
        unsqueeze_1092: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1091, 3);  unsqueeze_1091 = None
        mul_1194: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, primals_210);  primals_210 = None
        unsqueeze_1093: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1194, 0);  mul_1194 = None
        unsqueeze_1094: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1093, 2);  unsqueeze_1093 = None
        unsqueeze_1095: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, 3);  unsqueeze_1094 = None
        mul_1195: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_1092);  sub_330 = unsqueeze_1092 = None
        sub_332: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_533, mul_1195);  convert_element_type_533 = mul_1195 = None
        sub_333: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_332, unsqueeze_1089);  sub_332 = unsqueeze_1089 = None
        mul_1196: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_1095);  sub_333 = unsqueeze_1095 = None
        mul_1197: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, squeeze_103);  sum_121 = squeeze_103 = None
        convert_element_type_535: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1196, torch.bfloat16);  mul_1196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_59 = torch.ops.aten.convolution_backward.default(convert_element_type_535, cat_3, convert_element_type_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_535 = convert_element_type_103 = None
        getitem_373: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_59[0]
        getitem_374: "bf16[128, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_59[1];  convolution_backward_59 = None
        add_491: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.add.Tensor(avg_pool2d_backward_5, getitem_373);  avg_pool2d_backward_5 = getitem_373 = None
        convert_element_type_536: "f32[128, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_374, torch.float32);  getitem_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_33: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_33, getitem_73)
        mul_231: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        unsqueeze_132: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_204, -1)
        unsqueeze_133: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_237: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
        unsqueeze_134: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_205, -1);  primals_205 = None
        unsqueeze_135: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_169: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
        convert_element_type_102: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_169, torch.bfloat16);  add_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_33: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_102);  convert_element_type_102 = None
        le_60: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        where_60: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_60, full_default, slice_33);  le_60 = slice_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_537: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_60, torch.float32);  where_60 = None
        squeeze_99: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        unsqueeze_1096: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_1097: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, 2);  unsqueeze_1096 = None
        unsqueeze_1098: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1097, 3);  unsqueeze_1097 = None
        sum_122: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_537, [0, 2, 3])
        convert_element_type_101: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32);  convolution_33 = None
        sub_334: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_101, unsqueeze_1098);  convert_element_type_101 = unsqueeze_1098 = None
        mul_1198: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_537, sub_334)
        sum_123: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1198, [0, 2, 3]);  mul_1198 = None
        mul_1199: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_122, 2.703287197231834e-05)
        unsqueeze_1099: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1199, 0);  mul_1199 = None
        unsqueeze_1100: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1099, 2);  unsqueeze_1099 = None
        unsqueeze_1101: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, 3);  unsqueeze_1100 = None
        mul_1200: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, 2.703287197231834e-05)
        squeeze_100: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_1201: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_1202: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1200, mul_1201);  mul_1200 = mul_1201 = None
        unsqueeze_1102: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1202, 0);  mul_1202 = None
        unsqueeze_1103: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, 2);  unsqueeze_1102 = None
        unsqueeze_1104: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1103, 3);  unsqueeze_1103 = None
        mul_1203: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, primals_204);  primals_204 = None
        unsqueeze_1105: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1203, 0);  mul_1203 = None
        unsqueeze_1106: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1105, 2);  unsqueeze_1105 = None
        unsqueeze_1107: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, 3);  unsqueeze_1106 = None
        mul_1204: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_1104);  sub_334 = unsqueeze_1104 = None
        sub_336: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_537, mul_1204);  convert_element_type_537 = mul_1204 = None
        sub_337: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_336, unsqueeze_1101);  sub_336 = unsqueeze_1101 = None
        mul_1205: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_337, unsqueeze_1107);  sub_337 = unsqueeze_1107 = None
        mul_1206: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, squeeze_100);  sum_123 = squeeze_100 = None
        convert_element_type_539: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1205, torch.bfloat16);  mul_1205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_60 = torch.ops.aten.convolution_backward.default(convert_element_type_539, relu_32, convert_element_type_100, [0], [1, 1], [3, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_539 = convert_element_type_100 = None
        getitem_376: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = convolution_backward_60[0]
        getitem_377: "bf16[192, 128, 7, 1][896, 1, 128, 128]cuda:0" = convolution_backward_60[1];  convolution_backward_60 = None
        convert_element_type_540: "f32[192, 128, 7, 1][896, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_377, torch.float32);  getitem_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_61: "b8[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
        where_61: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.where.self(le_61, full_default, getitem_376);  le_61 = getitem_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_541: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_61, torch.float32);  where_61 = None
        sum_124: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_541, [0, 2, 3])
        convert_element_type_98: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32);  convolution_32 = None
        sub_338: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_98, unsqueeze_1110);  convert_element_type_98 = unsqueeze_1110 = None
        mul_1207: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_541, sub_338)
        sum_125: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1207, [0, 2, 3]);  mul_1207 = None
        mul_1208: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_124, 2.703287197231834e-05)
        unsqueeze_1111: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1208, 0);  mul_1208 = None
        unsqueeze_1112: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1111, 2);  unsqueeze_1111 = None
        unsqueeze_1113: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, 3);  unsqueeze_1112 = None
        mul_1209: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_125, 2.703287197231834e-05)
        mul_1210: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_1211: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1209, mul_1210);  mul_1209 = mul_1210 = None
        unsqueeze_1114: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1211, 0);  mul_1211 = None
        unsqueeze_1115: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, 2);  unsqueeze_1114 = None
        unsqueeze_1116: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1115, 3);  unsqueeze_1115 = None
        mul_1212: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, primals_198);  primals_198 = None
        unsqueeze_1117: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1212, 0);  mul_1212 = None
        unsqueeze_1118: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1117, 2);  unsqueeze_1117 = None
        unsqueeze_1119: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, 3);  unsqueeze_1118 = None
        mul_1213: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_1116);  sub_338 = unsqueeze_1116 = None
        sub_340: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_541, mul_1213);  convert_element_type_541 = mul_1213 = None
        sub_341: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_340, unsqueeze_1113);  sub_340 = unsqueeze_1113 = None
        mul_1214: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_1119);  sub_341 = unsqueeze_1119 = None
        mul_1215: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_125, squeeze_97);  sum_125 = squeeze_97 = None
        convert_element_type_543: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1214, torch.bfloat16);  mul_1214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_61 = torch.ops.aten.convolution_backward.default(convert_element_type_543, relu_31, convert_element_type_97, [0], [1, 1], [0, 3], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_543 = convert_element_type_97 = None
        getitem_379: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = convolution_backward_61[0]
        getitem_380: "bf16[128, 128, 1, 7][896, 1, 896, 128]cuda:0" = convolution_backward_61[1];  convolution_backward_61 = None
        convert_element_type_544: "f32[128, 128, 1, 7][896, 1, 896, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_380, torch.float32);  getitem_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_62: "b8[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        where_62: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.where.self(le_62, full_default, getitem_379);  le_62 = getitem_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_545: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_62, torch.float32);  where_62 = None
        sum_126: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_545, [0, 2, 3])
        convert_element_type_95: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32);  convolution_31 = None
        sub_342: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_95, unsqueeze_1122);  convert_element_type_95 = unsqueeze_1122 = None
        mul_1216: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_545, sub_342)
        sum_127: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1216, [0, 2, 3]);  mul_1216 = None
        mul_1217: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_126, 2.703287197231834e-05)
        unsqueeze_1123: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1217, 0);  mul_1217 = None
        unsqueeze_1124: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1123, 2);  unsqueeze_1123 = None
        unsqueeze_1125: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, 3);  unsqueeze_1124 = None
        mul_1218: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_127, 2.703287197231834e-05)
        mul_1219: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_1220: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1218, mul_1219);  mul_1218 = mul_1219 = None
        unsqueeze_1126: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1220, 0);  mul_1220 = None
        unsqueeze_1127: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, 2);  unsqueeze_1126 = None
        unsqueeze_1128: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1127, 3);  unsqueeze_1127 = None
        mul_1221: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, primals_192);  primals_192 = None
        unsqueeze_1129: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1221, 0);  mul_1221 = None
        unsqueeze_1130: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1129, 2);  unsqueeze_1129 = None
        unsqueeze_1131: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, 3);  unsqueeze_1130 = None
        mul_1222: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_342, unsqueeze_1128);  sub_342 = unsqueeze_1128 = None
        sub_344: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_545, mul_1222);  convert_element_type_545 = mul_1222 = None
        sub_345: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_344, unsqueeze_1125);  sub_344 = unsqueeze_1125 = None
        mul_1223: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_1131);  sub_345 = unsqueeze_1131 = None
        mul_1224: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_127, squeeze_94);  sum_127 = squeeze_94 = None
        convert_element_type_547: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1223, torch.bfloat16);  mul_1223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_62 = torch.ops.aten.convolution_backward.default(convert_element_type_547, cat_3, convert_element_type_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_547 = convert_element_type_94 = None
        getitem_382: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_62[0]
        getitem_383: "bf16[128, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_62[1];  convolution_backward_62 = None
        add_492: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.add.Tensor(add_491, getitem_382);  add_491 = getitem_382 = None
        convert_element_type_548: "f32[128, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_383, torch.float32);  getitem_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_30: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, getitem_67)
        mul_210: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        unsqueeze_120: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_186, -1)
        unsqueeze_121: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_216: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
        unsqueeze_122: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_187, -1);  primals_187 = None
        unsqueeze_123: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_154: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
        convert_element_type_93: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_154, torch.bfloat16);  add_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_30: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_93);  convert_element_type_93 = None
        le_63: "b8[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        where_63: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.where.self(le_63, full_default, slice_32);  le_63 = slice_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_549: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_63, torch.float32);  where_63 = None
        squeeze_90: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        unsqueeze_1132: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_1133: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, 2);  unsqueeze_1132 = None
        unsqueeze_1134: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1133, 3);  unsqueeze_1133 = None
        sum_128: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_549, [0, 2, 3])
        convert_element_type_92: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32);  convolution_30 = None
        sub_346: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_92, unsqueeze_1134);  convert_element_type_92 = unsqueeze_1134 = None
        mul_1225: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_549, sub_346)
        sum_129: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1225, [0, 2, 3]);  mul_1225 = None
        mul_1226: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_128, 2.703287197231834e-05)
        unsqueeze_1135: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1226, 0);  mul_1226 = None
        unsqueeze_1136: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1135, 2);  unsqueeze_1135 = None
        unsqueeze_1137: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, 3);  unsqueeze_1136 = None
        mul_1227: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_129, 2.703287197231834e-05)
        squeeze_91: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_1228: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_1229: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1227, mul_1228);  mul_1227 = mul_1228 = None
        unsqueeze_1138: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1229, 0);  mul_1229 = None
        unsqueeze_1139: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, 2);  unsqueeze_1138 = None
        unsqueeze_1140: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1139, 3);  unsqueeze_1139 = None
        mul_1230: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, primals_186);  primals_186 = None
        unsqueeze_1141: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1230, 0);  mul_1230 = None
        unsqueeze_1142: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1141, 2);  unsqueeze_1141 = None
        unsqueeze_1143: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, 3);  unsqueeze_1142 = None
        mul_1231: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_346, unsqueeze_1140);  sub_346 = unsqueeze_1140 = None
        sub_348: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_549, mul_1231);  convert_element_type_549 = mul_1231 = None
        sub_349: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_348, unsqueeze_1137);  sub_348 = unsqueeze_1137 = None
        mul_1232: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_349, unsqueeze_1143);  sub_349 = unsqueeze_1143 = None
        mul_1233: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_129, squeeze_91);  sum_129 = squeeze_91 = None
        convert_element_type_551: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1232, torch.bfloat16);  mul_1232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_63 = torch.ops.aten.convolution_backward.default(convert_element_type_551, cat_3, convert_element_type_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_551 = cat_3 = convert_element_type_91 = None
        getitem_385: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = convolution_backward_63[0]
        getitem_386: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_63[1];  convolution_backward_63 = None
        add_493: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.add.Tensor(add_492, getitem_385);  add_492 = getitem_385 = None
        convert_element_type_552: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_386, torch.float32);  getitem_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:100 in forward, code: return torch.cat(outputs, 1)
        slice_36: "bf16[128, 384, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_493, 1, 0, 384)
        slice_37: "bf16[128, 96, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_493, 1, 384, 480)
        slice_38: "bf16[128, 288, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.slice.Tensor(add_493, 1, 480, 768);  add_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:93 in _forward, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        full_default_65: "f32[36864, 1225][1225, 1]cuda:0" = torch.ops.aten.full.default([36864, 1225], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        clone_4: "bf16[128, 288, 17, 17][83232, 289, 17, 1]cuda:0" = torch.ops.aten.clone.default(slice_38, memory_format = torch.contiguous_format);  slice_38 = None
        view_6: "bf16[36864, 289][289, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [36864, 289]);  clone_4 = None
        _low_memory_max_pool_offsets_to_indices_2: "i64[128, 288, 17, 17][83232, 1, 4896, 288]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_65, [3, 3], [35, 35], [2, 2], [0, 0], [1, 1]);  getitem_65 = None
        clone_5: "i64[128, 288, 17, 17][83232, 289, 17, 1]cuda:0" = torch.ops.aten.clone.default(_low_memory_max_pool_offsets_to_indices_2, memory_format = torch.contiguous_format);  _low_memory_max_pool_offsets_to_indices_2 = None
        view_7: "i64[36864, 289][289, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [36864, 289]);  clone_5 = None
        convert_element_type_553: "f32[36864, 289][289, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        scatter_add_1: "f32[36864, 1225][1225, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_65, 1, view_7, convert_element_type_553);  full_default_65 = view_7 = convert_element_type_553 = None
        view_8: "f32[128, 288, 35, 35][352800, 1225, 35, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add_1, [128, 288, 35, 35]);  scatter_add_1 = None
        convert_element_type_554: "bf16[128, 288, 35, 35][352800, 1225, 35, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_8, torch.bfloat16);  view_8 = None
        clone_6: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.clone.default(convert_element_type_554, memory_format = torch.channels_last);  convert_element_type_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_29: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_29, getitem_63)
        mul_203: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        unsqueeze_116: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_180, -1)
        unsqueeze_117: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_209: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
        unsqueeze_118: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_181, -1);  primals_181 = None
        unsqueeze_119: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_149: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
        convert_element_type_90: "bf16[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_29: "bf16[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.relu.default(convert_element_type_90);  convert_element_type_90 = None
        le_64: "b8[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        where_64: "bf16[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.where.self(le_64, full_default, slice_37);  le_64 = slice_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_555: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.prims.convert_element_type.default(where_64, torch.float32);  where_64 = None
        squeeze_87: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        unsqueeze_1144: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_1145: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, 2);  unsqueeze_1144 = None
        unsqueeze_1146: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1145, 3);  unsqueeze_1145 = None
        sum_130: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_555, [0, 2, 3])
        convert_element_type_89: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_29, torch.float32);  convolution_29 = None
        sub_350: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_89, unsqueeze_1146);  convert_element_type_89 = unsqueeze_1146 = None
        mul_1234: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_555, sub_350)
        sum_131: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1234, [0, 2, 3]);  mul_1234 = None
        mul_1235: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_130, 2.703287197231834e-05)
        unsqueeze_1147: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1235, 0);  mul_1235 = None
        unsqueeze_1148: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1147, 2);  unsqueeze_1147 = None
        unsqueeze_1149: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, 3);  unsqueeze_1148 = None
        mul_1236: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_131, 2.703287197231834e-05)
        squeeze_88: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_1237: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_1238: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1236, mul_1237);  mul_1236 = mul_1237 = None
        unsqueeze_1150: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1238, 0);  mul_1238 = None
        unsqueeze_1151: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, 2);  unsqueeze_1150 = None
        unsqueeze_1152: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1151, 3);  unsqueeze_1151 = None
        mul_1239: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, primals_180);  primals_180 = None
        unsqueeze_1153: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1239, 0);  mul_1239 = None
        unsqueeze_1154: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1153, 2);  unsqueeze_1153 = None
        unsqueeze_1155: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, 3);  unsqueeze_1154 = None
        mul_1240: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_350, unsqueeze_1152);  sub_350 = unsqueeze_1152 = None
        sub_352: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_555, mul_1240);  convert_element_type_555 = mul_1240 = None
        sub_353: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_352, unsqueeze_1149);  sub_352 = unsqueeze_1149 = None
        mul_1241: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_353, unsqueeze_1155);  sub_353 = unsqueeze_1155 = None
        mul_1242: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_131, squeeze_88);  sum_131 = squeeze_88 = None
        convert_element_type_557: "bf16[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1241, torch.bfloat16);  mul_1241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_64 = torch.ops.aten.convolution_backward.default(convert_element_type_557, relu_28, convert_element_type_88, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_557 = convert_element_type_88 = None
        getitem_388: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = convolution_backward_64[0]
        getitem_389: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = convolution_backward_64[1];  convolution_backward_64 = None
        convert_element_type_558: "f32[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_389, torch.float32);  getitem_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_65: "b8[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
        where_65: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.where.self(le_65, full_default, getitem_388);  le_65 = getitem_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_559: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(where_65, torch.float32);  where_65 = None
        sum_132: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_559, [0, 2, 3])
        convert_element_type_86: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32);  convolution_28 = None
        sub_354: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_86, unsqueeze_1158);  convert_element_type_86 = unsqueeze_1158 = None
        mul_1243: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_559, sub_354)
        sum_133: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1243, [0, 2, 3]);  mul_1243 = None
        mul_1244: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_132, 6.3775510204081635e-06)
        unsqueeze_1159: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1244, 0);  mul_1244 = None
        unsqueeze_1160: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1159, 2);  unsqueeze_1159 = None
        unsqueeze_1161: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, 3);  unsqueeze_1160 = None
        mul_1245: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_133, 6.3775510204081635e-06)
        mul_1246: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_1247: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1245, mul_1246);  mul_1245 = mul_1246 = None
        unsqueeze_1162: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1247, 0);  mul_1247 = None
        unsqueeze_1163: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, 2);  unsqueeze_1162 = None
        unsqueeze_1164: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1163, 3);  unsqueeze_1163 = None
        mul_1248: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, primals_174);  primals_174 = None
        unsqueeze_1165: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1248, 0);  mul_1248 = None
        unsqueeze_1166: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1165, 2);  unsqueeze_1165 = None
        unsqueeze_1167: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, 3);  unsqueeze_1166 = None
        mul_1249: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_354, unsqueeze_1164);  sub_354 = unsqueeze_1164 = None
        sub_356: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_559, mul_1249);  convert_element_type_559 = mul_1249 = None
        sub_357: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_356, unsqueeze_1161);  sub_356 = unsqueeze_1161 = None
        mul_1250: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_357, unsqueeze_1167);  sub_357 = unsqueeze_1167 = None
        mul_1251: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_133, squeeze_85);  sum_133 = squeeze_85 = None
        convert_element_type_561: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1250, torch.bfloat16);  mul_1250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_65 = torch.ops.aten.convolution_backward.default(convert_element_type_561, relu_27, convert_element_type_85, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_561 = convert_element_type_85 = None
        getitem_391: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = convolution_backward_65[0]
        getitem_392: "bf16[96, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_65[1];  convolution_backward_65 = None
        convert_element_type_562: "f32[96, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_392, torch.float32);  getitem_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_66: "b8[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        where_66: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.where.self(le_66, full_default, getitem_391);  le_66 = getitem_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_563: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_66, torch.float32);  where_66 = None
        sum_134: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_563, [0, 2, 3])
        convert_element_type_83: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32);  convolution_27 = None
        sub_358: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_83, unsqueeze_1170);  convert_element_type_83 = unsqueeze_1170 = None
        mul_1252: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_563, sub_358)
        sum_135: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1252, [0, 2, 3]);  mul_1252 = None
        mul_1253: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_134, 6.3775510204081635e-06)
        unsqueeze_1171: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1253, 0);  mul_1253 = None
        unsqueeze_1172: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1171, 2);  unsqueeze_1171 = None
        unsqueeze_1173: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, 3);  unsqueeze_1172 = None
        mul_1254: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_135, 6.3775510204081635e-06)
        mul_1255: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_1256: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1254, mul_1255);  mul_1254 = mul_1255 = None
        unsqueeze_1174: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1256, 0);  mul_1256 = None
        unsqueeze_1175: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, 2);  unsqueeze_1174 = None
        unsqueeze_1176: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1175, 3);  unsqueeze_1175 = None
        mul_1257: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, primals_168);  primals_168 = None
        unsqueeze_1177: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1257, 0);  mul_1257 = None
        unsqueeze_1178: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1177, 2);  unsqueeze_1177 = None
        unsqueeze_1179: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, 3);  unsqueeze_1178 = None
        mul_1258: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_358, unsqueeze_1176);  sub_358 = unsqueeze_1176 = None
        sub_360: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_563, mul_1258);  convert_element_type_563 = mul_1258 = None
        sub_361: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_360, unsqueeze_1173);  sub_360 = unsqueeze_1173 = None
        mul_1259: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_361, unsqueeze_1179);  sub_361 = unsqueeze_1179 = None
        mul_1260: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_135, squeeze_82);  sum_135 = squeeze_82 = None
        convert_element_type_565: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1259, torch.bfloat16);  mul_1259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_66 = torch.ops.aten.convolution_backward.default(convert_element_type_565, cat_2, convert_element_type_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_565 = convert_element_type_82 = None
        getitem_394: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0" = convolution_backward_66[0]
        getitem_395: "bf16[64, 288, 1, 1][288, 1, 288, 288]cuda:0" = convolution_backward_66[1];  convolution_backward_66 = None
        add_494: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.add.Tensor(clone_6, getitem_394);  clone_6 = getitem_394 = None
        convert_element_type_566: "f32[64, 288, 1, 1][288, 1, 288, 288]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_395, torch.float32);  getitem_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_26: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, getitem_57)
        mul_182: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        unsqueeze_104: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_162, -1)
        unsqueeze_105: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_188: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
        unsqueeze_106: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_163, -1);  primals_163 = None
        unsqueeze_107: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_134: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
        convert_element_type_81: "bf16[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_134, torch.bfloat16);  add_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_26: "bf16[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_81);  convert_element_type_81 = None
        le_67: "b8[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        where_67: "bf16[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.where.self(le_67, full_default, slice_36);  le_67 = slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_567: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_67, torch.float32);  where_67 = None
        squeeze_78: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        unsqueeze_1180: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_1181: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, 2);  unsqueeze_1180 = None
        unsqueeze_1182: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1181, 3);  unsqueeze_1181 = None
        sum_136: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_567, [0, 2, 3])
        convert_element_type_80: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32);  convolution_26 = None
        sub_362: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_80, unsqueeze_1182);  convert_element_type_80 = unsqueeze_1182 = None
        mul_1261: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_567, sub_362)
        sum_137: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1261, [0, 2, 3]);  mul_1261 = None
        mul_1262: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_136, 2.703287197231834e-05)
        unsqueeze_1183: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1262, 0);  mul_1262 = None
        unsqueeze_1184: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1183, 2);  unsqueeze_1183 = None
        unsqueeze_1185: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, 3);  unsqueeze_1184 = None
        mul_1263: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_137, 2.703287197231834e-05)
        squeeze_79: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_1264: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_1265: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1263, mul_1264);  mul_1263 = mul_1264 = None
        unsqueeze_1186: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1265, 0);  mul_1265 = None
        unsqueeze_1187: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, 2);  unsqueeze_1186 = None
        unsqueeze_1188: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1187, 3);  unsqueeze_1187 = None
        mul_1266: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, primals_162);  primals_162 = None
        unsqueeze_1189: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1266, 0);  mul_1266 = None
        unsqueeze_1190: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1189, 2);  unsqueeze_1189 = None
        unsqueeze_1191: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, 3);  unsqueeze_1190 = None
        mul_1267: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_362, unsqueeze_1188);  sub_362 = unsqueeze_1188 = None
        sub_364: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_567, mul_1267);  convert_element_type_567 = mul_1267 = None
        sub_365: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_364, unsqueeze_1185);  sub_364 = unsqueeze_1185 = None
        mul_1268: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_365, unsqueeze_1191);  sub_365 = unsqueeze_1191 = None
        mul_1269: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_137, squeeze_79);  sum_137 = squeeze_79 = None
        convert_element_type_569: "bf16[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1268, torch.bfloat16);  mul_1268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_67 = torch.ops.aten.convolution_backward.default(convert_element_type_569, cat_2, convert_element_type_79, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_569 = cat_2 = convert_element_type_79 = None
        getitem_397: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0" = convolution_backward_67[0]
        getitem_398: "bf16[384, 288, 3, 3][2592, 1, 864, 288]cuda:0" = convolution_backward_67[1];  convolution_backward_67 = None
        add_495: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.add.Tensor(add_494, getitem_397);  add_494 = getitem_397 = None
        convert_element_type_570: "f32[384, 288, 3, 3][2592, 1, 864, 288]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_398, torch.float32);  getitem_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:65 in forward, code: return torch.cat(outputs, 1)
        slice_39: "bf16[128, 64, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.slice.Tensor(add_495, 1, 0, 64)
        slice_40: "bf16[128, 64, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.slice.Tensor(add_495, 1, 64, 128)
        slice_41: "bf16[128, 96, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.slice.Tensor(add_495, 1, 128, 224)
        slice_42: "bf16[128, 64, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.slice.Tensor(add_495, 1, 224, 288);  add_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_25: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, getitem_55)
        mul_175: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        unsqueeze_100: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_156, -1)
        unsqueeze_101: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_181: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
        unsqueeze_102: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_157, -1);  primals_157 = None
        unsqueeze_103: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_129: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
        convert_element_type_78: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_25: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_78);  convert_element_type_78 = None
        le_68: "b8[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        where_68: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.where.self(le_68, full_default, slice_42);  le_68 = slice_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_571: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_68, torch.float32);  where_68 = None
        squeeze_75: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        unsqueeze_1192: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_1193: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, 2);  unsqueeze_1192 = None
        unsqueeze_1194: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1193, 3);  unsqueeze_1193 = None
        sum_138: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_571, [0, 2, 3])
        convert_element_type_77: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32);  convolution_25 = None
        sub_366: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_77, unsqueeze_1194);  convert_element_type_77 = unsqueeze_1194 = None
        mul_1270: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_571, sub_366)
        sum_139: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1270, [0, 2, 3]);  mul_1270 = None
        mul_1271: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_138, 6.3775510204081635e-06)
        unsqueeze_1195: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1271, 0);  mul_1271 = None
        unsqueeze_1196: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1195, 2);  unsqueeze_1195 = None
        unsqueeze_1197: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, 3);  unsqueeze_1196 = None
        mul_1272: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_139, 6.3775510204081635e-06)
        squeeze_76: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_1273: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_1274: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1272, mul_1273);  mul_1272 = mul_1273 = None
        unsqueeze_1198: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1274, 0);  mul_1274 = None
        unsqueeze_1199: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, 2);  unsqueeze_1198 = None
        unsqueeze_1200: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1199, 3);  unsqueeze_1199 = None
        mul_1275: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, primals_156);  primals_156 = None
        unsqueeze_1201: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1275, 0);  mul_1275 = None
        unsqueeze_1202: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1201, 2);  unsqueeze_1201 = None
        unsqueeze_1203: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, 3);  unsqueeze_1202 = None
        mul_1276: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_366, unsqueeze_1200);  sub_366 = unsqueeze_1200 = None
        sub_368: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_571, mul_1276);  convert_element_type_571 = mul_1276 = None
        sub_369: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_368, unsqueeze_1197);  sub_368 = unsqueeze_1197 = None
        mul_1277: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_369, unsqueeze_1203);  sub_369 = unsqueeze_1203 = None
        mul_1278: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_139, squeeze_76);  sum_139 = squeeze_76 = None
        convert_element_type_573: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1277, torch.bfloat16);  mul_1277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_68 = torch.ops.aten.convolution_backward.default(convert_element_type_573, avg_pool2d_2, convert_element_type_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_573 = avg_pool2d_2 = convert_element_type_76 = None
        getitem_400: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0" = convolution_backward_68[0]
        getitem_401: "bf16[64, 288, 1, 1][288, 1, 288, 288]cuda:0" = convolution_backward_68[1];  convolution_backward_68 = None
        convert_element_type_574: "f32[64, 288, 1, 1][288, 1, 288, 288]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_401, torch.float32);  getitem_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:57 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_6: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(getitem_400, cat_1, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_24: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_24, getitem_53)
        mul_168: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        unsqueeze_96: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_150, -1)
        unsqueeze_97: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_174: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
        unsqueeze_98: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_151, -1);  primals_151 = None
        unsqueeze_99: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_124: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
        convert_element_type_75: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_24: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.relu.default(convert_element_type_75);  convert_element_type_75 = None
        le_69: "b8[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        where_69: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.where.self(le_69, full_default, slice_41);  le_69 = slice_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_575: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(where_69, torch.float32);  where_69 = None
        squeeze_72: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        unsqueeze_1204: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_1205: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, 2);  unsqueeze_1204 = None
        unsqueeze_1206: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1205, 3);  unsqueeze_1205 = None
        sum_140: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_575, [0, 2, 3])
        convert_element_type_74: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32);  convolution_24 = None
        sub_370: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_74, unsqueeze_1206);  convert_element_type_74 = unsqueeze_1206 = None
        mul_1279: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_575, sub_370)
        sum_141: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1279, [0, 2, 3]);  mul_1279 = None
        mul_1280: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_140, 6.3775510204081635e-06)
        unsqueeze_1207: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1280, 0);  mul_1280 = None
        unsqueeze_1208: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1207, 2);  unsqueeze_1207 = None
        unsqueeze_1209: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, 3);  unsqueeze_1208 = None
        mul_1281: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_141, 6.3775510204081635e-06)
        squeeze_73: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_1282: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_1283: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1281, mul_1282);  mul_1281 = mul_1282 = None
        unsqueeze_1210: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1283, 0);  mul_1283 = None
        unsqueeze_1211: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, 2);  unsqueeze_1210 = None
        unsqueeze_1212: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1211, 3);  unsqueeze_1211 = None
        mul_1284: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, primals_150);  primals_150 = None
        unsqueeze_1213: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1284, 0);  mul_1284 = None
        unsqueeze_1214: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1213, 2);  unsqueeze_1213 = None
        unsqueeze_1215: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, 3);  unsqueeze_1214 = None
        mul_1285: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_370, unsqueeze_1212);  sub_370 = unsqueeze_1212 = None
        sub_372: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_575, mul_1285);  convert_element_type_575 = mul_1285 = None
        sub_373: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_372, unsqueeze_1209);  sub_372 = unsqueeze_1209 = None
        mul_1286: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_373, unsqueeze_1215);  sub_373 = unsqueeze_1215 = None
        mul_1287: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_141, squeeze_73);  sum_141 = squeeze_73 = None
        convert_element_type_577: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1286, torch.bfloat16);  mul_1286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_69 = torch.ops.aten.convolution_backward.default(convert_element_type_577, relu_23, convert_element_type_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_577 = convert_element_type_73 = None
        getitem_403: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = convolution_backward_69[0]
        getitem_404: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = convolution_backward_69[1];  convolution_backward_69 = None
        convert_element_type_578: "f32[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_404, torch.float32);  getitem_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_70: "b8[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        where_70: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.where.self(le_70, full_default, getitem_403);  le_70 = getitem_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_579: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(where_70, torch.float32);  where_70 = None
        sum_142: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_579, [0, 2, 3])
        convert_element_type_71: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_23, torch.float32);  convolution_23 = None
        sub_374: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_71, unsqueeze_1218);  convert_element_type_71 = unsqueeze_1218 = None
        mul_1288: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_579, sub_374)
        sum_143: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1288, [0, 2, 3]);  mul_1288 = None
        mul_1289: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_142, 6.3775510204081635e-06)
        unsqueeze_1219: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1289, 0);  mul_1289 = None
        unsqueeze_1220: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1219, 2);  unsqueeze_1219 = None
        unsqueeze_1221: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, 3);  unsqueeze_1220 = None
        mul_1290: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_143, 6.3775510204081635e-06)
        mul_1291: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_1292: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1290, mul_1291);  mul_1290 = mul_1291 = None
        unsqueeze_1222: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1292, 0);  mul_1292 = None
        unsqueeze_1223: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, 2);  unsqueeze_1222 = None
        unsqueeze_1224: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1223, 3);  unsqueeze_1223 = None
        mul_1293: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, primals_144);  primals_144 = None
        unsqueeze_1225: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1293, 0);  mul_1293 = None
        unsqueeze_1226: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1225, 2);  unsqueeze_1225 = None
        unsqueeze_1227: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, 3);  unsqueeze_1226 = None
        mul_1294: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_374, unsqueeze_1224);  sub_374 = unsqueeze_1224 = None
        sub_376: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_579, mul_1294);  convert_element_type_579 = mul_1294 = None
        sub_377: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_376, unsqueeze_1221);  sub_376 = unsqueeze_1221 = None
        mul_1295: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_377, unsqueeze_1227);  sub_377 = unsqueeze_1227 = None
        mul_1296: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_143, squeeze_70);  sum_143 = squeeze_70 = None
        convert_element_type_581: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1295, torch.bfloat16);  mul_1295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_70 = torch.ops.aten.convolution_backward.default(convert_element_type_581, relu_22, convert_element_type_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_581 = convert_element_type_70 = None
        getitem_406: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = convolution_backward_70[0]
        getitem_407: "bf16[96, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_70[1];  convolution_backward_70 = None
        convert_element_type_582: "f32[96, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_407, torch.float32);  getitem_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_71: "b8[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_71: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.where.self(le_71, full_default, getitem_406);  le_71 = getitem_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_583: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_71, torch.float32);  where_71 = None
        sum_144: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_583, [0, 2, 3])
        convert_element_type_68: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        sub_378: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_68, unsqueeze_1230);  convert_element_type_68 = unsqueeze_1230 = None
        mul_1297: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_583, sub_378)
        sum_145: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1297, [0, 2, 3]);  mul_1297 = None
        mul_1298: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_144, 6.3775510204081635e-06)
        unsqueeze_1231: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1298, 0);  mul_1298 = None
        unsqueeze_1232: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1231, 2);  unsqueeze_1231 = None
        unsqueeze_1233: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, 3);  unsqueeze_1232 = None
        mul_1299: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_145, 6.3775510204081635e-06)
        mul_1300: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_1301: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1299, mul_1300);  mul_1299 = mul_1300 = None
        unsqueeze_1234: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1301, 0);  mul_1301 = None
        unsqueeze_1235: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, 2);  unsqueeze_1234 = None
        unsqueeze_1236: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1235, 3);  unsqueeze_1235 = None
        mul_1302: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, primals_138);  primals_138 = None
        unsqueeze_1237: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1302, 0);  mul_1302 = None
        unsqueeze_1238: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1237, 2);  unsqueeze_1237 = None
        unsqueeze_1239: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, 3);  unsqueeze_1238 = None
        mul_1303: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_378, unsqueeze_1236);  sub_378 = unsqueeze_1236 = None
        sub_380: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_583, mul_1303);  convert_element_type_583 = mul_1303 = None
        sub_381: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_380, unsqueeze_1233);  sub_380 = unsqueeze_1233 = None
        mul_1304: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_381, unsqueeze_1239);  sub_381 = unsqueeze_1239 = None
        mul_1305: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_145, squeeze_67);  sum_145 = squeeze_67 = None
        convert_element_type_585: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1304, torch.bfloat16);  mul_1304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_71 = torch.ops.aten.convolution_backward.default(convert_element_type_585, cat_1, convert_element_type_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_585 = convert_element_type_67 = None
        getitem_409: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0" = convolution_backward_71[0]
        getitem_410: "bf16[64, 288, 1, 1][288, 1, 288, 288]cuda:0" = convolution_backward_71[1];  convolution_backward_71 = None
        add_496: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.add.Tensor(avg_pool2d_backward_6, getitem_409);  avg_pool2d_backward_6 = getitem_409 = None
        convert_element_type_586: "f32[64, 288, 1, 1][288, 1, 288, 288]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_410, torch.float32);  getitem_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_21: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_21, getitem_47)
        mul_147: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        unsqueeze_84: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_132, -1)
        unsqueeze_85: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_153: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
        unsqueeze_86: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_133, -1);  primals_133 = None
        unsqueeze_87: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_109: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
        convert_element_type_66: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_21: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_66);  convert_element_type_66 = None
        le_72: "b8[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        where_72: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.where.self(le_72, full_default, slice_40);  le_72 = slice_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_587: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_72, torch.float32);  where_72 = None
        squeeze_63: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        unsqueeze_1240: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_1241: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, 2);  unsqueeze_1240 = None
        unsqueeze_1242: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1241, 3);  unsqueeze_1241 = None
        sum_146: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_587, [0, 2, 3])
        convert_element_type_65: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_21, torch.float32);  convolution_21 = None
        sub_382: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_65, unsqueeze_1242);  convert_element_type_65 = unsqueeze_1242 = None
        mul_1306: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_587, sub_382)
        sum_147: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1306, [0, 2, 3]);  mul_1306 = None
        mul_1307: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_146, 6.3775510204081635e-06)
        unsqueeze_1243: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1307, 0);  mul_1307 = None
        unsqueeze_1244: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1243, 2);  unsqueeze_1243 = None
        unsqueeze_1245: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, 3);  unsqueeze_1244 = None
        mul_1308: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_147, 6.3775510204081635e-06)
        squeeze_64: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_1309: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_1310: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1308, mul_1309);  mul_1308 = mul_1309 = None
        unsqueeze_1246: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1310, 0);  mul_1310 = None
        unsqueeze_1247: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, 2);  unsqueeze_1246 = None
        unsqueeze_1248: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1247, 3);  unsqueeze_1247 = None
        mul_1311: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, primals_132);  primals_132 = None
        unsqueeze_1249: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1311, 0);  mul_1311 = None
        unsqueeze_1250: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1249, 2);  unsqueeze_1249 = None
        unsqueeze_1251: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, 3);  unsqueeze_1250 = None
        mul_1312: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_382, unsqueeze_1248);  sub_382 = unsqueeze_1248 = None
        sub_384: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_587, mul_1312);  convert_element_type_587 = mul_1312 = None
        sub_385: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_384, unsqueeze_1245);  sub_384 = unsqueeze_1245 = None
        mul_1313: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_385, unsqueeze_1251);  sub_385 = unsqueeze_1251 = None
        mul_1314: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_147, squeeze_64);  sum_147 = squeeze_64 = None
        convert_element_type_589: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1313, torch.bfloat16);  mul_1313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_72 = torch.ops.aten.convolution_backward.default(convert_element_type_589, relu_20, convert_element_type_64, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_589 = convert_element_type_64 = None
        getitem_412: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = convolution_backward_72[0]
        getitem_413: "bf16[64, 48, 5, 5][1200, 1, 240, 48]cuda:0" = convolution_backward_72[1];  convolution_backward_72 = None
        convert_element_type_590: "f32[64, 48, 5, 5][1200, 1, 240, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_413, torch.float32);  getitem_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_73: "b8[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_73: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.where.self(le_73, full_default, getitem_412);  le_73 = getitem_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_591: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.prims.convert_element_type.default(where_73, torch.float32);  where_73 = None
        sum_148: "f32[48][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_591, [0, 2, 3])
        convert_element_type_62: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32);  convolution_20 = None
        sub_386: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_62, unsqueeze_1254);  convert_element_type_62 = unsqueeze_1254 = None
        mul_1315: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_591, sub_386)
        sum_149: "f32[48][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1315, [0, 2, 3]);  mul_1315 = None
        mul_1316: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_148, 6.3775510204081635e-06)
        unsqueeze_1255: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1316, 0);  mul_1316 = None
        unsqueeze_1256: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1255, 2);  unsqueeze_1255 = None
        unsqueeze_1257: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, 3);  unsqueeze_1256 = None
        mul_1317: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_149, 6.3775510204081635e-06)
        mul_1318: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_1319: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1317, mul_1318);  mul_1317 = mul_1318 = None
        unsqueeze_1258: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1319, 0);  mul_1319 = None
        unsqueeze_1259: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, 2);  unsqueeze_1258 = None
        unsqueeze_1260: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1259, 3);  unsqueeze_1259 = None
        mul_1320: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, primals_126);  primals_126 = None
        unsqueeze_1261: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1320, 0);  mul_1320 = None
        unsqueeze_1262: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1261, 2);  unsqueeze_1261 = None
        unsqueeze_1263: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, 3);  unsqueeze_1262 = None
        mul_1321: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.mul.Tensor(sub_386, unsqueeze_1260);  sub_386 = unsqueeze_1260 = None
        sub_388: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_591, mul_1321);  convert_element_type_591 = mul_1321 = None
        sub_389: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.sub.Tensor(sub_388, unsqueeze_1257);  sub_388 = unsqueeze_1257 = None
        mul_1322: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.mul.Tensor(sub_389, unsqueeze_1263);  sub_389 = unsqueeze_1263 = None
        mul_1323: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_149, squeeze_61);  sum_149 = squeeze_61 = None
        convert_element_type_593: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1322, torch.bfloat16);  mul_1322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_73 = torch.ops.aten.convolution_backward.default(convert_element_type_593, cat_1, convert_element_type_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_593 = convert_element_type_61 = None
        getitem_415: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0" = convolution_backward_73[0]
        getitem_416: "bf16[48, 288, 1, 1][288, 1, 288, 288]cuda:0" = convolution_backward_73[1];  convolution_backward_73 = None
        add_497: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.add.Tensor(add_496, getitem_415);  add_496 = getitem_415 = None
        convert_element_type_594: "f32[48, 288, 1, 1][288, 1, 288, 288]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_416, torch.float32);  getitem_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_19: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, getitem_43)
        mul_133: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        unsqueeze_76: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_120, -1)
        unsqueeze_77: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_139: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        unsqueeze_78: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
        unsqueeze_79: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_99: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
        convert_element_type_60: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_19: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_60);  convert_element_type_60 = None
        le_74: "b8[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_74: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.where.self(le_74, full_default, slice_39);  le_74 = slice_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_595: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_74, torch.float32);  where_74 = None
        squeeze_57: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        unsqueeze_1264: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_1265: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, 2);  unsqueeze_1264 = None
        unsqueeze_1266: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1265, 3);  unsqueeze_1265 = None
        sum_150: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_595, [0, 2, 3])
        convert_element_type_59: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32);  convolution_19 = None
        sub_390: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_59, unsqueeze_1266);  convert_element_type_59 = unsqueeze_1266 = None
        mul_1324: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_595, sub_390)
        sum_151: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1324, [0, 2, 3]);  mul_1324 = None
        mul_1325: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_150, 6.3775510204081635e-06)
        unsqueeze_1267: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1325, 0);  mul_1325 = None
        unsqueeze_1268: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1267, 2);  unsqueeze_1267 = None
        unsqueeze_1269: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, 3);  unsqueeze_1268 = None
        mul_1326: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_151, 6.3775510204081635e-06)
        squeeze_58: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_1327: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_1328: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1326, mul_1327);  mul_1326 = mul_1327 = None
        unsqueeze_1270: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1328, 0);  mul_1328 = None
        unsqueeze_1271: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, 2);  unsqueeze_1270 = None
        unsqueeze_1272: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1271, 3);  unsqueeze_1271 = None
        mul_1329: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, primals_120);  primals_120 = None
        unsqueeze_1273: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1329, 0);  mul_1329 = None
        unsqueeze_1274: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1273, 2);  unsqueeze_1273 = None
        unsqueeze_1275: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, 3);  unsqueeze_1274 = None
        mul_1330: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_390, unsqueeze_1272);  sub_390 = unsqueeze_1272 = None
        sub_392: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_595, mul_1330);  convert_element_type_595 = mul_1330 = None
        sub_393: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_392, unsqueeze_1269);  sub_392 = unsqueeze_1269 = None
        mul_1331: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_393, unsqueeze_1275);  sub_393 = unsqueeze_1275 = None
        mul_1332: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_151, squeeze_58);  sum_151 = squeeze_58 = None
        convert_element_type_597: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1331, torch.bfloat16);  mul_1331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_74 = torch.ops.aten.convolution_backward.default(convert_element_type_597, cat_1, convert_element_type_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_597 = cat_1 = convert_element_type_58 = None
        getitem_418: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0" = convolution_backward_74[0]
        getitem_419: "bf16[64, 288, 1, 1][288, 1, 288, 288]cuda:0" = convolution_backward_74[1];  convolution_backward_74 = None
        add_498: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.add.Tensor(add_497, getitem_418);  add_497 = getitem_418 = None
        convert_element_type_598: "f32[64, 288, 1, 1][288, 1, 288, 288]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_419, torch.float32);  getitem_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:65 in forward, code: return torch.cat(outputs, 1)
        slice_43: "bf16[128, 64, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.slice.Tensor(add_498, 1, 0, 64)
        slice_44: "bf16[128, 64, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.slice.Tensor(add_498, 1, 64, 128)
        slice_45: "bf16[128, 96, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.slice.Tensor(add_498, 1, 128, 224)
        slice_46: "bf16[128, 64, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.slice.Tensor(add_498, 1, 224, 288);  add_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_18: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, getitem_41)
        mul_126: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        unsqueeze_72: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_114, -1)
        unsqueeze_73: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_132: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
        unsqueeze_74: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
        unsqueeze_75: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_94: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
        convert_element_type_57: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16);  add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_18: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_57);  convert_element_type_57 = None
        le_75: "b8[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        where_75: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.where.self(le_75, full_default, slice_46);  le_75 = slice_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_599: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_75, torch.float32);  where_75 = None
        squeeze_54: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        unsqueeze_1276: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_1277: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, 2);  unsqueeze_1276 = None
        unsqueeze_1278: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1277, 3);  unsqueeze_1277 = None
        sum_152: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_599, [0, 2, 3])
        convert_element_type_56: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        sub_394: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_56, unsqueeze_1278);  convert_element_type_56 = unsqueeze_1278 = None
        mul_1333: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_599, sub_394)
        sum_153: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1333, [0, 2, 3]);  mul_1333 = None
        mul_1334: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_152, 6.3775510204081635e-06)
        unsqueeze_1279: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1334, 0);  mul_1334 = None
        unsqueeze_1280: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1279, 2);  unsqueeze_1279 = None
        unsqueeze_1281: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1280, 3);  unsqueeze_1280 = None
        mul_1335: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_153, 6.3775510204081635e-06)
        squeeze_55: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_1336: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_1337: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1335, mul_1336);  mul_1335 = mul_1336 = None
        unsqueeze_1282: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1337, 0);  mul_1337 = None
        unsqueeze_1283: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, 2);  unsqueeze_1282 = None
        unsqueeze_1284: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1283, 3);  unsqueeze_1283 = None
        mul_1338: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, primals_114);  primals_114 = None
        unsqueeze_1285: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1338, 0);  mul_1338 = None
        unsqueeze_1286: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1285, 2);  unsqueeze_1285 = None
        unsqueeze_1287: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1286, 3);  unsqueeze_1286 = None
        mul_1339: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_394, unsqueeze_1284);  sub_394 = unsqueeze_1284 = None
        sub_396: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_599, mul_1339);  convert_element_type_599 = mul_1339 = None
        sub_397: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_396, unsqueeze_1281);  sub_396 = unsqueeze_1281 = None
        mul_1340: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_397, unsqueeze_1287);  sub_397 = unsqueeze_1287 = None
        mul_1341: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_153, squeeze_55);  sum_153 = squeeze_55 = None
        convert_element_type_601: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1340, torch.bfloat16);  mul_1340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_75 = torch.ops.aten.convolution_backward.default(convert_element_type_601, avg_pool2d_1, convert_element_type_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_601 = avg_pool2d_1 = convert_element_type_55 = None
        getitem_421: "bf16[128, 256, 35, 35][313600, 1, 8960, 256]cuda:0" = convolution_backward_75[0]
        getitem_422: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_75[1];  convolution_backward_75 = None
        convert_element_type_602: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_422, torch.float32);  getitem_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:57 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_7: "bf16[128, 256, 35, 35][313600, 1, 8960, 256]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(getitem_421, cat, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_17: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_17, getitem_39)
        mul_119: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        unsqueeze_68: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_108, -1)
        unsqueeze_69: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_125: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
        unsqueeze_70: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
        unsqueeze_71: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_89: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
        convert_element_type_54: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_17: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.relu.default(convert_element_type_54);  convert_element_type_54 = None
        le_76: "b8[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_76: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.where.self(le_76, full_default, slice_45);  le_76 = slice_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_603: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(where_76, torch.float32);  where_76 = None
        squeeze_51: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        unsqueeze_1288: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_1289: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, 2);  unsqueeze_1288 = None
        unsqueeze_1290: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1289, 3);  unsqueeze_1289 = None
        sum_154: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_603, [0, 2, 3])
        convert_element_type_53: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32);  convolution_17 = None
        sub_398: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_53, unsqueeze_1290);  convert_element_type_53 = unsqueeze_1290 = None
        mul_1342: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_603, sub_398)
        sum_155: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1342, [0, 2, 3]);  mul_1342 = None
        mul_1343: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_154, 6.3775510204081635e-06)
        unsqueeze_1291: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1343, 0);  mul_1343 = None
        unsqueeze_1292: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1291, 2);  unsqueeze_1291 = None
        unsqueeze_1293: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1292, 3);  unsqueeze_1292 = None
        mul_1344: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_155, 6.3775510204081635e-06)
        squeeze_52: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_1345: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_1346: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1344, mul_1345);  mul_1344 = mul_1345 = None
        unsqueeze_1294: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1346, 0);  mul_1346 = None
        unsqueeze_1295: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, 2);  unsqueeze_1294 = None
        unsqueeze_1296: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1295, 3);  unsqueeze_1295 = None
        mul_1347: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, primals_108);  primals_108 = None
        unsqueeze_1297: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1347, 0);  mul_1347 = None
        unsqueeze_1298: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1297, 2);  unsqueeze_1297 = None
        unsqueeze_1299: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1298, 3);  unsqueeze_1298 = None
        mul_1348: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_398, unsqueeze_1296);  sub_398 = unsqueeze_1296 = None
        sub_400: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_603, mul_1348);  convert_element_type_603 = mul_1348 = None
        sub_401: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_400, unsqueeze_1293);  sub_400 = unsqueeze_1293 = None
        mul_1349: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_401, unsqueeze_1299);  sub_401 = unsqueeze_1299 = None
        mul_1350: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_155, squeeze_52);  sum_155 = squeeze_52 = None
        convert_element_type_605: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1349, torch.bfloat16);  mul_1349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_76 = torch.ops.aten.convolution_backward.default(convert_element_type_605, relu_16, convert_element_type_52, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_605 = convert_element_type_52 = None
        getitem_424: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = convolution_backward_76[0]
        getitem_425: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = convolution_backward_76[1];  convolution_backward_76 = None
        convert_element_type_606: "f32[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_425, torch.float32);  getitem_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_77: "b8[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_77: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.where.self(le_77, full_default, getitem_424);  le_77 = getitem_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_607: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(where_77, torch.float32);  where_77 = None
        sum_156: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_607, [0, 2, 3])
        convert_element_type_50: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32);  convolution_16 = None
        sub_402: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_50, unsqueeze_1302);  convert_element_type_50 = unsqueeze_1302 = None
        mul_1351: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_607, sub_402)
        sum_157: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1351, [0, 2, 3]);  mul_1351 = None
        mul_1352: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_156, 6.3775510204081635e-06)
        unsqueeze_1303: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1352, 0);  mul_1352 = None
        unsqueeze_1304: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1303, 2);  unsqueeze_1303 = None
        unsqueeze_1305: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1304, 3);  unsqueeze_1304 = None
        mul_1353: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, 6.3775510204081635e-06)
        mul_1354: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_1355: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1353, mul_1354);  mul_1353 = mul_1354 = None
        unsqueeze_1306: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1355, 0);  mul_1355 = None
        unsqueeze_1307: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, 2);  unsqueeze_1306 = None
        unsqueeze_1308: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1307, 3);  unsqueeze_1307 = None
        mul_1356: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, primals_102);  primals_102 = None
        unsqueeze_1309: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1356, 0);  mul_1356 = None
        unsqueeze_1310: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1309, 2);  unsqueeze_1309 = None
        unsqueeze_1311: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1310, 3);  unsqueeze_1310 = None
        mul_1357: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_402, unsqueeze_1308);  sub_402 = unsqueeze_1308 = None
        sub_404: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_607, mul_1357);  convert_element_type_607 = mul_1357 = None
        sub_405: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_404, unsqueeze_1305);  sub_404 = unsqueeze_1305 = None
        mul_1358: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_405, unsqueeze_1311);  sub_405 = unsqueeze_1311 = None
        mul_1359: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, squeeze_49);  sum_157 = squeeze_49 = None
        convert_element_type_609: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1358, torch.bfloat16);  mul_1358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_77 = torch.ops.aten.convolution_backward.default(convert_element_type_609, relu_15, convert_element_type_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_609 = convert_element_type_49 = None
        getitem_427: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = convolution_backward_77[0]
        getitem_428: "bf16[96, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_77[1];  convolution_backward_77 = None
        convert_element_type_610: "f32[96, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_428, torch.float32);  getitem_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_78: "b8[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_78: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.where.self(le_78, full_default, getitem_427);  le_78 = getitem_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_611: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_78, torch.float32);  where_78 = None
        sum_158: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_611, [0, 2, 3])
        convert_element_type_47: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32);  convolution_15 = None
        sub_406: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_47, unsqueeze_1314);  convert_element_type_47 = unsqueeze_1314 = None
        mul_1360: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_611, sub_406)
        sum_159: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1360, [0, 2, 3]);  mul_1360 = None
        mul_1361: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_158, 6.3775510204081635e-06)
        unsqueeze_1315: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1361, 0);  mul_1361 = None
        unsqueeze_1316: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1315, 2);  unsqueeze_1315 = None
        unsqueeze_1317: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1316, 3);  unsqueeze_1316 = None
        mul_1362: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_159, 6.3775510204081635e-06)
        mul_1363: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_1364: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1362, mul_1363);  mul_1362 = mul_1363 = None
        unsqueeze_1318: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1364, 0);  mul_1364 = None
        unsqueeze_1319: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, 2);  unsqueeze_1318 = None
        unsqueeze_1320: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1319, 3);  unsqueeze_1319 = None
        mul_1365: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, primals_96);  primals_96 = None
        unsqueeze_1321: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1365, 0);  mul_1365 = None
        unsqueeze_1322: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1321, 2);  unsqueeze_1321 = None
        unsqueeze_1323: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1322, 3);  unsqueeze_1322 = None
        mul_1366: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_406, unsqueeze_1320);  sub_406 = unsqueeze_1320 = None
        sub_408: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_611, mul_1366);  convert_element_type_611 = mul_1366 = None
        sub_409: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_408, unsqueeze_1317);  sub_408 = unsqueeze_1317 = None
        mul_1367: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_409, unsqueeze_1323);  sub_409 = unsqueeze_1323 = None
        mul_1368: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_159, squeeze_46);  sum_159 = squeeze_46 = None
        convert_element_type_613: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1367, torch.bfloat16);  mul_1367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_78 = torch.ops.aten.convolution_backward.default(convert_element_type_613, cat, convert_element_type_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_613 = convert_element_type_46 = None
        getitem_430: "bf16[128, 256, 35, 35][313600, 1, 8960, 256]cuda:0" = convolution_backward_78[0]
        getitem_431: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_78[1];  convolution_backward_78 = None
        add_499: "bf16[128, 256, 35, 35][313600, 1, 8960, 256]cuda:0" = torch.ops.aten.add.Tensor(avg_pool2d_backward_7, getitem_430);  avg_pool2d_backward_7 = getitem_430 = None
        convert_element_type_614: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_431, torch.float32);  getitem_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_14: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, getitem_33)
        mul_98: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        unsqueeze_56: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_57: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
        unsqueeze_59: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_74: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
        convert_element_type_45: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_14: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_45);  convert_element_type_45 = None
        le_79: "b8[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_79: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.where.self(le_79, full_default, slice_44);  le_79 = slice_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_615: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_79, torch.float32);  where_79 = None
        squeeze_42: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        unsqueeze_1324: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_1325: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, 2);  unsqueeze_1324 = None
        unsqueeze_1326: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1325, 3);  unsqueeze_1325 = None
        sum_160: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_615, [0, 2, 3])
        convert_element_type_44: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        sub_410: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_44, unsqueeze_1326);  convert_element_type_44 = unsqueeze_1326 = None
        mul_1369: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_615, sub_410)
        sum_161: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1369, [0, 2, 3]);  mul_1369 = None
        mul_1370: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_160, 6.3775510204081635e-06)
        unsqueeze_1327: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1370, 0);  mul_1370 = None
        unsqueeze_1328: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1327, 2);  unsqueeze_1327 = None
        unsqueeze_1329: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1328, 3);  unsqueeze_1328 = None
        mul_1371: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_161, 6.3775510204081635e-06)
        squeeze_43: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_1372: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_1373: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1371, mul_1372);  mul_1371 = mul_1372 = None
        unsqueeze_1330: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1373, 0);  mul_1373 = None
        unsqueeze_1331: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, 2);  unsqueeze_1330 = None
        unsqueeze_1332: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1331, 3);  unsqueeze_1331 = None
        mul_1374: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, primals_90);  primals_90 = None
        unsqueeze_1333: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1374, 0);  mul_1374 = None
        unsqueeze_1334: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1333, 2);  unsqueeze_1333 = None
        unsqueeze_1335: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1334, 3);  unsqueeze_1334 = None
        mul_1375: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_410, unsqueeze_1332);  sub_410 = unsqueeze_1332 = None
        sub_412: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_615, mul_1375);  convert_element_type_615 = mul_1375 = None
        sub_413: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_412, unsqueeze_1329);  sub_412 = unsqueeze_1329 = None
        mul_1376: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_413, unsqueeze_1335);  sub_413 = unsqueeze_1335 = None
        mul_1377: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_161, squeeze_43);  sum_161 = squeeze_43 = None
        convert_element_type_617: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1376, torch.bfloat16);  mul_1376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_79 = torch.ops.aten.convolution_backward.default(convert_element_type_617, relu_13, convert_element_type_43, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_617 = convert_element_type_43 = None
        getitem_433: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = convolution_backward_79[0]
        getitem_434: "bf16[64, 48, 5, 5][1200, 1, 240, 48]cuda:0" = convolution_backward_79[1];  convolution_backward_79 = None
        convert_element_type_618: "f32[64, 48, 5, 5][1200, 1, 240, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_434, torch.float32);  getitem_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_80: "b8[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_80: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.where.self(le_80, full_default, getitem_433);  le_80 = getitem_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_619: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.prims.convert_element_type.default(where_80, torch.float32);  where_80 = None
        sum_162: "f32[48][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_619, [0, 2, 3])
        convert_element_type_41: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32);  convolution_13 = None
        sub_414: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_41, unsqueeze_1338);  convert_element_type_41 = unsqueeze_1338 = None
        mul_1378: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_619, sub_414)
        sum_163: "f32[48][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1378, [0, 2, 3]);  mul_1378 = None
        mul_1379: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_162, 6.3775510204081635e-06)
        unsqueeze_1339: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1379, 0);  mul_1379 = None
        unsqueeze_1340: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1339, 2);  unsqueeze_1339 = None
        unsqueeze_1341: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1340, 3);  unsqueeze_1340 = None
        mul_1380: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_163, 6.3775510204081635e-06)
        mul_1381: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_1382: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1380, mul_1381);  mul_1380 = mul_1381 = None
        unsqueeze_1342: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1382, 0);  mul_1382 = None
        unsqueeze_1343: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, 2);  unsqueeze_1342 = None
        unsqueeze_1344: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1343, 3);  unsqueeze_1343 = None
        mul_1383: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, primals_84);  primals_84 = None
        unsqueeze_1345: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1383, 0);  mul_1383 = None
        unsqueeze_1346: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1345, 2);  unsqueeze_1345 = None
        unsqueeze_1347: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1346, 3);  unsqueeze_1346 = None
        mul_1384: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.mul.Tensor(sub_414, unsqueeze_1344);  sub_414 = unsqueeze_1344 = None
        sub_416: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_619, mul_1384);  convert_element_type_619 = mul_1384 = None
        sub_417: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.sub.Tensor(sub_416, unsqueeze_1341);  sub_416 = unsqueeze_1341 = None
        mul_1385: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.mul.Tensor(sub_417, unsqueeze_1347);  sub_417 = unsqueeze_1347 = None
        mul_1386: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_163, squeeze_40);  sum_163 = squeeze_40 = None
        convert_element_type_621: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1385, torch.bfloat16);  mul_1385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_80 = torch.ops.aten.convolution_backward.default(convert_element_type_621, cat, convert_element_type_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_621 = convert_element_type_40 = None
        getitem_436: "bf16[128, 256, 35, 35][313600, 1, 8960, 256]cuda:0" = convolution_backward_80[0]
        getitem_437: "bf16[48, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_80[1];  convolution_backward_80 = None
        add_500: "bf16[128, 256, 35, 35][313600, 1, 8960, 256]cuda:0" = torch.ops.aten.add.Tensor(add_499, getitem_436);  add_499 = getitem_436 = None
        convert_element_type_622: "f32[48, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_437, torch.float32);  getitem_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_12: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_12, getitem_29)
        mul_84: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        unsqueeze_48: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_49: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_90: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        unsqueeze_50: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
        unsqueeze_51: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_64: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
        convert_element_type_39: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_12: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_39);  convert_element_type_39 = None
        le_81: "b8[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_81: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.where.self(le_81, full_default, slice_43);  le_81 = slice_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_623: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_81, torch.float32);  where_81 = None
        squeeze_36: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        unsqueeze_1348: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_1349: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, 2);  unsqueeze_1348 = None
        unsqueeze_1350: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1349, 3);  unsqueeze_1349 = None
        sum_164: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_623, [0, 2, 3])
        convert_element_type_38: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        sub_418: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_38, unsqueeze_1350);  convert_element_type_38 = unsqueeze_1350 = None
        mul_1387: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_623, sub_418)
        sum_165: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1387, [0, 2, 3]);  mul_1387 = None
        mul_1388: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_164, 6.3775510204081635e-06)
        unsqueeze_1351: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1388, 0);  mul_1388 = None
        unsqueeze_1352: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1351, 2);  unsqueeze_1351 = None
        unsqueeze_1353: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1352, 3);  unsqueeze_1352 = None
        mul_1389: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_165, 6.3775510204081635e-06)
        squeeze_37: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_1390: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_1391: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1389, mul_1390);  mul_1389 = mul_1390 = None
        unsqueeze_1354: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1391, 0);  mul_1391 = None
        unsqueeze_1355: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, 2);  unsqueeze_1354 = None
        unsqueeze_1356: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1355, 3);  unsqueeze_1355 = None
        mul_1392: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, primals_78);  primals_78 = None
        unsqueeze_1357: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1392, 0);  mul_1392 = None
        unsqueeze_1358: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1357, 2);  unsqueeze_1357 = None
        unsqueeze_1359: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1358, 3);  unsqueeze_1358 = None
        mul_1393: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_418, unsqueeze_1356);  sub_418 = unsqueeze_1356 = None
        sub_420: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_623, mul_1393);  convert_element_type_623 = mul_1393 = None
        sub_421: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_420, unsqueeze_1353);  sub_420 = unsqueeze_1353 = None
        mul_1394: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_421, unsqueeze_1359);  sub_421 = unsqueeze_1359 = None
        mul_1395: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_165, squeeze_37);  sum_165 = squeeze_37 = None
        convert_element_type_625: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1394, torch.bfloat16);  mul_1394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_81 = torch.ops.aten.convolution_backward.default(convert_element_type_625, cat, convert_element_type_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_625 = cat = convert_element_type_37 = None
        getitem_439: "bf16[128, 256, 35, 35][313600, 1, 8960, 256]cuda:0" = convolution_backward_81[0]
        getitem_440: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_81[1];  convolution_backward_81 = None
        add_501: "bf16[128, 256, 35, 35][313600, 1, 8960, 256]cuda:0" = torch.ops.aten.add.Tensor(add_500, getitem_439);  add_500 = getitem_439 = None
        convert_element_type_626: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_440, torch.float32);  getitem_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:65 in forward, code: return torch.cat(outputs, 1)
        slice_47: "bf16[128, 64, 35, 35][313600, 1, 8960, 256]cuda:0" = torch.ops.aten.slice.Tensor(add_501, 1, 0, 64)
        slice_48: "bf16[128, 64, 35, 35][313600, 1, 8960, 256]cuda:0" = torch.ops.aten.slice.Tensor(add_501, 1, 64, 128)
        slice_49: "bf16[128, 96, 35, 35][313600, 1, 8960, 256]cuda:0" = torch.ops.aten.slice.Tensor(add_501, 1, 128, 224)
        slice_50: "bf16[128, 32, 35, 35][313600, 1, 8960, 256]cuda:0" = torch.ops.aten.slice.Tensor(add_501, 1, 224, 256);  add_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_11: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, getitem_27)
        mul_77: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        unsqueeze_44: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_45: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_73, -1);  primals_73 = None
        unsqueeze_47: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_59: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        convert_element_type_36: "bf16[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_11: "bf16[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.relu.default(convert_element_type_36);  convert_element_type_36 = None
        le_82: "b8[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_82: "bf16[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.where.self(le_82, full_default, slice_50);  le_82 = slice_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_627: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.prims.convert_element_type.default(where_82, torch.float32);  where_82 = None
        squeeze_33: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        unsqueeze_1360: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_1361: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, 2);  unsqueeze_1360 = None
        unsqueeze_1362: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1361, 3);  unsqueeze_1361 = None
        sum_166: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_627, [0, 2, 3])
        convert_element_type_35: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32);  convolution_11 = None
        sub_422: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_35, unsqueeze_1362);  convert_element_type_35 = unsqueeze_1362 = None
        mul_1396: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_627, sub_422)
        sum_167: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1396, [0, 2, 3]);  mul_1396 = None
        mul_1397: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_166, 6.3775510204081635e-06)
        unsqueeze_1363: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1397, 0);  mul_1397 = None
        unsqueeze_1364: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1363, 2);  unsqueeze_1363 = None
        unsqueeze_1365: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1364, 3);  unsqueeze_1364 = None
        mul_1398: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_167, 6.3775510204081635e-06)
        squeeze_34: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_1399: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_1400: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1398, mul_1399);  mul_1398 = mul_1399 = None
        unsqueeze_1366: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1400, 0);  mul_1400 = None
        unsqueeze_1367: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, 2);  unsqueeze_1366 = None
        unsqueeze_1368: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1367, 3);  unsqueeze_1367 = None
        mul_1401: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, primals_72);  primals_72 = None
        unsqueeze_1369: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1401, 0);  mul_1401 = None
        unsqueeze_1370: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1369, 2);  unsqueeze_1369 = None
        unsqueeze_1371: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1370, 3);  unsqueeze_1370 = None
        mul_1402: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_422, unsqueeze_1368);  sub_422 = unsqueeze_1368 = None
        sub_424: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_627, mul_1402);  convert_element_type_627 = mul_1402 = None
        sub_425: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.sub.Tensor(sub_424, unsqueeze_1365);  sub_424 = unsqueeze_1365 = None
        mul_1403: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_425, unsqueeze_1371);  sub_425 = unsqueeze_1371 = None
        mul_1404: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_167, squeeze_34);  sum_167 = squeeze_34 = None
        convert_element_type_629: "bf16[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1403, torch.bfloat16);  mul_1403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_82 = torch.ops.aten.convolution_backward.default(convert_element_type_629, avg_pool2d, convert_element_type_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_629 = avg_pool2d = convert_element_type_34 = None
        getitem_442: "bf16[128, 192, 35, 35][235200, 1, 6720, 192]cuda:0" = convolution_backward_82[0]
        getitem_443: "bf16[32, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_82[1];  convolution_backward_82 = None
        convert_element_type_630: "f32[32, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_443, torch.float32);  getitem_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:57 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_backward_8: "bf16[128, 192, 35, 35][235200, 1, 6720, 192]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(getitem_442, getitem_12, [3, 3], [1, 1], [1, 1], False, True, None);  getitem_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_10: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, getitem_25)
        mul_70: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        unsqueeze_40: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_41: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
        unsqueeze_43: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_54: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
        convert_element_type_33: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_10: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.relu.default(convert_element_type_33);  convert_element_type_33 = None
        le_83: "b8[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_83: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.where.self(le_83, full_default, slice_49);  le_83 = slice_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_631: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(where_83, torch.float32);  where_83 = None
        squeeze_30: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        unsqueeze_1372: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_1373: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, 2);  unsqueeze_1372 = None
        unsqueeze_1374: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1373, 3);  unsqueeze_1373 = None
        sum_168: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_631, [0, 2, 3])
        convert_element_type_32: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32);  convolution_10 = None
        sub_426: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_32, unsqueeze_1374);  convert_element_type_32 = unsqueeze_1374 = None
        mul_1405: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_631, sub_426)
        sum_169: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1405, [0, 2, 3]);  mul_1405 = None
        mul_1406: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_168, 6.3775510204081635e-06)
        unsqueeze_1375: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1406, 0);  mul_1406 = None
        unsqueeze_1376: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1375, 2);  unsqueeze_1375 = None
        unsqueeze_1377: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1376, 3);  unsqueeze_1376 = None
        mul_1407: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, 6.3775510204081635e-06)
        squeeze_31: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_1408: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_1409: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1407, mul_1408);  mul_1407 = mul_1408 = None
        unsqueeze_1378: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1409, 0);  mul_1409 = None
        unsqueeze_1379: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, 2);  unsqueeze_1378 = None
        unsqueeze_1380: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1379, 3);  unsqueeze_1379 = None
        mul_1410: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, primals_66);  primals_66 = None
        unsqueeze_1381: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1410, 0);  mul_1410 = None
        unsqueeze_1382: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1381, 2);  unsqueeze_1381 = None
        unsqueeze_1383: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1382, 3);  unsqueeze_1382 = None
        mul_1411: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_426, unsqueeze_1380);  sub_426 = unsqueeze_1380 = None
        sub_428: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_631, mul_1411);  convert_element_type_631 = mul_1411 = None
        sub_429: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_428, unsqueeze_1377);  sub_428 = unsqueeze_1377 = None
        mul_1412: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_429, unsqueeze_1383);  sub_429 = unsqueeze_1383 = None
        mul_1413: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, squeeze_31);  sum_169 = squeeze_31 = None
        convert_element_type_633: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1412, torch.bfloat16);  mul_1412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_83 = torch.ops.aten.convolution_backward.default(convert_element_type_633, relu_9, convert_element_type_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_633 = convert_element_type_31 = None
        getitem_445: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = convolution_backward_83[0]
        getitem_446: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = convolution_backward_83[1];  convolution_backward_83 = None
        convert_element_type_634: "f32[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_446, torch.float32);  getitem_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_84: "b8[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_84: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.where.self(le_84, full_default, getitem_445);  le_84 = getitem_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_635: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(where_84, torch.float32);  where_84 = None
        sum_170: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_635, [0, 2, 3])
        convert_element_type_29: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        sub_430: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_29, unsqueeze_1386);  convert_element_type_29 = unsqueeze_1386 = None
        mul_1414: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_635, sub_430)
        sum_171: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1414, [0, 2, 3]);  mul_1414 = None
        mul_1415: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_170, 6.3775510204081635e-06)
        unsqueeze_1387: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1415, 0);  mul_1415 = None
        unsqueeze_1388: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1387, 2);  unsqueeze_1387 = None
        unsqueeze_1389: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1388, 3);  unsqueeze_1388 = None
        mul_1416: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_171, 6.3775510204081635e-06)
        mul_1417: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_1418: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1416, mul_1417);  mul_1416 = mul_1417 = None
        unsqueeze_1390: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1418, 0);  mul_1418 = None
        unsqueeze_1391: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, 2);  unsqueeze_1390 = None
        unsqueeze_1392: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1391, 3);  unsqueeze_1391 = None
        mul_1419: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, primals_60);  primals_60 = None
        unsqueeze_1393: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1419, 0);  mul_1419 = None
        unsqueeze_1394: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1393, 2);  unsqueeze_1393 = None
        unsqueeze_1395: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1394, 3);  unsqueeze_1394 = None
        mul_1420: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_430, unsqueeze_1392);  sub_430 = unsqueeze_1392 = None
        sub_432: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_635, mul_1420);  convert_element_type_635 = mul_1420 = None
        sub_433: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_432, unsqueeze_1389);  sub_432 = unsqueeze_1389 = None
        mul_1421: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_433, unsqueeze_1395);  sub_433 = unsqueeze_1395 = None
        mul_1422: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_171, squeeze_28);  sum_171 = squeeze_28 = None
        convert_element_type_637: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1421, torch.bfloat16);  mul_1421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_84 = torch.ops.aten.convolution_backward.default(convert_element_type_637, relu_8, convert_element_type_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_637 = convert_element_type_28 = None
        getitem_448: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = convolution_backward_84[0]
        getitem_449: "bf16[96, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_84[1];  convolution_backward_84 = None
        convert_element_type_638: "f32[96, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_449, torch.float32);  getitem_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_85: "b8[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_85: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.where.self(le_85, full_default, getitem_448);  le_85 = getitem_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_639: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_85, torch.float32);  where_85 = None
        sum_172: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_639, [0, 2, 3])
        convert_element_type_26: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32);  convolution_8 = None
        sub_434: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_26, unsqueeze_1398);  convert_element_type_26 = unsqueeze_1398 = None
        mul_1423: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_639, sub_434)
        sum_173: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1423, [0, 2, 3]);  mul_1423 = None
        mul_1424: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_172, 6.3775510204081635e-06)
        unsqueeze_1399: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1424, 0);  mul_1424 = None
        unsqueeze_1400: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1399, 2);  unsqueeze_1399 = None
        unsqueeze_1401: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1400, 3);  unsqueeze_1400 = None
        mul_1425: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_173, 6.3775510204081635e-06)
        mul_1426: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_1427: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1425, mul_1426);  mul_1425 = mul_1426 = None
        unsqueeze_1402: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1427, 0);  mul_1427 = None
        unsqueeze_1403: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1402, 2);  unsqueeze_1402 = None
        unsqueeze_1404: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1403, 3);  unsqueeze_1403 = None
        mul_1428: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  primals_54 = None
        unsqueeze_1405: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1428, 0);  mul_1428 = None
        unsqueeze_1406: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1405, 2);  unsqueeze_1405 = None
        unsqueeze_1407: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1406, 3);  unsqueeze_1406 = None
        mul_1429: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_434, unsqueeze_1404);  sub_434 = unsqueeze_1404 = None
        sub_436: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_639, mul_1429);  convert_element_type_639 = mul_1429 = None
        sub_437: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_436, unsqueeze_1401);  sub_436 = unsqueeze_1401 = None
        mul_1430: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_437, unsqueeze_1407);  sub_437 = unsqueeze_1407 = None
        mul_1431: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_173, squeeze_25);  sum_173 = squeeze_25 = None
        convert_element_type_641: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1430, torch.bfloat16);  mul_1430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_85 = torch.ops.aten.convolution_backward.default(convert_element_type_641, getitem_12, convert_element_type_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_641 = convert_element_type_25 = None
        getitem_451: "bf16[128, 192, 35, 35][235200, 1, 6720, 192]cuda:0" = convolution_backward_85[0]
        getitem_452: "bf16[64, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_85[1];  convolution_backward_85 = None
        add_502: "bf16[128, 192, 35, 35][235200, 1, 6720, 192]cuda:0" = torch.ops.aten.add.Tensor(avg_pool2d_backward_8, getitem_451);  avg_pool2d_backward_8 = getitem_451 = None
        convert_element_type_642: "f32[64, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_452, torch.float32);  getitem_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_7: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, getitem_19)
        mul_49: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        unsqueeze_28: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_49, -1);  primals_49 = None
        unsqueeze_31: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_39: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        convert_element_type_24: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_7: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_24);  convert_element_type_24 = None
        le_86: "b8[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_86: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.where.self(le_86, full_default, slice_48);  le_86 = slice_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_643: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_86, torch.float32);  where_86 = None
        squeeze_21: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        unsqueeze_1408: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_1409: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1408, 2);  unsqueeze_1408 = None
        unsqueeze_1410: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1409, 3);  unsqueeze_1409 = None
        sum_174: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_643, [0, 2, 3])
        convert_element_type_23: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        sub_438: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_23, unsqueeze_1410);  convert_element_type_23 = unsqueeze_1410 = None
        mul_1432: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_643, sub_438)
        sum_175: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1432, [0, 2, 3]);  mul_1432 = None
        mul_1433: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_174, 6.3775510204081635e-06)
        unsqueeze_1411: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1433, 0);  mul_1433 = None
        unsqueeze_1412: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1411, 2);  unsqueeze_1411 = None
        unsqueeze_1413: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1412, 3);  unsqueeze_1412 = None
        mul_1434: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, 6.3775510204081635e-06)
        squeeze_22: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_1435: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_1436: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1434, mul_1435);  mul_1434 = mul_1435 = None
        unsqueeze_1414: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1436, 0);  mul_1436 = None
        unsqueeze_1415: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1414, 2);  unsqueeze_1414 = None
        unsqueeze_1416: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1415, 3);  unsqueeze_1415 = None
        mul_1437: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  primals_48 = None
        unsqueeze_1417: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1437, 0);  mul_1437 = None
        unsqueeze_1418: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1417, 2);  unsqueeze_1417 = None
        unsqueeze_1419: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1418, 3);  unsqueeze_1418 = None
        mul_1438: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_438, unsqueeze_1416);  sub_438 = unsqueeze_1416 = None
        sub_440: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_643, mul_1438);  convert_element_type_643 = mul_1438 = None
        sub_441: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_440, unsqueeze_1413);  sub_440 = unsqueeze_1413 = None
        mul_1439: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_441, unsqueeze_1419);  sub_441 = unsqueeze_1419 = None
        mul_1440: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, squeeze_22);  sum_175 = squeeze_22 = None
        convert_element_type_645: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1439, torch.bfloat16);  mul_1439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_86 = torch.ops.aten.convolution_backward.default(convert_element_type_645, relu_6, convert_element_type_22, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_645 = convert_element_type_22 = None
        getitem_454: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = convolution_backward_86[0]
        getitem_455: "bf16[64, 48, 5, 5][1200, 1, 240, 48]cuda:0" = convolution_backward_86[1];  convolution_backward_86 = None
        convert_element_type_646: "f32[64, 48, 5, 5][1200, 1, 240, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_455, torch.float32);  getitem_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_87: "b8[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_87: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.where.self(le_87, full_default, getitem_454);  le_87 = getitem_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_647: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.prims.convert_element_type.default(where_87, torch.float32);  where_87 = None
        sum_176: "f32[48][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_647, [0, 2, 3])
        convert_element_type_20: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32);  convolution_6 = None
        sub_442: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, unsqueeze_1422);  convert_element_type_20 = unsqueeze_1422 = None
        mul_1441: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_647, sub_442)
        sum_177: "f32[48][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1441, [0, 2, 3]);  mul_1441 = None
        mul_1442: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_176, 6.3775510204081635e-06)
        unsqueeze_1423: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1442, 0);  mul_1442 = None
        unsqueeze_1424: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1423, 2);  unsqueeze_1423 = None
        unsqueeze_1425: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1424, 3);  unsqueeze_1424 = None
        mul_1443: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_177, 6.3775510204081635e-06)
        mul_1444: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_1445: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1443, mul_1444);  mul_1443 = mul_1444 = None
        unsqueeze_1426: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1445, 0);  mul_1445 = None
        unsqueeze_1427: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1426, 2);  unsqueeze_1426 = None
        unsqueeze_1428: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1427, 3);  unsqueeze_1427 = None
        mul_1446: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, primals_42);  primals_42 = None
        unsqueeze_1429: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1446, 0);  mul_1446 = None
        unsqueeze_1430: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1429, 2);  unsqueeze_1429 = None
        unsqueeze_1431: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1430, 3);  unsqueeze_1430 = None
        mul_1447: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.mul.Tensor(sub_442, unsqueeze_1428);  sub_442 = unsqueeze_1428 = None
        sub_444: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_647, mul_1447);  convert_element_type_647 = mul_1447 = None
        sub_445: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.sub.Tensor(sub_444, unsqueeze_1425);  sub_444 = unsqueeze_1425 = None
        mul_1448: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.mul.Tensor(sub_445, unsqueeze_1431);  sub_445 = unsqueeze_1431 = None
        mul_1449: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_177, squeeze_19);  sum_177 = squeeze_19 = None
        convert_element_type_649: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1448, torch.bfloat16);  mul_1448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_87 = torch.ops.aten.convolution_backward.default(convert_element_type_649, getitem_12, convert_element_type_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_649 = convert_element_type_19 = None
        getitem_457: "bf16[128, 192, 35, 35][235200, 1, 6720, 192]cuda:0" = convolution_backward_87[0]
        getitem_458: "bf16[48, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_87[1];  convolution_backward_87 = None
        add_503: "bf16[128, 192, 35, 35][235200, 1, 6720, 192]cuda:0" = torch.ops.aten.add.Tensor(add_502, getitem_457);  add_502 = getitem_457 = None
        convert_element_type_650: "f32[48, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_458, torch.float32);  getitem_458 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_5: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, getitem_15)
        mul_35: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        unsqueeze_20: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
        unsqueeze_23: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_29: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        convert_element_type_18: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_5: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_18);  convert_element_type_18 = None
        le_88: "b8[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_88: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.where.self(le_88, full_default, slice_47);  le_88 = slice_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_651: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_88, torch.float32);  where_88 = None
        squeeze_15: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        unsqueeze_1432: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_1433: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1432, 2);  unsqueeze_1432 = None
        unsqueeze_1434: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1433, 3);  unsqueeze_1433 = None
        sum_178: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_651, [0, 2, 3])
        convert_element_type_17: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        sub_446: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_17, unsqueeze_1434);  convert_element_type_17 = unsqueeze_1434 = None
        mul_1450: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_651, sub_446)
        sum_179: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1450, [0, 2, 3]);  mul_1450 = None
        mul_1451: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_178, 6.3775510204081635e-06)
        unsqueeze_1435: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1451, 0);  mul_1451 = None
        unsqueeze_1436: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1435, 2);  unsqueeze_1435 = None
        unsqueeze_1437: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1436, 3);  unsqueeze_1436 = None
        mul_1452: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_179, 6.3775510204081635e-06)
        squeeze_16: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_1453: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_1454: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1452, mul_1453);  mul_1452 = mul_1453 = None
        unsqueeze_1438: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1454, 0);  mul_1454 = None
        unsqueeze_1439: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1438, 2);  unsqueeze_1438 = None
        unsqueeze_1440: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1439, 3);  unsqueeze_1439 = None
        mul_1455: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  primals_36 = None
        unsqueeze_1441: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1455, 0);  mul_1455 = None
        unsqueeze_1442: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1441, 2);  unsqueeze_1441 = None
        unsqueeze_1443: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1442, 3);  unsqueeze_1442 = None
        mul_1456: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_446, unsqueeze_1440);  sub_446 = unsqueeze_1440 = None
        sub_448: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_651, mul_1456);  convert_element_type_651 = mul_1456 = None
        sub_449: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_448, unsqueeze_1437);  sub_448 = unsqueeze_1437 = None
        mul_1457: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_449, unsqueeze_1443);  sub_449 = unsqueeze_1443 = None
        mul_1458: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_179, squeeze_16);  sum_179 = squeeze_16 = None
        convert_element_type_653: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1457, torch.bfloat16);  mul_1457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_88 = torch.ops.aten.convolution_backward.default(convert_element_type_653, getitem_12, convert_element_type_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_653 = getitem_12 = convert_element_type_16 = None
        getitem_460: "bf16[128, 192, 35, 35][235200, 1, 6720, 192]cuda:0" = convolution_backward_88[0]
        getitem_461: "bf16[64, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_88[1];  convolution_backward_88 = None
        add_504: "bf16[128, 192, 35, 35][235200, 1, 6720, 192]cuda:0" = torch.ops.aten.add.Tensor(add_503, getitem_460);  add_503 = getitem_460 = None
        convert_element_type_654: "f32[64, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_461, torch.float32);  getitem_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:399 in forward_preaux, code: x = self.Pool2(x)  # N x 192 x 35 x 35
        full_default_91: "f32[24576, 5041][5041, 1]cuda:0" = torch.ops.aten.full.default([24576, 5041], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        clone_7: "bf16[128, 192, 35, 35][235200, 1225, 35, 1]cuda:0" = torch.ops.aten.clone.default(add_504, memory_format = torch.contiguous_format);  add_504 = None
        view_9: "bf16[24576, 1225][1225, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [24576, 1225]);  clone_7 = None
        _low_memory_max_pool_offsets_to_indices_1: "i64[128, 192, 35, 35][235200, 1, 6720, 192]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_13, [3, 3], [71, 71], [2, 2], [0, 0], [1, 1]);  getitem_13 = None
        clone_8: "i64[128, 192, 35, 35][235200, 1225, 35, 1]cuda:0" = torch.ops.aten.clone.default(_low_memory_max_pool_offsets_to_indices_1, memory_format = torch.contiguous_format);  _low_memory_max_pool_offsets_to_indices_1 = None
        view_10: "i64[24576, 1225][1225, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [24576, 1225]);  clone_8 = None
        convert_element_type_655: "f32[24576, 1225][1225, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None
        scatter_add_2: "f32[24576, 5041][5041, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_91, 1, view_10, convert_element_type_655);  full_default_91 = view_10 = convert_element_type_655 = None
        view_11: "f32[128, 192, 71, 71][967872, 5041, 71, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add_2, [128, 192, 71, 71]);  scatter_add_2 = None
        convert_element_type_656: "bf16[128, 192, 71, 71][967872, 5041, 71, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.bfloat16);  view_11 = None
        clone_9: "bf16[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.clone.default(convert_element_type_656, memory_format = torch.channels_last);  convert_element_type_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_4: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
        mul_28: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        unsqueeze_16: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_19: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_24: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        convert_element_type_15: "bf16[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_4: "bf16[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_15);  convert_element_type_15 = None
        le_89: "b8[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_89: "bf16[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.where.self(le_89, full_default, clone_9);  le_89 = clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_657: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_89, torch.float32);  where_89 = None
        squeeze_12: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        unsqueeze_1444: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_1445: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1444, 2);  unsqueeze_1444 = None
        unsqueeze_1446: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1445, 3);  unsqueeze_1445 = None
        sum_180: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_657, [0, 2, 3])
        convert_element_type_14: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32);  convolution_4 = None
        sub_450: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_14, unsqueeze_1446);  convert_element_type_14 = unsqueeze_1446 = None
        mul_1459: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_657, sub_450)
        sum_181: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1459, [0, 2, 3]);  mul_1459 = None
        mul_1460: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_180, 1.5497917079944455e-06)
        unsqueeze_1447: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1460, 0);  mul_1460 = None
        unsqueeze_1448: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1447, 2);  unsqueeze_1447 = None
        unsqueeze_1449: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1448, 3);  unsqueeze_1448 = None
        mul_1461: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_181, 1.5497917079944455e-06)
        squeeze_13: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_1462: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_1463: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1461, mul_1462);  mul_1461 = mul_1462 = None
        unsqueeze_1450: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1463, 0);  mul_1463 = None
        unsqueeze_1451: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1450, 2);  unsqueeze_1450 = None
        unsqueeze_1452: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1451, 3);  unsqueeze_1451 = None
        mul_1464: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_1453: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1464, 0);  mul_1464 = None
        unsqueeze_1454: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1453, 2);  unsqueeze_1453 = None
        unsqueeze_1455: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1454, 3);  unsqueeze_1454 = None
        mul_1465: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_450, unsqueeze_1452);  sub_450 = unsqueeze_1452 = None
        sub_452: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_657, mul_1465);  convert_element_type_657 = mul_1465 = None
        sub_453: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_452, unsqueeze_1449);  sub_452 = unsqueeze_1449 = None
        mul_1466: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_453, unsqueeze_1455);  sub_453 = unsqueeze_1455 = None
        mul_1467: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_181, squeeze_13);  sum_181 = squeeze_13 = None
        convert_element_type_659: "bf16[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1466, torch.bfloat16);  mul_1466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_89 = torch.ops.aten.convolution_backward.default(convert_element_type_659, relu_3, convert_element_type_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_659 = convert_element_type_13 = None
        getitem_463: "bf16[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = convolution_backward_89[0]
        getitem_464: "bf16[192, 80, 3, 3][720, 1, 240, 80]cuda:0" = convolution_backward_89[1];  convolution_backward_89 = None
        convert_element_type_660: "f32[192, 80, 3, 3][720, 1, 240, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_464, torch.float32);  getitem_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_90: "b8[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_90: "bf16[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.aten.where.self(le_90, full_default, getitem_463);  le_90 = getitem_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_661: "f32[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.prims.convert_element_type.default(where_90, torch.float32);  where_90 = None
        sum_182: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_661, [0, 2, 3])
        convert_element_type_11: "f32[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        sub_454: "f32[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_11, unsqueeze_1458);  convert_element_type_11 = unsqueeze_1458 = None
        mul_1468: "f32[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_661, sub_454)
        sum_183: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1468, [0, 2, 3]);  mul_1468 = None
        mul_1469: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_182, 1.4660349033589792e-06)
        unsqueeze_1459: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1469, 0);  mul_1469 = None
        unsqueeze_1460: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1459, 2);  unsqueeze_1459 = None
        unsqueeze_1461: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1460, 3);  unsqueeze_1460 = None
        mul_1470: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_183, 1.4660349033589792e-06)
        mul_1471: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_1472: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1470, mul_1471);  mul_1470 = mul_1471 = None
        unsqueeze_1462: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1472, 0);  mul_1472 = None
        unsqueeze_1463: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1462, 2);  unsqueeze_1462 = None
        unsqueeze_1464: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1463, 3);  unsqueeze_1463 = None
        mul_1473: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_1465: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1473, 0);  mul_1473 = None
        unsqueeze_1466: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1465, 2);  unsqueeze_1465 = None
        unsqueeze_1467: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1466, 3);  unsqueeze_1466 = None
        mul_1474: "f32[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_454, unsqueeze_1464);  sub_454 = unsqueeze_1464 = None
        sub_456: "f32[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_661, mul_1474);  convert_element_type_661 = mul_1474 = None
        sub_457: "f32[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_456, unsqueeze_1461);  sub_456 = unsqueeze_1461 = None
        mul_1475: "f32[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_457, unsqueeze_1467);  sub_457 = unsqueeze_1467 = None
        mul_1476: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_183, squeeze_10);  sum_183 = squeeze_10 = None
        convert_element_type_663: "bf16[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1475, torch.bfloat16);  mul_1475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_90 = torch.ops.aten.convolution_backward.default(convert_element_type_663, getitem_6, convert_element_type_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_663 = getitem_6 = convert_element_type_10 = None
        getitem_466: "bf16[128, 64, 73, 73][341056, 1, 4672, 64]cuda:0" = convolution_backward_90[0]
        getitem_467: "bf16[80, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_90[1];  convolution_backward_90 = None
        convert_element_type_664: "f32[80, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_467, torch.float32);  getitem_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:396 in forward_preaux, code: x = self.Pool1(x)  # N x 64 x 73 x 73
        full_default_94: "f32[8192, 21609][21609, 1]cuda:0" = torch.ops.aten.full.default([8192, 21609], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        clone_10: "bf16[128, 64, 73, 73][341056, 5329, 73, 1]cuda:0" = torch.ops.aten.clone.default(getitem_466, memory_format = torch.contiguous_format);  getitem_466 = None
        view_12: "bf16[8192, 5329][5329, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [8192, 5329]);  clone_10 = None
        _low_memory_max_pool_offsets_to_indices: "i64[128, 64, 73, 73][341056, 1, 4672, 64]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_7, [3, 3], [147, 147], [2, 2], [0, 0], [1, 1]);  getitem_7 = None
        clone_11: "i64[128, 64, 73, 73][341056, 5329, 73, 1]cuda:0" = torch.ops.aten.clone.default(_low_memory_max_pool_offsets_to_indices, memory_format = torch.contiguous_format);  _low_memory_max_pool_offsets_to_indices = None
        view_13: "i64[8192, 5329][5329, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [8192, 5329]);  clone_11 = None
        convert_element_type_665: "f32[8192, 5329][5329, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_12, torch.float32);  view_12 = None
        scatter_add_3: "f32[8192, 21609][21609, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_94, 1, view_13, convert_element_type_665);  full_default_94 = view_13 = convert_element_type_665 = None
        view_14: "f32[128, 64, 147, 147][1382976, 21609, 147, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add_3, [128, 64, 147, 147]);  scatter_add_3 = None
        convert_element_type_666: "bf16[128, 64, 147, 147][1382976, 21609, 147, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_14, torch.bfloat16);  view_14 = None
        clone_12: "bf16[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.clone.default(convert_element_type_666, memory_format = torch.channels_last);  convert_element_type_666 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_2: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_14: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        unsqueeze_8: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_11: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        convert_element_type_9: "bf16[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_2: "bf16[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_9);  convert_element_type_9 = None
        le_91: "b8[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_91: "bf16[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.where.self(le_91, full_default, clone_12);  le_91 = clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_667: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_91, torch.float32);  where_91 = None
        squeeze_6: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        unsqueeze_1468: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_1469: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1468, 2);  unsqueeze_1468 = None
        unsqueeze_1470: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1469, 3);  unsqueeze_1469 = None
        sum_184: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_667, [0, 2, 3])
        convert_element_type_8: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        sub_458: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_8, unsqueeze_1470);  convert_element_type_8 = unsqueeze_1470 = None
        mul_1477: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_667, sub_458)
        sum_185: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1477, [0, 2, 3]);  mul_1477 = None
        mul_1478: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_184, 3.6153917349252627e-07)
        unsqueeze_1471: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1478, 0);  mul_1478 = None
        unsqueeze_1472: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1471, 2);  unsqueeze_1471 = None
        unsqueeze_1473: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1472, 3);  unsqueeze_1472 = None
        mul_1479: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_185, 3.6153917349252627e-07)
        squeeze_7: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_1480: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_1481: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1479, mul_1480);  mul_1479 = mul_1480 = None
        unsqueeze_1474: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1481, 0);  mul_1481 = None
        unsqueeze_1475: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1474, 2);  unsqueeze_1474 = None
        unsqueeze_1476: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1475, 3);  unsqueeze_1475 = None
        mul_1482: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_1477: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1482, 0);  mul_1482 = None
        unsqueeze_1478: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1477, 2);  unsqueeze_1477 = None
        unsqueeze_1479: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1478, 3);  unsqueeze_1478 = None
        mul_1483: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_458, unsqueeze_1476);  sub_458 = unsqueeze_1476 = None
        sub_460: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_667, mul_1483);  convert_element_type_667 = mul_1483 = None
        sub_461: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_460, unsqueeze_1473);  sub_460 = unsqueeze_1473 = None
        mul_1484: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_461, unsqueeze_1479);  sub_461 = unsqueeze_1479 = None
        mul_1485: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_185, squeeze_7);  sum_185 = squeeze_7 = None
        convert_element_type_669: "bf16[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1484, torch.bfloat16);  mul_1484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_91 = torch.ops.aten.convolution_backward.default(convert_element_type_669, relu_1, convert_element_type_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_669 = convert_element_type_7 = None
        getitem_469: "bf16[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = convolution_backward_91[0]
        getitem_470: "bf16[64, 32, 3, 3][288, 1, 96, 32]cuda:0" = convolution_backward_91[1];  convolution_backward_91 = None
        convert_element_type_670: "f32[64, 32, 3, 3][288, 1, 96, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_470, torch.float32);  getitem_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_92: "b8[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_92: "bf16[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.aten.where.self(le_92, full_default, getitem_469);  le_92 = getitem_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_671: "f32[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.prims.convert_element_type.default(where_92, torch.float32);  where_92 = None
        sum_186: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_671, [0, 2, 3])
        convert_element_type_5: "f32[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        sub_462: "f32[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_5, unsqueeze_1482);  convert_element_type_5 = unsqueeze_1482 = None
        mul_1486: "f32[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_671, sub_462)
        sum_187: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1486, [0, 2, 3]);  mul_1486 = None
        mul_1487: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_186, 3.6153917349252627e-07)
        unsqueeze_1483: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1487, 0);  mul_1487 = None
        unsqueeze_1484: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1483, 2);  unsqueeze_1483 = None
        unsqueeze_1485: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1484, 3);  unsqueeze_1484 = None
        mul_1488: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_187, 3.6153917349252627e-07)
        mul_1489: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_1490: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1488, mul_1489);  mul_1488 = mul_1489 = None
        unsqueeze_1486: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1490, 0);  mul_1490 = None
        unsqueeze_1487: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1486, 2);  unsqueeze_1486 = None
        unsqueeze_1488: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1487, 3);  unsqueeze_1487 = None
        mul_1491: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_1489: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1491, 0);  mul_1491 = None
        unsqueeze_1490: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1489, 2);  unsqueeze_1489 = None
        unsqueeze_1491: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1490, 3);  unsqueeze_1490 = None
        mul_1492: "f32[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_462, unsqueeze_1488);  sub_462 = unsqueeze_1488 = None
        sub_464: "f32[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_671, mul_1492);  convert_element_type_671 = mul_1492 = None
        sub_465: "f32[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.aten.sub.Tensor(sub_464, unsqueeze_1485);  sub_464 = unsqueeze_1485 = None
        mul_1493: "f32[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_465, unsqueeze_1491);  sub_465 = unsqueeze_1491 = None
        mul_1494: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_187, squeeze_4);  sum_187 = squeeze_4 = None
        convert_element_type_673: "bf16[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1493, torch.bfloat16);  mul_1493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_92 = torch.ops.aten.convolution_backward.default(convert_element_type_673, relu, convert_element_type_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_673 = convert_element_type_4 = None
        getitem_472: "bf16[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = convolution_backward_92[0]
        getitem_473: "bf16[32, 32, 3, 3][288, 1, 96, 32]cuda:0" = convolution_backward_92[1];  convolution_backward_92 = None
        convert_element_type_674: "f32[32, 32, 3, 3][288, 1, 96, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_473, torch.float32);  getitem_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_93: "b8[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_93: "bf16[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.aten.where.self(le_93, full_default, getitem_472);  le_93 = full_default = getitem_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_675: "f32[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.prims.convert_element_type.default(where_93, torch.float32);  where_93 = None
        sum_188: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_675, [0, 2, 3])
        convert_element_type_2: "f32[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        sub_466: "f32[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_1494);  convert_element_type_2 = unsqueeze_1494 = None
        mul_1495: "f32[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_675, sub_466)
        sum_189: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1495, [0, 2, 3]);  mul_1495 = None
        mul_1496: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_188, 3.5189856312778704e-07)
        unsqueeze_1495: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1496, 0);  mul_1496 = None
        unsqueeze_1496: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1495, 2);  unsqueeze_1495 = None
        unsqueeze_1497: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1496, 3);  unsqueeze_1496 = None
        mul_1497: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_189, 3.5189856312778704e-07)
        mul_1498: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_1499: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1497, mul_1498);  mul_1497 = mul_1498 = None
        unsqueeze_1498: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1499, 0);  mul_1499 = None
        unsqueeze_1499: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1498, 2);  unsqueeze_1498 = None
        unsqueeze_1500: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1499, 3);  unsqueeze_1499 = None
        mul_1500: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_1501: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1500, 0);  mul_1500 = None
        unsqueeze_1502: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1501, 2);  unsqueeze_1501 = None
        unsqueeze_1503: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1502, 3);  unsqueeze_1502 = None
        mul_1501: "f32[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_466, unsqueeze_1500);  sub_466 = unsqueeze_1500 = None
        sub_468: "f32[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_675, mul_1501);  convert_element_type_675 = mul_1501 = None
        sub_469: "f32[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.aten.sub.Tensor(sub_468, unsqueeze_1497);  sub_468 = unsqueeze_1497 = None
        mul_1502: "f32[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_469, unsqueeze_1503);  sub_469 = unsqueeze_1503 = None
        mul_1503: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_189, squeeze_1);  sum_189 = squeeze_1 = None
        convert_element_type_677: "bf16[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1502, torch.bfloat16);  mul_1502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_93 = torch.ops.aten.convolution_backward.default(convert_element_type_677, convert_element_type_1, convert_element_type, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  convert_element_type_677 = convert_element_type_1 = convert_element_type = None
        getitem_476: "bf16[32, 3, 3, 3][27, 1, 9, 3]cuda:0" = convolution_backward_93[1];  convolution_backward_93 = None
        convert_element_type_678: "f32[32, 3, 3, 3][27, 1, 9, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_476, torch.float32);  getitem_476 = None
        return (convert_element_type_678, None, None, None, None, mul_1503, sum_188, convert_element_type_674, None, None, None, mul_1494, sum_186, convert_element_type_670, None, None, None, mul_1485, sum_184, convert_element_type_664, None, None, None, mul_1476, sum_182, convert_element_type_660, None, None, None, mul_1467, sum_180, convert_element_type_654, None, None, None, mul_1458, sum_178, convert_element_type_650, None, None, None, mul_1449, sum_176, convert_element_type_646, None, None, None, mul_1440, sum_174, convert_element_type_642, None, None, None, mul_1431, sum_172, convert_element_type_638, None, None, None, mul_1422, sum_170, convert_element_type_634, None, None, None, mul_1413, sum_168, convert_element_type_630, None, None, None, mul_1404, sum_166, convert_element_type_626, None, None, None, mul_1395, sum_164, convert_element_type_622, None, None, None, mul_1386, sum_162, convert_element_type_618, None, None, None, mul_1377, sum_160, convert_element_type_614, None, None, None, mul_1368, sum_158, convert_element_type_610, None, None, None, mul_1359, sum_156, convert_element_type_606, None, None, None, mul_1350, sum_154, convert_element_type_602, None, None, None, mul_1341, sum_152, convert_element_type_598, None, None, None, mul_1332, sum_150, convert_element_type_594, None, None, None, mul_1323, sum_148, convert_element_type_590, None, None, None, mul_1314, sum_146, convert_element_type_586, None, None, None, mul_1305, sum_144, convert_element_type_582, None, None, None, mul_1296, sum_142, convert_element_type_578, None, None, None, mul_1287, sum_140, convert_element_type_574, None, None, None, mul_1278, sum_138, convert_element_type_570, None, None, None, mul_1269, sum_136, convert_element_type_566, None, None, None, mul_1260, sum_134, convert_element_type_562, None, None, None, mul_1251, sum_132, convert_element_type_558, None, None, None, mul_1242, sum_130, convert_element_type_552, None, None, None, mul_1233, sum_128, convert_element_type_548, None, None, None, mul_1224, sum_126, convert_element_type_544, None, None, None, mul_1215, sum_124, convert_element_type_540, None, None, None, mul_1206, sum_122, convert_element_type_536, None, None, None, mul_1197, sum_120, convert_element_type_532, None, None, None, mul_1188, sum_118, convert_element_type_528, None, None, None, mul_1179, sum_116, convert_element_type_524, None, None, None, mul_1170, sum_114, convert_element_type_520, None, None, None, mul_1161, sum_112, convert_element_type_516, None, None, None, mul_1152, sum_110, convert_element_type_512, None, None, None, mul_1143, sum_108, convert_element_type_508, None, None, None, mul_1134, sum_106, convert_element_type_504, None, None, None, mul_1125, sum_104, convert_element_type_500, None, None, None, mul_1116, sum_102, convert_element_type_496, None, None, None, mul_1107, sum_100, convert_element_type_492, None, None, None, mul_1098, sum_98, convert_element_type_488, None, None, None, mul_1089, sum_96, convert_element_type_484, None, None, None, mul_1080, sum_94, convert_element_type_480, None, None, None, mul_1071, sum_92, convert_element_type_476, None, None, None, mul_1062, sum_90, convert_element_type_472, None, None, None, mul_1053, sum_88, convert_element_type_468, None, None, None, mul_1044, sum_86, convert_element_type_464, None, None, None, mul_1035, sum_84, convert_element_type_460, None, None, None, mul_1026, sum_82, convert_element_type_456, None, None, None, mul_1017, sum_80, convert_element_type_452, None, None, None, mul_1008, sum_78, convert_element_type_448, None, None, None, mul_999, sum_76, convert_element_type_444, None, None, None, mul_990, sum_74, convert_element_type_440, None, None, None, mul_981, sum_72, convert_element_type_436, None, None, None, mul_972, sum_70, convert_element_type_432, None, None, None, mul_963, sum_68, convert_element_type_428, None, None, None, mul_954, sum_66, convert_element_type_424, None, None, None, mul_945, sum_64, convert_element_type_420, None, None, None, mul_936, sum_62, convert_element_type_416, None, None, None, mul_927, sum_60, convert_element_type_412, None, None, None, mul_918, sum_58, convert_element_type_408, None, None, None, mul_909, sum_56, convert_element_type_404, None, None, None, mul_900, sum_54, convert_element_type_400, None, None, None, mul_891, sum_52, convert_element_type_396, None, None, None, mul_882, sum_50, convert_element_type_392, None, None, None, mul_873, sum_48, convert_element_type_388, None, None, None, mul_864, sum_46, convert_element_type_384, None, None, None, mul_855, sum_44, convert_element_type_380, None, None, None, mul_846, sum_42, convert_element_type_376, None, None, None, mul_837, sum_40, convert_element_type_372, None, None, None, mul_828, sum_38, convert_element_type_366, None, None, None, mul_819, sum_36, convert_element_type_362, None, None, None, mul_810, sum_34, convert_element_type_358, None, None, None, mul_801, sum_32, convert_element_type_354, None, None, None, mul_792, sum_30, convert_element_type_350, None, None, None, mul_783, sum_28, convert_element_type_346, None, None, None, mul_774, sum_26, convert_element_type_342, None, None, None, mul_765, sum_24, convert_element_type_338, None, None, None, mul_756, sum_22, convert_element_type_334, None, None, None, mul_747, sum_20, convert_element_type_330, None, None, None, mul_738, sum_18, convert_element_type_326, None, None, None, mul_729, sum_16, convert_element_type_322, None, None, None, mul_720, sum_14, convert_element_type_318, None, None, None, mul_711, sum_12, convert_element_type_314, None, None, None, mul_702, sum_10, convert_element_type_310, None, None, None, mul_693, sum_8, convert_element_type_306, None, None, None, mul_684, sum_6, convert_element_type_302, None, None, None, mul_675, sum_4, convert_element_type_298, None, None, None, mul_666, sum_2, convert_element_type_293, convert_element_type_294)
